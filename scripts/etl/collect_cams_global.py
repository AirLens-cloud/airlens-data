#!/usr/bin/env python3
"""CAMS global atmospheric composition forecast 수집기 — 가스(O3/NO2/SO2/CO) + PM.

설계 배경(비공개 내부 노트 2026-07-16 — 동작 정의는 코드·테스트·계약):
§"무료 글로벌 소스 판정" — "CAMS global atmospheric composition forecasts ... 전 세계 가스와
모델 기본값의 정본". 이 스크립트는 Copernicus ADS(Atmosphere Data Store) 의
`cams-global-atmospheric-composition-forecasts` dataset 을 최신 0h step(analysis)으로
받는다.

기존 `models/pipeline/fetchers_cams.py` 와 **같은 인증 관례**(`CDS_API_URL`/`CDS_API_KEY`,
무료 ADS 계정)를 재사용하지만 dataset 이 다르다 — 그 파일은 EAC4 *재분석*(historical
monthly, 학습용) 을, 이 스크립트는 *근실시간 예보 0h analysis*(mac 파이프라인용) 를 받는다.
둘 다 같은 키를 쓰되 서로 다른 목적의 별도 호출이다.

`cdsapi` 클라이언트(신규 pip dep)를 쓴다 — 형제 수집기들(`collect_noaa_aq.py` 등)의
"pip 신규 deps 0" 관례에서 의도적으로 벗어난 선택이다. ADS 의 비동기 job 제출/폴링/다운로드
REST 프로토콜을 직접 구현하면 그 계약을 잘못 추정할 위험이 크고, `cdsapi` 는 Copernicus
공식 클라이언트이자 이미 이 저장소(`models/pipeline/fetchers_cams.py`)에서 검증된 경로다.
`import cdsapi` 는 `main()` 경로에서만 lazy 로 일어난다 — 순수 함수(요청 조립·단위 변환·
스키마 조립)는 이 의존성 없이 임포트·테스트 가능.

CAMS 가스(O3/NO2/SO2/CO) 는 원자료가 질량혼합비(kg/kg) 라 지표기압(sp)+2m기온(2t) 을
같은 요청에 포함해 `mac_aq_adapter.air_density_kg_m3()` 로 변환한다. PM2.5/PM10 은 이미
질량농도(kg/m³). 변수 short name(pm2p5/pm10/go3/no2/so2/co/sp/2t) 은 `CAMS_VARIABLES`/
`AUX_VARIABLES` 테이블 하나에 모아뒀다. ADS 폼과의 대조는 2026-07-28 에 카탈로그
form.json/constraints.json 으로 마쳤다 — 변수는 **Single level / Multi level 두 그룹**이고
가스 4종은 후자라 `model_level` 없이는 응답에서 조용히 빠진다(§MULTI_LEVEL_POLLUTANTS).

정직성 (형제 수집기와 동일 원칙):
  - 요청/다운로드/디코딩 실패, 또는 6개 오염물질 중 하나라도 빠지면 exit 1 — 0 으로
    메우지 않고 부분 snapshot 을 발행하지 않는다.
  - 성공한 레코드만 임시 파일에 쓰고 `os.replace` 로 교체 — 실패 시 기존 last-good 파일은
    절대 건드리지 않는다.
  - 이 스크립트는 NOAA collector(`collect_gefs_chem_global.py`)와 완전히 독립된
    프로세스다. 한쪽이 실패해도 다른 쪽 실행에 영향을 주지 않는다.
  - `CDS_API_KEY` 값은 로그에 출력하지 않는다 — 이름으로만 참조한다(§2 secret 가드).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_gfs_wind import build_dense  # noqa: E402
import mac_aq_adapter as adapter  # noqa: E402

DATASET = "cams-global-atmospheric-composition-forecasts"
GRID_RES_DEG = float(os.environ.get("MAC_CAMS_RES_DEG", "0.4"))  # CAMS native 0.4°
OUTPUT_PATH = os.environ.get("MAC_CAMS_OUTPUT_PATH", "cams-global-snapshot.json")

ATTRIBUTION = "Contains modified Copernicus Atmosphere Monitoring Service information"

CYCLE_HOURS = (12, 0)  # design SOT §"스케줄": 하루 2회 00/12 UTC
# ECMWF 가 보장하는 가용 시각은 00z→10:00Z, 12z→22:00Z 다(Copernicus Knowledge Base
# "CAMS: Global atmospheric composition forecast data documentation"). 7h 로 두었을 땐
# 2026-10-09~17 run 38개 중 9개가 아직 안 올라온 cycle 을 요청해 400("Request has not
# produced a valid combination of values")으로 실패했다 — 실패는 모두 7.4~9.6h, 성공은 10h 이상.
PUBLISH_LATENCY_HOURS = 10
FRESHNESS_CUTOFF_HOURS = 12  # CAMS global 은 12시간 주기(design SOT §"스케줄")

# pollutant_key -> (ADS variable name, GRIB short name, kind)
# kind: "mass" = 이미 질량농도(kg/m3), "mixing" = 질량혼합비(kg/kg, 밀도 변환 필요)
CAMS_VARIABLES: dict[str, tuple[str, str, str]] = {
    "pm25": ("particulate_matter_2.5um", "pm2p5", "mass"),
    "pm10": ("particulate_matter_10um", "pm10", "mass"),
    "o3": ("ozone", "go3", "mixing"),
    "no2": ("nitrogen_dioxide", "no2", "mixing"),
    "so2": ("sulphur_dioxide", "so2", "mixing"),
    "co": ("carbon_monoxide", "co", "mixing"),
}
AUX_VARIABLES: dict[str, str] = {  # 밀도 변환용 — pollutants 에는 노출 안 함
    "surface_pressure": "sp",
    "2m_temperature": "2t",
}

EXPECTED_POLLUTANT_COUNT = len(CAMS_VARIABLES)

# ADS 는 이 데이터셋의 변수를 "Single level" / "Multi level" 두 그룹으로 나눈다.
# PM2.5/PM10/sp/2t 는 Single level 이라 레벨 키 없이 받지만, 가스 4종(O3/NO2/SO2/CO)은
# Multi level 이라 `model_level` 지정이 **필수**다. 레벨 없이 같이 요청하면 ADS 는
# 에러를 내지 않고 가스만 조용히 빼고 응답한다 — 그게 이 수집기가 만성 실패하던 원인이다
# (2026-07-28 run 30356957266: 요청 8변수, 도착 shortName=['2t','pm10','pm2p5','sp']).
#
# 두 그룹은 한 요청에 섞을 수도 없다. 카탈로그 constraints.json 261블록 중
# pm2.5 와 ozone 이 함께 든 블록이 **0건**이다 — 그래서 요청을 둘로 쪼갠다.
# 재확인: `curl <collection>/constraints.json` (아래 SURFACE_MODEL_LEVEL 주석 참조)
MULTI_LEVEL_POLLUTANTS = frozenset({"o3", "no2", "so2", "co"})

# L137 격자의 최하단 = 지표 바로 위(~10m). 우리 사이클(00/12 UTC · forecast ·
# leadtime 0)에서 가스 4종 전부에 137 이 허용됨을 constraints.json 으로 확인했다.
# `pressure_level: 1000` 도 열려 있으나 고지대에서 1000hPa 는 지면 아래라
# 외삽값이 된다 — 전지구 지표 산출물에는 model level 이 맞다.
SURFACE_MODEL_LEVEL = "137"


# ────────────────────────── pure helpers (테스트 대상) ──────────────────────────

def pick_latest_cams_cycle(now: datetime) -> tuple[str, str]:
    """00/12 UTC cycle 중 발행 지연(PUBLISH_LATENCY_HOURS)을 감안해 가장 최근 것.

    예: 09:59Z 시점엔 어제 12z(21h59m 전, 발행 완료 확실)를 고른다 — 오늘 00z(9h59m 전) 는
    보장 시각(10:00Z) 전이라 아직 ADS 에 없을 수 있어 건너뛴다.
    """
    for day_offset in range(0, 3):
        d = now - timedelta(days=day_offset)
        for hh in CYCLE_HOURS:
            cycle_dt = d.replace(hour=hh, minute=0, second=0, microsecond=0)
            if cycle_dt > now:
                continue
            age_hours = (now - cycle_dt).total_seconds() / 3600
            if age_hours >= PUBLISH_LATENCY_HOURS:
                return cycle_dt.strftime("%Y-%m-%d"), f"{hh:02d}:00"
    raise RuntimeError("no CAMS cycle old enough to be published (today..2d ago)")


def _base_request(date_str: str, time_str: str) -> dict:
    """두 요청이 공유하는 축 — 레벨 키와 변수 목록만 호출자가 채운다."""
    return {
        "date": [date_str],
        "time": [time_str],
        "leadtime_hour": ["0"],
        "type": "forecast",
        "format": "grib",
        "area": [90, -180, -90, 180],  # N, W, S, E — 전 세계
    }


def build_single_level_request(date_str: str, time_str: str) -> dict:
    """Single level 그룹 — PM2.5/PM10 + 밀도 변환용 aux(sp/2t). 레벨 키 없음."""
    variables = [
        ads_var for key, (ads_var, _sn, _kind) in CAMS_VARIABLES.items()
        if key not in MULTI_LEVEL_POLLUTANTS
    ] + list(AUX_VARIABLES.keys())
    return {**_base_request(date_str, time_str), "variable": variables}


def build_multi_level_request(date_str: str, time_str: str) -> dict:
    """Multi level 그룹 — 가스 4종. `model_level` 없이는 응답에서 조용히 빠진다."""
    variables = [
        ads_var for key, (ads_var, _sn, _kind) in CAMS_VARIABLES.items()
        if key in MULTI_LEVEL_POLLUTANTS
    ]
    return {
        **_base_request(date_str, time_str),
        "variable": variables,
        "model_level": [SURFACE_MODEL_LEVEL],
    }


def parse_grib_inventory(stdout: str) -> list[tuple[str, str, int]]:
    """`grib_ls -p shortName,typeOfLevel,level` 출력에서 메시지 목록만 뽑는다.

    grib_ls 는 파일명 줄 / 컬럼 헤더 / 데이터 줄 / "N of M messages…" 꼬리를 섞어 낸다.
    3 필드이면서 마지막이 정수인 줄만 데이터로 본다.
    """
    rows: list[tuple[str, str, int]] = []
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        short_name, type_of_level, level_str = parts
        try:
            level = int(level_str)
        except ValueError:
            continue  # 컬럼 헤더("shortName typeOfLevel level")
        rows.append((short_name, type_of_level, level))
    return rows


def resolve_single_level(
    inventory: list[tuple[str, str, int]],
    short_name: str,
) -> tuple[str, int]:
    """인벤토리에서 `short_name` 의 레벨을 **하나로** 확정한다. 아니면 raise.

    두 실패를 모두 정보와 함께 세운다:

    - **부재** — 기존 메시지는 `"go3: decode 0 rows"` 였는데, 그건 무엇이 대신
      들어왔는지를 안 알려줘서 원인 분류가 불가능했다. 여기서는 파일에 실제로 있는
      shortName 목록을 함께 낸다("요청한 것 vs 받은 것").
    - **다중 레벨** — 조용히 병합하지 않는다. 병합은 last-write-wins 라 엉뚱한 고도의
      값이 지표면 값으로 발행된다. 어떤 레벨을 쓸지는 호출부가 명시할 문제지
      디코더가 임의로 고를 문제가 아니다.
    """
    levels = sorted({(tol, lvl) for sn, tol, lvl in inventory if sn == short_name})
    if not levels:
        present = sorted({sn for sn, _tol, _lvl in inventory})
        raise RuntimeError(
            f"{short_name}: GRIB 에 해당 메시지 없음 — 발행 중단. "
            f"파일에 실제로 있는 shortName={present or '(없음)'}"
        )
    if len(levels) > 1:
        raise RuntimeError(
            f"{short_name}: 레벨이 {len(levels)}개 — {levels}. 레벨을 지정하지 않고 "
            "디코드하면 전 레벨이 한 스트림으로 이어져 last-write-wins 로 뭉개진다 "
            "(엉뚱한 고도 값이 지표면 값으로 발행). 호출부에서 레벨을 명시할 것."
        )
    return levels[0]


def decode_at_resolved_level(
    path: str,
    short_name: str,
    inventory: list[tuple[str, str, int]],
) -> list[tuple[float, float, float]]:
    """인벤토리로 레벨을 확정한 뒤 그 레벨만 디코드한다 — 유일한 디코드 진입점.

    `resolve_single_level()` 로 레벨을 *계산* 만 하고 디코드 호출엔 안 넘기는 반쪽
    수정이 가능해서(정확히 #1028 형태), 두 단계를 한 함수로 묶어 그 틈을 없앤다.
    """
    type_of_level, level = resolve_single_level(inventory, short_name)
    return decode_grib_by_shortname(
        path, short_name, type_of_level=type_of_level, level=level
    )


def grid_header_for(res_deg: float) -> dict:
    """la1=북단(90)/lo1=-180 관례(collect_gfs_wind 와 동일)."""
    nx = round(360.0 / res_deg)
    ny = round(180.0 / res_deg) + 1
    return {"nx": nx, "ny": ny, "la1": 90.0, "lo1": -180.0, "dx": res_deg, "dy": res_deg}


def convert_mass_rows(rows: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """PM2.5/PM10 — 이미 kg/m³ → µg/m³ (×1e9). 0 으로 메우지 않는다."""
    if not rows:
        raise ValueError("empty rows — 0 으로 메우지 않는다")
    return [(la, lo, adapter.mass_conc_kgm3_to_ugm3(v)) for la, lo, v in rows]


def convert_mixing_rows(
    rows: list[tuple[float, float, float]],
    sp_by_coord: dict[tuple[float, float], float],
    t2m_by_coord: dict[tuple[float, float], float],
) -> list[tuple[float, float, float]]:
    """가스(kg/kg) → µg/m³. 같은 좌표의 sp(Pa)/2t(K) 로 셀별 공기밀도를 구한다.

    보조 변수가 그 좌표에 없으면 그 셀은 raise — 0/평균으로 메우지 않는다(정직성).
    """
    if not rows:
        raise ValueError("empty rows — 0 으로 메우지 않는다")
    out = []
    for lat, lon, mixing_ratio in rows:
        key = (round(lat, 4), round(lon, 4))
        if key not in sp_by_coord or key not in t2m_by_coord:
            raise ValueError(f"missing sp/2t at ({lat},{lon}) — 밀도 변환 불가, 0 으로 메우지 않는다")
        density = adapter.air_density_kg_m3(sp_by_coord[key], t2m_by_coord[key])
        conc = adapter.mixing_ratio_to_ugm3(mixing_ratio, density)
        out.append((lat, lon, conc))
    return out


def rows_to_coord_map(rows: list[tuple[float, float, float]]) -> dict[tuple[float, float], float]:
    """(lat,lon,val) 리스트 → {(lat,lon): val} — convert_mixing_rows 의 sp/2t 조회용."""
    return {(round(la, 4), round(lo, 4)): v for la, lo, v in rows}


def build_pollutant_block(rows_ugm3: list[tuple[float, float, float]], header: dict,
                           pollutant_key: str) -> dict:
    ads_var, short_name, kind = CAMS_VARIABLES[pollutant_key]
    dense = build_dense(
        rows_ugm3, nx=header["nx"], ny=header["ny"],
        la1=header["la1"], lo1=header["lo1"], dx=header["dx"], dy=header["dy"],
    )
    conversion = ("GRIB kg/m3 x 1e9" if kind == "mass"
                  else "GRIB kg/kg x air_density(sp,2t) x 1e9")
    return {"unit": "ug/m3", "sourceVariable": short_name, "conversion": conversion, "data": dense}


def compute_expires_at(generated_at_iso: str, hours: float) -> str:
    dt = datetime.fromisoformat(generated_at_iso.replace("Z", "+00:00"))
    return (dt + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def assemble_reading(pollutant_blocks: dict[str, dict], header: dict, generated_at: str,
                      valid_at: str, age_hours: float) -> dict:
    """envelope + grid_reading 조립. 완전성 미달이면 assemble 전에 호출자가 이미 raise 한다."""
    quality = adapter.estimate_quality(
        pollutant_count=len(pollutant_blocks),
        expected_count=EXPECTED_POLLUTANT_COUNT,
        age_hours=age_hours,
    )
    envelope = adapter.build_envelope(
        kind="analysis",
        source="CAMS",
        source_version=f"cycle-{valid_at.replace('-', '').replace(':', '')}",
        generated_at=generated_at,
        observed_at=None,
        valid_at=valid_at,
        expires_at=compute_expires_at(generated_at, FRESHNESS_CUTOFF_HOURS),
        resolution_km=round(header["dx"] * 111.0, 1),
        attribution=ATTRIBUTION,
        quality=quality,
    )
    reading = adapter.build_grid_reading(
        envelope,
        {"nx": header["nx"], "ny": header["ny"], "lo1": header["lo1"], "la1": header["la1"],
         "dx": header["dx"], "dy": header["dy"]},
        pollutant_blocks,
    )
    errors = adapter.validate_grid_reading(reading)
    if errors:
        raise ValueError(f"assembled reading failed validation: {errors}")
    return reading


# ────────────────────────────── IO (네트워크/subprocess, 미테스트 — 형제 관례) ──────────────────────────────

def fetch_cams_grib(request: dict, target_path: str) -> None:
    """ADS 비동기 retrieve(submit→poll→download) — 프로토콜 세부는 cdsapi 에 위임한다.

    `CDS_API_URL`/`CDS_API_KEY` 는 이름으로만 참조 — 값은 여기서도, 어떤 로그에도
    출력하지 않는다. 둘 중 하나라도 미설정이면 KeyError → main() 이 잡아 exit 1.
    """
    import cdsapi  # lazy import — 신규 pip dep, 이 함수가 실제로 불릴 때만 필요

    client = cdsapi.Client(url=os.environ["CDS_API_URL"], key=os.environ["CDS_API_KEY"])
    client.retrieve(DATASET, request, target_path)


def read_grib_inventory(path: str) -> list[tuple[str, str, int]]:
    """eccodes `grib_ls` 로 파일에 **실제로 들어온** 메시지 목록을 읽는다.

    반환 = `(shortName, typeOfLevel, level)` 리스트 (파일 안 순서 그대로, 중복 유지).
    IO 라 미테스트 — 파싱은 `parse_grib_inventory()` 로 분리해 테스트한다.
    """
    out = subprocess.run(
        ["grib_ls", "-p", "shortName,typeOfLevel,level", path],
        capture_output=True, text=True, timeout=180, check=True,
    ).stdout
    return parse_grib_inventory(out)


def decode_grib_by_shortname(
    path: str,
    short_name: str,
    *,
    type_of_level: str | None = None,
    level: int | None = None,
) -> list[tuple[float, float, float]]:
    """eccodes `grib_get_data -w shortName=X` 로 다중 메시지 GRIB 에서 변수 하나만 추출.

    `type_of_level`/`level` 을 주면 `-w` 조건에 함께 걸어 **정확히 한 레벨**만 뽑는다.
    다중레벨 필드(가스 종은 model/pressure level 로 오는 경우가 있다)를 레벨 구분 없이
    뽑으면 137 레벨이 한 스트림으로 이어져 나오고, 뒤이은 `rows_to_coord_map()` 이
    last-write-wins 로 뭉갠다 — "0 rows" 보다 나쁜 **조용한 오염**(엉뚱한 고도의 농도를
    지표면 값으로 발행)이다. 그래서 호출부는 인벤토리로 레벨을 확정한 뒤 넘긴다.
    """
    where = f"shortName={short_name}"
    if type_of_level is not None:
        where += f",typeOfLevel={type_of_level}"
    if level is not None:
        where += f",level={level}"
    out = subprocess.run(
        ["grib_get_data", "-m", "9999", "-w", where, path],
        capture_output=True, text=True, timeout=180, check=True,
    ).stdout
    rows = []
    for line in out.split("\n"):
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            lat, lon, val = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue  # header "Latitude Longitude Value"
        if val == 9999.0:
            continue
        rows.append((lat, lon, val))
    if not rows:
        raise RuntimeError(f"{short_name}: decode 0 rows — 발행 중단")
    return rows


def main() -> int:
    grib_paths: dict[str, str] = {}
    try:
        now = datetime.now(timezone.utc)
        date_str, time_str = pick_latest_cams_cycle(now)
        valid_at = f"{date_str}T{time_str}:00Z"
        generated_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        valid_dt = datetime.fromisoformat(valid_at.replace("Z", "+00:00"))
        age_hours = (now - valid_dt).total_seconds() / 3600

        # 두 그룹은 ADS 제약상 한 요청에 못 섞는다(§MULTI_LEVEL_POLLUTANTS) — 두 번 받는다.
        # 둘 다 성공해야 발행한다. 가스 요청이 실패하면 PM 만으로 부분 발행하지 않는다.
        requests = {
            "single": build_single_level_request(date_str, time_str),
            "multi": build_multi_level_request(date_str, time_str),
        }
        inventories: dict[str, list[tuple[str, str, int]]] = {}
        for group, request in requests.items():
            with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
                grib_paths[group] = f.name
            fetch_cams_grib(request, grib_paths[group])
            # 디코드 전에 파일에 뭐가 들어왔는지 먼저 읽는다 — 레벨을 확정해서 넘겨야
            # 다중레벨 필드가 조용히 뭉개지지 않고, 부재 시에도 "뭐가 대신 왔는지" 를
            # 에러에 담을 수 있다.
            inventories[group] = read_grib_inventory(grib_paths[group])

        def _decode(short_name: str, group: str) -> list[tuple[float, float, float]]:
            return decode_at_resolved_level(
                grib_paths[group], short_name, inventories[group]
            )

        header = grid_header_for(GRID_RES_DEG)
        sp_rows = _decode(AUX_VARIABLES["surface_pressure"], "single")
        t2m_rows = _decode(AUX_VARIABLES["2m_temperature"], "single")
        sp_by_coord = rows_to_coord_map(sp_rows)
        t2m_by_coord = rows_to_coord_map(t2m_rows)

        blocks: dict[str, dict] = {}
        for key, (_ads_var, short_name, kind) in CAMS_VARIABLES.items():
            group = "multi" if key in MULTI_LEVEL_POLLUTANTS else "single"
            rows = _decode(short_name, group)
            if kind == "mass":
                rows_ugm3 = convert_mass_rows(rows)
            else:
                rows_ugm3 = convert_mixing_rows(rows, sp_by_coord, t2m_by_coord)
            blocks[key] = build_pollutant_block(rows_ugm3, header, key)

        if len(blocks) != EXPECTED_POLLUTANT_COUNT:
            raise ValueError(
                f"incomplete pollutant set: {sorted(blocks)} "
                f"(expected {EXPECTED_POLLUTANT_COUNT}) — 부분 snapshot 은 발행하지 않는다"
            )

        reading = assemble_reading(blocks, header, generated_at, valid_at, age_hours)
    except Exception as e:  # noqa: BLE001 — outage 는 fail-loud, last-good 은 안 건드림
        print(f"ERROR: CAMS collection failed — {e}", file=sys.stderr)
        print("  기존 snapshot(있다면) 은 그대로 유지한다.", file=sys.stderr)
        return 1
    finally:
        for path in grib_paths.values():
            if os.path.exists(path):
                os.unlink(path)

    tmp_path = f"{OUTPUT_PATH}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(reading, f, separators=(",", ":"))
    os.replace(tmp_path, OUTPUT_PATH)
    print(f"Done: CAMS {date_str} {time_str}Z -> {OUTPUT_PATH} "
          f"(pollutants={sorted(blocks)}, grade={reading['quality']['grade']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
