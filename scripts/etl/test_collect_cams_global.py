"""collect_cams_global.py 단위 테스트 (AAA). 네트워크/cdsapi/subprocess 없음 — fixture mock.

outage/부분실패 경로는 monkeypatch 로 시뮬레이트 — 실 ADS 호출 없이
"last-good snapshot 보존 + fail-loud" 계약을 검증한다.
"""
from datetime import datetime, timezone

import pytest

import collect_cams_global as m
import mac_aq_adapter as adapter

UTC = timezone.utc


# ────────────────────────── pure helpers ──────────────────────────

def test_pick_latest_cams_cycle_skips_cycle_younger_than_publish_latency():
    # Arrange — 09:59Z: 오늘 00z 는 9h59m 전(< 10h latency) → 스킵, 어제 12z(21h59m)로
    now = datetime(2026, 7, 16, 9, 59, tzinfo=UTC)
    # Act
    date_str, time_str = m.pick_latest_cams_cycle(now)
    # Assert
    assert (date_str, time_str) == ("2026-07-15", "12:00")


def test_pick_latest_cams_cycle_accepts_cycle_exactly_at_latency_boundary():
    # Arrange — 10:00Z: 오늘 00z 는 정확히 10h 전 → 경계값 포함(>=)
    now = datetime(2026, 7, 16, 10, 0, tzinfo=UTC)
    # Act
    date_str, time_str = m.pick_latest_cams_cycle(now)
    # Assert
    assert (date_str, time_str) == ("2026-07-16", "00:00")


def test_two_requests_together_cover_every_declared_variable_exactly_once():
    """요청 분리(single/multi)가 변수를 흘리지도, 중복 요청하지도 않는지.

    기대값을 손으로 나열하지 않고 `CAMS_VARIABLES`/`AUX_VARIABLES` 를 **순회**한다 —
    목록을 테스트에 복사하면 새 변수가 추가돼도 테스트가 모른다(#1028 의 구조).
    """
    # Arrange
    declared = {v[0] for v in m.CAMS_VARIABLES.values()} | set(m.AUX_VARIABLES)
    # Act
    single = m.build_single_level_request("2026-07-16", "12:00")
    multi = m.build_multi_level_request("2026-07-16", "12:00")
    # Assert
    assert set(single["variable"]) | set(multi["variable"]) == declared
    assert set(single["variable"]) & set(multi["variable"]) == set()


def test_single_level_request_carries_no_level_key():
    # Arrange / Act
    req = m.build_single_level_request("2026-07-16", "12:00")
    # Assert — Single level 그룹에 레벨 키를 붙이면 ADS 가 거절한다
    assert "model_level" not in req and "pressure_level" not in req
    assert "ozone" not in req["variable"]
    assert req["date"] == ["2026-07-16"]
    assert req["time"] == ["12:00"]
    assert req["leadtime_hour"] == ["0"]
    assert req["area"] == [90, -180, -90, 180]


def test_multi_level_request_pins_gases_to_the_surface_model_level():
    """레벨 키가 빠지면 ADS 는 에러 없이 가스만 빼고 응답한다 — 그 회귀를 고정한다."""
    # Arrange
    expected_gases = {
        m.CAMS_VARIABLES[k][0] for k in m.MULTI_LEVEL_POLLUTANTS
    }
    # Act
    req = m.build_multi_level_request("2026-07-16", "12:00")
    # Assert
    assert set(req["variable"]) == expected_gases
    assert req["model_level"] == [m.SURFACE_MODEL_LEVEL]
    assert req["area"] == [90, -180, -90, 180]


def test_grid_header_for_native_cams_resolution():
    # Arrange / Act
    header = m.grid_header_for(0.4)
    # Assert
    assert header == {"nx": 900, "ny": 451, "la1": 90.0, "lo1": -180.0, "dx": 0.4, "dy": 0.4}


def test_convert_mass_rows_scales_kgm3_to_ugm3():
    # Arrange — CAMS PM2.5 예시값 근사
    rows = [(0.0, 0.0, 1.32e-8)]
    # Act
    out = m.convert_mass_rows(rows)
    # Assert
    assert out[0][2] == pytest.approx(13.2, rel=1e-2)


def test_convert_mass_rows_rejects_empty():
    with pytest.raises(ValueError):
        m.convert_mass_rows([])


def test_convert_mixing_rows_uses_paired_density():
    # Arrange — 표준 해면 조건 근사(101325 Pa, 288.15 K)
    rows = [(0.0, 0.0, 1e-8)]
    sp = {(0.0, 0.0): 101325.0}
    t2m = {(0.0, 0.0): 288.15}
    # Act
    out = m.convert_mixing_rows(rows, sp, t2m)
    # Assert — density ≈ 1.225 → conc ≈ 1e-8 * 1.225 * 1e9 ≈ 12.25
    assert out[0][2] == pytest.approx(12.25, rel=1e-2)


def test_convert_mixing_rows_raises_when_density_inputs_missing():
    # Arrange — sp/2t 에 해당 좌표가 없음(격자 정렬 불일치 가정)
    rows = [(5.0, 5.0, 1e-8)]
    # Act / Assert
    with pytest.raises(ValueError):
        m.convert_mixing_rows(rows, sp_by_coord={}, t2m_by_coord={})


def test_rows_to_coord_map_rounds_keys():
    # Arrange / Act
    cmap = m.rows_to_coord_map([(1.00001, 2.00001, 99.0)])
    # Assert
    assert cmap == {(1.0, 2.0): 99.0}


def test_build_pollutant_block_mass_kind_conversion_string():
    header = {"nx": 1, "ny": 1, "la1": 0.0, "lo1": 0.0, "dx": 1.0, "dy": 1.0}
    block = m.build_pollutant_block([(0.0, 0.0, 10.0)], header, "pm25")
    assert block["sourceVariable"] == "pm2p5"
    assert "kg/m3" in block["conversion"]
    assert block["data"] == [10.0]


def test_build_pollutant_block_mixing_kind_conversion_string():
    header = {"nx": 1, "ny": 1, "la1": 0.0, "lo1": 0.0, "dx": 1.0, "dy": 1.0}
    block = m.build_pollutant_block([(0.0, 0.0, 10.0)], header, "o3")
    assert block["sourceVariable"] == "go3"
    assert "air_density" in block["conversion"]


def test_assemble_reading_matches_adapter_schema_and_validates():
    # Arrange
    header = {"nx": 1, "ny": 1, "la1": 0.0, "lo1": 0.0, "dx": 0.4, "dy": 0.4}
    blocks = {
        key: {"unit": "ug/m3", "sourceVariable": short, "conversion": "x", "data": [1.0]}
        for key, (_ads, short, _kind) in m.CAMS_VARIABLES.items()
    }
    # Act
    reading = m.assemble_reading(blocks, header, "2026-07-16T15:00:00Z",
                                  "2026-07-16T12:00:00Z", age_hours=3.0)
    # Assert
    assert reading["source"] == "CAMS"
    assert reading["kind"] == "analysis"
    assert reading["quality"]["grade"] == "A"
    assert adapter.validate_grid_reading(reading) == []


def test_assemble_reading_raises_on_empty_pollutants():
    header = {"nx": 1, "ny": 1, "la1": 0.0, "lo1": 0.0, "dx": 0.4, "dy": 0.4}
    with pytest.raises(Exception):
        m.assemble_reading({}, header, "2026-07-16T15:00:00Z", "2026-07-16T12:00:00Z", 3.0)


# ────────────────────────── GRIB 인벤토리 / 레벨 가드 ──────────────────────────

# grib_ls 실출력 형태 — 파일명 줄, 컬럼 헤더, 데이터, "N of M messages" 꼬리가 섞인다.
_GRIB_LS_STDOUT = """cams.grib2
shortName    typeOfLevel    level
pm2p5        surface        0
pm10         surface        0
sp           surface        0
2t           heightAboveGround 2
4 of 4 messages in cams.grib2

4 of 4 total messages in 1 files
"""


def test_parse_grib_inventory_keeps_only_message_rows():
    # Arrange / Act
    inv = m.parse_grib_inventory(_GRIB_LS_STDOUT)

    # Assert — 헤더/파일명/꼬리 줄은 빠지고 메시지 4건만
    assert inv == [
        ("pm2p5", "surface", 0),
        ("pm10", "surface", 0),
        ("sp", "surface", 0),
        ("2t", "heightAboveGround", 2),
    ]


def test_resolve_single_level_returns_the_only_level():
    # Arrange / Act
    assert m.resolve_single_level(m.parse_grib_inventory(_GRIB_LS_STDOUT), "2t") == (
        "heightAboveGround", 2,
    )


def test_resolve_single_level_reports_what_arrived_when_variable_absent():
    # Arrange — go3 가 안 온 실제 실패 상황 (mass 종만 도착)
    inv = m.parse_grib_inventory(_GRIB_LS_STDOUT)

    # Act
    with pytest.raises(RuntimeError) as exc:
        m.resolve_single_level(inv, "go3")

    # Assert — 구 메시지("go3: decode 0 rows")는 무엇이 대신 왔는지를 안 알려줘
    # 원인 분류가 불가능했다. 빠진 것과 도착한 것을 **둘 다** 담아야 한다.
    msg = str(exc.value)
    assert "go3" in msg
    assert "pm2p5" in msg and "sp" in msg


def test_resolve_single_level_refuses_to_silently_merge_multiple_levels():
    # Arrange — 가스 종이 model level 로 오는 경우: 같은 shortName 이 여러 레벨
    inv = [("go3", "hybrid", lvl) for lvl in (135, 136, 137)]

    # Act
    with pytest.raises(RuntimeError) as exc:
        m.resolve_single_level(inv, "go3")

    # Assert — 병합은 last-write-wins 라 엉뚱한 고도 값이 지표면 값으로 발행된다.
    # "0 rows" 보다 나쁜 조용한 오염이라 반드시 세워야 한다.
    assert "3개" in str(exc.value)


def test_decode_passes_level_selectors_into_grib_get_data_argv(monkeypatch):
    # Arrange — 실제 argv 를 캡처한다. "레벨을 확정했다"고 계산만 하고 디코드
    # 호출엔 안 넘기는 반쪽 수정(#1028 형태)을 잡기 위함.
    captured = {}

    class _Result:
        stdout = "0.0 0.0 1.0\n"

    def _fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _Result()

    monkeypatch.setattr(m.subprocess, "run", _fake_run)

    # Act
    m.decode_grib_by_shortname("f.grib2", "go3", type_of_level="hybrid", level=137)

    # Assert
    where = captured["argv"][captured["argv"].index("-w") + 1]
    assert where == "shortName=go3,typeOfLevel=hybrid,level=137"


def test_decode_omits_level_selectors_when_not_given(monkeypatch):
    # Arrange
    captured = {}

    class _Result:
        stdout = "0.0 0.0 1.0\n"

    monkeypatch.setattr(
        m.subprocess, "run",
        lambda argv, **kw: (captured.__setitem__("argv", argv), _Result())[1],
    )

    # Act
    m.decode_grib_by_shortname("f.grib2", "pm2p5")

    # Assert — 기존 호출 형태와 하위호환
    assert captured["argv"][captured["argv"].index("-w") + 1] == "shortName=pm2p5"


def test_decode_at_resolved_level_forwards_the_resolved_level(monkeypatch):
    # Arrange — 레벨을 *계산* 만 하고 디코드 호출엔 안 넘기는 반쪽 수정(#1028 형태)을
    # 잡는다. 인벤토리가 hybrid/137 이라고 말했으면 그게 그대로 넘어가야 한다.
    captured = {}
    monkeypatch.setattr(
        m, "decode_grib_by_shortname",
        lambda path, sn, **kw: (captured.update(short_name=sn, **kw), [(0.0, 0.0, 1.0)])[1],
    )

    # Act
    m.decode_at_resolved_level("f.grib2", "go3", [("go3", "hybrid", 137)])

    # Assert
    assert captured == {"short_name": "go3", "type_of_level": "hybrid", "level": 137}


def test_decode_at_resolved_level_raises_before_decoding_when_ambiguous(monkeypatch):
    # Arrange — 다중 레벨이면 디코드를 **아예 시도하지 않아야** 한다.
    called = []
    monkeypatch.setattr(
        m, "decode_grib_by_shortname", lambda *a, **k: called.append(a) or [],
    )

    # Act
    with pytest.raises(RuntimeError):
        m.decode_at_resolved_level("f.grib2", "go3", [("go3", "hybrid", lv) for lv in (136, 137)])

    # Assert
    assert called == []


def test_main_routes_every_production_variable_through_the_resolved_decoder(tmp_path, monkeypatch):
    # Arrange — CAMS_VARIABLES/AUX_VARIABLES 를 **순회**한다. 목록을 테스트에
    # 복사하면 새 변수가 추가돼도 테스트가 모른다 — 그게 #1028 의 구조다.
    expected = {sn for _v, sn, _k in m.CAMS_VARIABLES.values()} | set(m.AUX_VARIABLES.values())

    monkeypatch.setattr(m, "OUTPUT_PATH", str(tmp_path / "cams-global-snapshot.json"))
    monkeypatch.setattr(m, "pick_latest_cams_cycle", lambda now: ("2026-07-16", "00:00"))
    monkeypatch.setattr(m, "fetch_cams_grib", lambda request, target_path: None)
    monkeypatch.setattr(m, "read_grib_inventory", lambda path: [])

    seen: list[str] = []

    def _decode_at(path, short_name, inventory):
        seen.append(short_name)
        return [(0.0, 0.0, 101325.0 if short_name == "sp" else 288.15)]

    monkeypatch.setattr(m, "decode_at_resolved_level", _decode_at)
    # 1셀 fixture 로는 첫 오염물질에서 405,900 셀 완전성 검사에 걸려 루프가 끊긴다 —
    # 그러면 뒤쪽 변수가 이 경로를 지나갔는지 볼 수가 없다. 조립만 비운다.
    monkeypatch.setattr(m, "build_pollutant_block", lambda rows, header, key: {"key": key})

    # Act — main 의 반환값은 보지 않는다. 1셀 fixture 라 그리드 완전성 검사에서
    # 어차피 실패하고, 이 테스트의 관심사는 "모든 변수가 이 경로를 지나갔나" 뿐이다.
    m.main()

    # Assert — 어떤 변수도 레벨 확정을 우회한 경로로 디코드되지 않는다
    assert expected.issubset(set(seen))


def test_main_decodes_each_variable_from_its_own_group_file(tmp_path, monkeypatch):
    """가스를 single-level 파일에서 찾으면 영원히 못 찾는다 — 파일 귀속을 고정한다."""
    # Arrange — 요청의 variable 목록으로 그 파일이 어느 그룹인지 라벨링한다
    monkeypatch.setattr(m, "OUTPUT_PATH", str(tmp_path / "cams-global-snapshot.json"))
    monkeypatch.setattr(m, "pick_latest_cams_cycle", lambda now: ("2026-07-16", "00:00"))
    monkeypatch.setattr(m, "build_pollutant_block", lambda rows, header, key: {"key": key})
    monkeypatch.setattr(m, "read_grib_inventory", lambda path: [])

    group_of_path: dict[str, str] = {}

    def _fetch(request, target_path):
        group_of_path[target_path] = "multi" if "model_level" in request else "single"

    monkeypatch.setattr(m, "fetch_cams_grib", _fetch)

    decoded_from: dict[str, str] = {}

    def _decode_at(path, short_name, inventory):
        decoded_from[short_name] = group_of_path[path]
        return [(0.0, 0.0, 101325.0 if short_name == "sp" else 288.15)]

    monkeypatch.setattr(m, "decode_at_resolved_level", _decode_at)

    # Act
    m.main()

    # Assert — 선언 테이블을 순회해 귀속을 검사한다(목록 복사 금지)
    for key, (_ads, short_name, _kind) in m.CAMS_VARIABLES.items():
        want = "multi" if key in m.MULTI_LEVEL_POLLUTANTS else "single"
        assert decoded_from[short_name] == want, f"{short_name} decoded from {want!r} file"
    for short_name in m.AUX_VARIABLES.values():
        assert decoded_from[short_name] == "single"


# ────────────────────────── outage/부분실패 경로 (fixture mock, 네트워크 0) ──────────────────────────

def test_main_preserves_last_good_snapshot_when_ads_unreachable(tmp_path, monkeypatch):
    # Arrange — 기존 snapshot 존재, ADS 요청 자체가 실패(네트워크/인증 오류 시뮬레이션)
    existing = tmp_path / "cams-global-snapshot.json"
    existing.write_text('{"schemaVersion":1,"note":"last-good"}')
    monkeypatch.setattr(m, "OUTPUT_PATH", str(existing))
    monkeypatch.setattr(m, "pick_latest_cams_cycle", lambda now: ("2026-07-16", "00:00"))

    def _raise_unreachable(request, target_path):
        raise RuntimeError("simulated ADS unreachable")

    monkeypatch.setattr(m, "fetch_cams_grib", _raise_unreachable)

    # Act
    rc = m.main()

    # Assert — fail-loud, 기존 파일 완전 불변
    assert rc == 1
    assert existing.read_text() == '{"schemaVersion":1,"note":"last-good"}'


def test_main_preserves_last_good_snapshot_when_one_pollutant_missing(tmp_path, monkeypatch):
    # Arrange — ADS 요청은 성공하지만 특정 변수(go3) decode 가 실패 → 부분 snapshot 발행 금지
    existing = tmp_path / "cams-global-snapshot.json"
    existing.write_text('{"schemaVersion":1,"note":"last-good"}')
    monkeypatch.setattr(m, "OUTPUT_PATH", str(existing))
    monkeypatch.setattr(m, "pick_latest_cams_cycle", lambda now: ("2026-07-16", "00:00"))
    monkeypatch.setattr(m, "fetch_cams_grib", lambda request, target_path: None)
    # 인벤토리도 스텁 — 안 하면 실 `grib_ls` 를 빈 temp 파일에 돌려 subprocess 오류로
    # 죽고, 테스트는 "go3 누락 때문에" 가 아니라 엉뚱한 이유로 통과한다(vacuous pass).
    monkeypatch.setattr(
        m, "read_grib_inventory",
        lambda path: [(sn, "surface", 0) for sn in ("sp", "2t", "pm2p5", "pm10", "go3", "no2", "so2", "co")],
    )

    def _decode(path, short_name, *, type_of_level=None, level=None):
        if short_name == "sp":
            return [(0.0, 0.0, 101325.0)]
        if short_name == "2t":
            return [(0.0, 0.0, 288.15)]
        if short_name == "go3":
            raise RuntimeError("go3: decode 0 rows — 발행 중단")
        return [(0.0, 0.0, 1.0)]

    monkeypatch.setattr(m, "decode_grib_by_shortname", _decode)

    # Act
    rc = m.main()

    # Assert
    assert rc == 1
    assert existing.read_text() == '{"schemaVersion":1,"note":"last-good"}'
