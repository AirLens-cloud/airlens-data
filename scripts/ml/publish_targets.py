#!/usr/bin/env python3
"""추론 산출물 → HF 발행 경로 매핑 (모드, 스탬프) → (src, dest). stdlib 전용.

워크플로가 모드(`PUBLISH_MODE`)를 한 곳에서 바꾸면 발행 위치가 통째로 바뀌게 하는 단일 지점이다.

  candidate   `aq-data/_candidate/` 아래 **타임스탬프 파일만** 쓴다. 운영 경로(latest 포함)는 건드리지 않는다.
              이름이 `<접두>_<10자리>.json` 이라 기존 `hf_publish.py prune` 의 정규식에 걸린다 — 정리에
              새 코드가 필요 없다. 자동 prune 은 하지 않는다(컷오버 뒤 `candidate-prune` 워크플로가 `--keep-days 0`
              으로 한 번에 비운다).
  production  운영 경로: latest + history 스냅샷, 30일 prune. 기존 운영 워크플로의 의미를 그대로 옮긴 것이다.

STAMP 는 예측 내부 timestamp(정각)의 `%Y%m%d%H` 다. 같은 실행의 grid·DQSS·shadow 를 파일명 시각으로
짝지으려면 실행 시각이 아니라 이 값을 써야 한다(시각 경계를 넘는 실행에서 파일명 ≠ 내부 timestamp 가 되어
이력 짝짓기가 깨진다).

예측 밴드(band 그룹)의 STAMP 는 발행물의 `issue_time`(예보 발행 시각)의 UTC `%Y%m%d%H` 다. 운영은 `latest.json`
하나만 덮어써서 이력이 없으므로, 후보 모드는 후보 파일과 함께 **그 시점의 운영 latest 사본**(`--prod-snapshot`)을
같은 접두 아래 `prod/` 에 남긴다. 둘이 같은 STAMP 일 때 `compare_candidate.py --product band` 가 짝짓는다.

CLI (워크플로의 발행 job 이 쓴다 — 출력은 줄 단위라 `mapfile` 로 받는다):
    publish_targets.py maps  --mode M --dir DIR --group G [--prod-snapshot FILE]
                                                             → `SRC=DEST` 줄들 (G: grid | dqss | shadow | band)
    publish_targets.py prune --mode M                        → `PREFIX KEEP_DAYS` 줄들 (candidate 는 없음)
    publish_targets.py candidates --product P [--workflow-dir D]
                                                             → 컷오버 뒤 비울 후보 접두 줄들 (P: predictions | band).
                                                               그 제품의 워크플로가 production 이 아니면 종료 코드 1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MODES = ("candidate", "production")
GROUPS = ("grid", "dqss", "shadow", "band")
CANDIDATE_ROOT = "aq-data/_candidate"
HISTORY_KEEP_DAYS = 30
STAMP_RE = re.compile(r"\d{10}")
PRODUCTS = ("predictions", "band")  # compare_candidate.py --product 와 같은 이름
PRODUCT_WORKFLOWS = {"predictions": "ml-predict.yml", "band": "band-forecast.yml"}
WORKFLOW_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"
# 워크플로 최상위 env 의 `PUBLISH_MODE: <값>  # 주석` 한 줄. 따옴표·식 등 다른 모양은 읽지 않는다(줄 수 0 → 거부).
MODE_LINE_RE = re.compile(r"^[ \t]*PUBLISH_MODE:[ \t]*([A-Za-z]+)[ \t]*(?:#.*)?$", re.MULTILINE)

# 산출물 파일명 (artifact 디렉터리 기준)
GRID_FILE = "grid_latest.json"
DQSS_FILE = "data_quality.json"
SHADOW_FILE = "shadow_stations.json"
BAND_FILE = "forecast_band_latest.json"  # 예측 밴드 산출물(artifact 디렉터리 기준)
BAND_PROD_SNAPSHOT = "prod_latest.json"  # 같은 시점 운영 latest 사본 — `--prod-snapshot` 이 가리키는 파일의 역할 이름
BAND_LATEST = "aq-data/forecast-band/latest.json"  # 운영 경로 — 운영 band 는 이 파일 하나만 덮어쓴다


@dataclass(frozen=True)
class Target:
    group: str  # grid | dqss | shadow
    src: str  # artifact 디렉터리 기준 파일명
    dest: str  # HF 저장소 안의 경로
    required: bool  # False = 실패해도 경고만 (shadow 는 점수의 원자료라 advisory)


def parse_stamp(value: str) -> str:
    """10자리 `YYYYMMDDHH` 인지 확인한다. 실제 시각이 아니면(월 13 등) 거부."""
    if not STAMP_RE.fullmatch(value):
        raise ValueError(f"스탬프는 10자리 숫자여야 한다: {value!r}")
    datetime.strptime(value, "%Y%m%d%H")  # 범위 밖이면 ValueError
    return value


def parse_utc(value: object) -> datetime:
    """ISO 8601 시각 문자열 → UTC 시각(tz 있음). `Z`·오프셋·오프셋 없음(UTC 로 본다)을 받는다. 시각이 아니면 ValueError.

    STAMP 를 만드는 `utc_stamp` 와 `band_guard.py` 의 "같은 시각인가" 비교가 이 한 곳을 쓴다 — 한쪽만 받아 주는 값이
    생기지 않는다. 오류 메시지는 값을 잘라 담는다(표준 메시지는 값 전체를 싣는다).
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"시각 문자열이 아니다: {value!r:.80}")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError):  # 범위 밖 시각(`0001-01-01T00:00:00+14:00`)은 UTC 로 바꾸다 OverflowError
        raise ValueError(f"ISO 시각이 아니다: {value!r:.80}") from None


def stamp_of(moment: datetime) -> str:
    """시각(tz 있음) → UTC `%Y%m%d%H`. 후보 발행물 파일명 STAMP 의 규칙이다."""
    return moment.astimezone(timezone.utc).strftime("%Y%m%d%H")


def utc_stamp(ts: object) -> str:
    """ISO 시각 문자열 → UTC `%Y%m%d%H`. 오프셋이 붙은 값은 UTC 로 바꾼 뒤 자른다(오프셋 없는 값은 UTC 로 본다)."""
    return stamp_of(parse_utc(ts))


def stamp_from_grid(grid: dict) -> str:
    """`predictions[0].timestamp`(없으면 `generated_at`)를 `%Y%m%d%H`(UTC)로.

    운영 규칙과 같다. 다른 점 하나: 오프셋이 붙은 timestamp 는 UTC 로 바꾼 뒤 자른다(운영은 오프셋 그대로의 시각을
    쓴다). 현재 추론 산출물은 `Z` 라 결과가 같다.
    """
    predictions = grid.get("predictions") or []
    ts = predictions[0]["timestamp"] if predictions else grid.get("generated_at")
    if not ts:
        raise ValueError("grid 에 timestamp/generated_at 이 없다")
    return utc_stamp(ts)


def stamp_from_band(payload: dict) -> str:
    """예측 밴드 발행물의 `issue_time`(예보 발행 시각)을 `%Y%m%d%H`(UTC)로. `generated_at`(실행 시각)은 쓰지 않는다."""
    ts = payload["issue_time"]
    if not isinstance(ts, str) or not ts:
        raise ValueError("band 에 issue_time 이 없다")
    return utc_stamp(ts)


def predictions_targets(mode: str, stamp: str) -> list[Target]:
    """predictions 제품(grid·DQSS·shadow)의 발행 대상. 순서 = 발행 순서."""
    if mode not in MODES:
        raise ValueError(f"mode 는 {MODES} 중 하나여야 한다: {mode!r}")
    stamp = parse_stamp(stamp)
    if mode == "candidate":
        root = CANDIDATE_ROOT
        return [
            Target("grid", GRID_FILE, f"{root}/predictions/history/grid_{stamp}.json", True),
            Target("dqss", DQSS_FILE, f"{root}/quality-history/quality_{stamp}.json", True),
            Target("shadow", SHADOW_FILE, f"{root}/quality-history/shadow_{stamp}.json", False),
        ]
    return [
        Target("grid", GRID_FILE, "aq-data/predictions/grid_latest.json", True),
        Target("grid", GRID_FILE, f"aq-data/predictions/history/grid_{stamp}.json", True),
        Target("dqss", DQSS_FILE, "aq-data/data_quality.json", True),
        Target("dqss", DQSS_FILE, f"aq-data/quality-history/quality_{stamp}.json", True),
        Target("shadow", SHADOW_FILE, f"aq-data/quality-history/shadow_{stamp}.json", False),
    ]


def band_history_dest(stamp: str) -> str:
    """후보 모드의 band 발행물 경로(타임스탬프 파일). STAMP = 후보 발행물의 issue_time."""
    return f"{CANDIDATE_ROOT}/forecast-band/history/band_{parse_stamp(stamp)}.json"


def band_prod_copy_dest(stamp: str) -> str:
    """후보 모드에서 그 시점 운영 latest 사본을 두는 경로. STAMP = 운영 latest 의 issue_time."""
    return f"{CANDIDATE_ROOT}/forecast-band/prod/band_{parse_stamp(stamp)}.json"


def band_targets(mode: str, stamp: str, prod_stamp: str | None = None) -> list[Target]:
    """예측 밴드 발행 대상. `prod_stamp` 는 후보 모드에서 운영 latest 사본을 같이 남길 때만 준다(운영 모드는 무시)."""
    if mode not in MODES:
        raise ValueError(f"mode 는 {MODES} 중 하나여야 한다: {mode!r}")
    stamp = parse_stamp(stamp)
    if mode == "candidate":
        targets = [Target("band", BAND_FILE, band_history_dest(stamp), True)]
        if prod_stamp is not None:
            targets.append(Target("band", BAND_PROD_SNAPSHOT, band_prod_copy_dest(prod_stamp), False))
        return targets
    return [Target("band", BAND_FILE, BAND_LATEST, True)]


def prune_prefixes(mode: str) -> list[tuple[str, int]]:
    """(prefix, keep_days). candidate 는 자동 정리가 없다."""
    if mode not in MODES:
        raise ValueError(f"mode 는 {MODES} 중 하나여야 한다: {mode!r}")
    if mode == "candidate":
        return []
    return [
        ("aq-data/predictions/history", HISTORY_KEEP_DAYS),
        ("aq-data/quality-history", HISTORY_KEEP_DAYS),
    ]


def candidate_prefixes(product: str) -> list[str]:
    """컷오버 뒤 한 번에 비울 후보 접두. 후보 발행 경로의 폴더를 뽑으므로 운영 경로가 섞일 수 없다."""
    if product not in PRODUCTS:
        raise ValueError(f"product 는 {PRODUCTS} 중 하나여야 한다: {product!r}")
    stamp = "2000010100"  # 경로 모양만 쓴다
    if product == "predictions":
        targets = predictions_targets("candidate", stamp)
    else:
        targets = band_targets("candidate", stamp, stamp)
    return sorted({t.dest.rsplit("/", 1)[0] for t in targets})


def workflow_mode(path: Path) -> str:
    """워크플로의 `PUBLISH_MODE` 값. 그 줄이 정확히 하나가 아니면 ValueError — 어느 모드로도 추측하지 않는다."""
    modes = MODE_LINE_RE.findall(path.read_text(encoding="utf-8"))
    if len(modes) != 1:
        raise ValueError(f"{path.name}: PUBLISH_MODE 줄이 {len(modes)}개다")
    return modes[0]


def _load_stamp(directory: Path) -> str:
    grid = json.loads((directory / GRID_FILE).read_text(encoding="utf-8"))
    return stamp_from_grid(grid)


def _band_stamp(path: Path) -> str:
    return stamp_from_band(json.loads(path.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_maps = sub.add_parser("maps", help="`SRC=DEST` 줄을 낸다")
    p_maps.add_argument("--mode", required=True, choices=MODES)
    p_maps.add_argument("--dir", required=True, type=Path)
    p_maps.add_argument("--group", required=True, choices=GROUPS)
    p_maps.add_argument("--prod-snapshot", type=Path, default=None,
                        help="band 후보 모드에서 같이 남길 운영 latest 사본(없는 파일이면 건너뛴다)")

    p_prune = sub.add_parser("prune", help="`PREFIX KEEP_DAYS` 줄을 낸다")
    p_prune.add_argument("--mode", required=True, choices=MODES)

    p_cand = sub.add_parser("candidates", help="컷오버 뒤 비울 후보 접두 줄을 낸다(그 제품 워크플로가 production 일 때만)")
    p_cand.add_argument("--product", required=True, choices=PRODUCTS)
    p_cand.add_argument("--workflow-dir", type=Path, default=WORKFLOW_DIR)

    args = parser.parse_args(argv)

    try:
        if args.cmd == "maps":
            prod: Path | None = None  # band 후보 모드에서만 쓰는 운영 latest 사본
            if args.group == "band":
                if args.prod_snapshot is not None and args.prod_snapshot.is_file():
                    prod = args.prod_snapshot
                targets = band_targets(args.mode, _band_stamp(args.dir / BAND_FILE), _band_stamp(prod) if prod else None)
            else:
                targets = predictions_targets(args.mode, _load_stamp(args.dir))
            for target in targets:
                if target.group != args.group:
                    continue
                src = prod if prod and target.src == BAND_PROD_SNAPSHOT else args.dir / target.src
                if not src.is_file():
                    # shadow 는 advisory 라 파일이 없을 수 있다 — 호출부가 빈 출력으로 판단한다.
                    if target.required:
                        print(f"산출물이 없다: {target.src}", file=sys.stderr)
                        return 1
                    continue
                print(f"{src}={target.dest}")
        elif args.cmd == "prune":
            for prefix, keep_days in prune_prefixes(args.mode):
                print(f"{prefix} {keep_days}")
        else:
            # 후보 모드에서 지우면 컷오버 판단의 비교 근거가 사라진다 — dry-run 도 같은 조건이다
            workflow = args.workflow_dir / PRODUCT_WORKFLOWS[args.product]
            mode = workflow_mode(workflow)
            if mode != "production":
                print(f"{workflow.name} 가 아직 {mode} 모드다 — 컷오버 뒤에만 후보를 비운다", file=sys.stderr)
                return 1
            for prefix in candidate_prefixes(args.product):
                print(prefix)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:  # 깨진 grid(모양 다름 포함)는 traceback 대신 한 줄
        print(f"publish_targets: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
