#!/usr/bin/env python3
"""추론 산출물 → HF 발행 경로 매핑 (모드, 스탬프) → (src, dest). stdlib 전용.

워크플로가 모드(`PUBLISH_MODE`)를 한 곳에서 바꾸면 발행 위치가 통째로 바뀌게 하는 단일 지점이다.

  candidate   `aq-data/_candidate/` 아래 **타임스탬프 파일만** 쓴다. 운영 경로(latest 포함)는 건드리지 않는다.
              이름이 `<접두>_<10자리>.json` 이라 기존 `hf_publish.py prune` 의 정규식에 걸린다 — 정리에
              새 코드가 필요 없다. 자동 prune 은 하지 않는다(컷오버 뒤 `--keep-days 0` 으로 한 번에 비운다).
  production  운영 경로: latest + history 스냅샷, 30일 prune. 기존 운영 워크플로의 의미를 그대로 옮긴 것이다.

STAMP 는 예측 내부 timestamp(정각)의 `%Y%m%d%H` 다. 같은 실행의 grid·DQSS·shadow 를 파일명 시각으로
짝지으려면 실행 시각이 아니라 이 값을 써야 한다(시각 경계를 넘는 실행에서 파일명 ≠ 내부 timestamp 가 되어
이력 짝짓기가 깨진다).

CLI (워크플로의 발행 job 이 쓴다 — 출력은 줄 단위라 `mapfile` 로 받는다):
    publish_targets.py maps  --mode M --dir DIR --group G    → `SRC=DEST` 줄들 (G: grid | dqss | shadow)
    publish_targets.py prune --mode M                        → `PREFIX KEEP_DAYS` 줄들 (candidate 는 없음)
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
GROUPS = ("grid", "dqss", "shadow")
CANDIDATE_ROOT = "aq-data/_candidate"
HISTORY_KEEP_DAYS = 30
STAMP_RE = re.compile(r"\d{10}")

# 산출물 파일명 (artifact 디렉터리 기준)
GRID_FILE = "grid_latest.json"
DQSS_FILE = "data_quality.json"
SHADOW_FILE = "shadow_stations.json"


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


def stamp_from_grid(grid: dict) -> str:
    """`predictions[0].timestamp`(없으면 `generated_at`)를 `%Y%m%d%H`(UTC)로.

    운영 규칙과 같다. 다른 점 하나: 오프셋이 붙은 timestamp 는 UTC 로 바꾼 뒤 자른다(운영은 오프셋 그대로의 시각을
    쓴다). 현재 추론 산출물은 `Z` 라 결과가 같다.
    """
    predictions = grid.get("predictions") or []
    ts = predictions[0]["timestamp"] if predictions else grid.get("generated_at")
    if not ts:
        raise ValueError("grid 에 timestamp/generated_at 이 없다")
    parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime("%Y%m%d%H")


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


def _load_stamp(directory: Path) -> str:
    grid = json.loads((directory / GRID_FILE).read_text(encoding="utf-8"))
    return stamp_from_grid(grid)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_maps = sub.add_parser("maps", help="`SRC=DEST` 줄을 낸다")
    p_maps.add_argument("--mode", required=True, choices=MODES)
    p_maps.add_argument("--dir", required=True, type=Path)
    p_maps.add_argument("--group", required=True, choices=GROUPS)

    p_prune = sub.add_parser("prune", help="`PREFIX KEEP_DAYS` 줄을 낸다")
    p_prune.add_argument("--mode", required=True, choices=MODES)

    args = parser.parse_args(argv)

    try:
        if args.cmd == "maps":
            stamp = _load_stamp(args.dir)
            for target in predictions_targets(args.mode, stamp):
                if target.group != args.group:
                    continue
                src = args.dir / target.src
                if not src.is_file():
                    # shadow 는 advisory 라 파일이 없을 수 있다 — 호출부가 빈 출력으로 판단한다.
                    if target.required:
                        print(f"산출물이 없다: {target.src}", file=sys.stderr)
                        return 1
                    continue
                print(f"{src}={target.dest}")
        else:
            for prefix, keep_days in prune_prefixes(args.mode):
                print(f"{prefix} {keep_days}")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:  # 깨진 grid(모양 다름 포함)는 traceback 대신 한 줄
        print(f"publish_targets: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
