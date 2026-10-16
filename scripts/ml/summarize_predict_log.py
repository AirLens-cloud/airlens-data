#!/usr/bin/env python3
"""predict 로그에서 Open-Meteo 기상 조회 실패 수를 뽑아 run 요약과 경고로 올린다 (stdlib 전용).

왜 로그를 읽나. `predict` 는 실시간 기상(Open-Meteo) 조회가 실패한 지점을 **상수 기상으로 대체**하고,
그 규모를 로그 한 줄로만 남긴다 — 페이로드에는 없다(`met 실시간 조회 실패 N건 — 해당 지점 기상은 상수
fallback`). 이 레포는 hosted 러너라 공유 IP 라서 Open-Meteo 가 throttle 하면 조용히 품질이 내려간다.
컷오버 조건("met 실패 0 또는 운영과 같은 수준")을 판단하려면 run 마다 이 수가 보여야 한다.

문구가 바뀌면 0건으로 보이는 위험이 있다 — 그래서 매칭이 0줄이면 "0건" 으로 적되, 로그가 비어 있으면
(= predict 가 아무것도 못 찍은 것) 이상으로 알린다.

사용: summarize_predict_log.py predict.log   (항상 exit 0 — advisory. 판단은 사람이 한다)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

MET_FAILED_RE = re.compile(r"met 실시간 조회 실패 (\d+)건")


def met_failures(log_text: str) -> int:
    """로그 전체의 met 실패 합계. predict 는 실행당 한 번 찍지만 여러 번 찍혀도 합친다."""
    return sum(int(n) for n in MET_FAILED_RE.findall(log_text))


def render_summary(failed: int) -> str:
    verdict = "없음" if failed == 0 else f"{failed}건 — 해당 지점은 상수 기상으로 예측됐다"
    return "\n".join(
        [
            "### predict 실행 요약",
            "",
            "| 항목 | 값 |",
            "|---|---|",
            f"| Open-Meteo 실시간 기상 조회 실패 | {verdict} |",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    args = parser.parse_args(argv)

    try:
        text = args.log.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"::warning title=predict log unreadable::predict 로그를 읽을 수 없다({type(exc).__name__}) — met 실패 수를 알 수 없다")
        return 0
    if not text.strip():
        print("::warning title=predict log empty::predict 로그가 비어 있다 — met 실패 수를 알 수 없다")
        return 0

    failed = met_failures(text)
    summary = render_summary(failed)
    print(summary)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(summary)
    if failed:
        print(
            f"::warning title=Open-Meteo fallback::실시간 기상 조회 {failed}건 실패 — "
            "해당 지점은 상수 기상으로 예측됐다 (공유 IP throttle 가능성)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
