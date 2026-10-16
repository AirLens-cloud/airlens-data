#!/usr/bin/env python3
"""band 추론 job 의 "이미 처리한 예보 발행이면 건너뛴다" 판정과 운영 latest 사본 받기 (stdlib 전용, 토큰 없음).

band job 은 예보 수집(forecast-collect) 직후와 백업 스케줄에 모두 돈다. 같은 예보 발행에 대한 발행물이 이미 있는데
추론(수십 분)을 다시 돌릴 이유가 없다. 건너뛰기는 **최적화일 뿐**이라, 판정을 못 하면(네트워크 오류·모양이 틀린
응답) 경고만 남기고 그대로 돈다 — 발행물을 놓치는 쪽이 더 나쁘다.

판정 기준은 예보 발행 시각 하나다.

  issue       공개 `aq-data/forecast.json` 의 `generated_at` (band 엔진이 쓰는 issue 시각과 같은 값)
  production  공개 운영 latest 의 `issue_time` 이 issue 와 같은 시각이면 건너뛴다
  candidate   후보 발행물(`band_<issue 의 UTC %Y%m%d%H>.json`)이 이미 있으면 건너뛴다
  --force true 는 어느 쪽이든 건너뛰지 않는다.

시각 문자열은 출처마다 모양이 다르다(`...Z` ↔ `...+00:00`). 문자열이 아니라 파싱한 시각으로 비교한다. 파서는
`publish_targets.parse_utc` 하나다 — snapshot 이 받은 `issue_time` 은 발행 매핑(`publish_targets.py maps`)도 받는다.
공개 저장소 조회(재시도·404·오류 문구)는 `compare_candidate.HttpSource` 를 그대로 쓴다 — 조회 코드를 두 벌 두지 않는다.

CLI (워크플로가 쓴다):
    band_guard.py check    --mode M --force true|false   → $GITHUB_OUTPUT 에 `skip=true|false`
    band_guard.py snapshot --mode M --dest FILE          → 후보 모드에서만: 지금의 운영 latest 원문을 FILE 로 저장
                                                            (같은 issue 의 사본이 이미 있거나 받지 못하면 쓰지 않는다)

운영 band 는 latest 하나만 덮어써서 이력이 없다. 후보 모드가 후보 발행물과 같은 시점의 운영 latest 사본을 같이
남겨야(`snapshot` → `publish_targets.py maps --prod-snapshot`) 나중에 같은 issue 끼리 짝지어 비교할 수 있다.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import compare_candidate as cc
import publish_targets as pt

FORECAST_PATH = "aq-data/forecast.json"  # 예보 수집이 덮어쓰는 최신 예보
BAND_SCHEMA = "w3-band-v1"  # 사본으로 남길 운영 latest 의 계약 버전


@dataclass(frozen=True)
class Decision:
    skip: bool
    reason: str


@dataclass(frozen=True)
class Facts:
    """조회로 알아낸 사실. 모르면 None / False — `decide` 는 모르는 쪽을 "돌린다"로 처리한다."""

    issue: datetime | None
    published_issue: datetime | None = None  # production 전용: 운영 latest 의 issue_time
    candidate_exists: bool = False  # candidate 전용


def decide(mode: str, force: bool, facts: Facts) -> Decision:
    """건너뛸지 정한다. 순수 함수 — 조회는 하지 않는다."""
    if mode not in pt.MODES:
        raise ValueError(f"mode 는 {pt.MODES} 중 하나여야 한다: {mode!r}")
    if force:
        return Decision(False, "force=true — 이미 처리한 발행이어도 돌린다")
    if facts.issue is None:
        return Decision(False, "예보 발행 시각을 알 수 없어 판정 없이 돌린다")
    issue = facts.issue.isoformat()
    if mode == "production":
        if facts.published_issue is None:
            return Decision(False, f"운영 latest 를 읽지 못했거나 아직 없다 — 예보 발행 {issue} 을(를) 돌린다")
        if facts.published_issue == facts.issue:
            return Decision(True, f"운영 latest 가 이미 예보 발행 {issue} 을(를) 담고 있다")
        return Decision(False, f"새 예보 발행 {issue} (운영 latest 는 {facts.published_issue.isoformat()})")
    if facts.candidate_exists:
        return Decision(True, f"후보 발행물 band_{pt.stamp_of(facts.issue)}.json 이 이미 있다")
    return Decision(False, f"후보 발행물 band_{pt.stamp_of(facts.issue)}.json 이 없다 — 예보 발행 {issue} 을(를) 돌린다")


def _read_instant(client: cc.HttpSource, rel_path: str, key: str, warnings: list[str], *, missing_ok: bool = False) -> datetime | None:
    """`rel_path` 의 JSON 에서 `key` 시각을 읽는다. 못 읽으면 경고를 모으고 None (missing_ok 면 파일 없음은 조용히)."""
    try:
        doc = client.read_json(rel_path)
        if doc is None:
            if not missing_ok:
                warnings.append(f"{rel_path}: 파일이 없다")
            return None
        return pt.parse_utc(doc.get(key))
    except cc.SourceError as exc:
        warnings.append(str(exc))
    except ValueError as exc:
        warnings.append(f"{rel_path}: {key} 를 읽을 수 없다 ({exc})")
    return None


def observe(mode: str, client: cc.HttpSource, warnings: list[str]) -> Facts:
    """판정에 필요한 사실을 공개 저장소에서 읽는다. 실패는 경고로 모으고 모르는 값으로 둔다."""
    issue = _read_instant(client, FORECAST_PATH, "generated_at", warnings)
    if issue is None:
        return Facts(None)
    if mode == "production":
        return Facts(issue, published_issue=_read_instant(client, pt.BAND_LATEST, "issue_time", warnings, missing_ok=True))
    candidate_path = pt.band_history_dest(pt.stamp_of(issue))
    try:
        return Facts(issue, candidate_exists=client.exists(candidate_path))
    except cc.SourceError as exc:
        warnings.append(str(exc))
        return Facts(issue)


def check(mode: str, force: bool, client: cc.HttpSource) -> tuple[Decision, list[str]]:
    """(판정, 경고들). `--force` 면 공개 저장소를 조회하지 않는다."""
    warnings: list[str] = []
    facts = Facts(None) if force else observe(mode, client, warnings)
    return decide(mode, force, facts), warnings


def snapshot(client: cc.HttpSource, dest: Path, warnings: list[str]) -> str:
    """지금의 운영 latest 원문을 dest 로 저장한다(후보 모드). 결과 설명 한 줄을 돌려준다.

    같은 issue 의 사본이 이미 있으면 쓰지 않는다. 운영 latest 를 못 받거나 band 계약 모양이 아니면 쓰지 않고 경고만 남긴다 —
    사본은 비교용 보조 자료라 후보 발행을 막지 않는다.
    """
    try:
        body = client.read_bytes(pt.BAND_LATEST)
        if body is None:
            return "운영 latest 가 아직 없다 — 사본을 남기지 않는다"
        doc = cc.parse_json_object(body, pt.BAND_LATEST)
    except cc.SourceError as exc:
        warnings.append(str(exc))
        return "운영 latest 를 받지 못해 사본을 남기지 않는다"
    try:
        if doc.get("schema_version") != BAND_SCHEMA:
            raise ValueError(f"schema_version 이 {BAND_SCHEMA} 가 아니다")
        stamp = pt.utc_stamp(doc.get("issue_time"))
    except ValueError as exc:
        warnings.append(f"{pt.BAND_LATEST}: band 발행물 모양이 아니다 ({exc})")
        return "운영 latest 가 band 발행물 모양이 아니라 사본을 남기지 않는다"
    try:
        if client.exists(pt.band_prod_copy_dest(stamp)):
            return f"issue {stamp} 의 운영 사본이 이미 있다 — 쓰지 않는다"
    except cc.SourceError as exc:  # 확인을 못 했으면 쓴다. 같은 issue 의 운영 latest 는 덮어써도 비교에 해가 없다
        warnings.append(str(exc))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return f"issue {stamp} 의 운영 latest 를 {dest.name} 로 저장했다"


def _append(env: Mapping[str, str], key: str, text: str) -> None:
    """GitHub Actions 가 알려 준 파일(`$GITHUB_OUTPUT`·`$GITHUB_STEP_SUMMARY`)에 덧붙인다. 환경변수가 없으면 건너뛴다."""
    path = env.get(key)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)


def main(argv: list[str] | None = None, *, client: cc.HttpSource | None = None, env: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="이번 실행을 건너뛸지 판정한다")
    p_check.add_argument("--mode", required=True, choices=pt.MODES)
    p_check.add_argument("--force", required=True, choices=("true", "false"))

    p_snap = sub.add_parser("snapshot", help="후보 모드: 지금의 운영 latest 원문을 FILE 로 저장한다")
    p_snap.add_argument("--mode", required=True, choices=pt.MODES)
    p_snap.add_argument("--dest", required=True, type=Path)

    args = parser.parse_args(argv)
    client = client or cc.HttpSource()
    env = os.environ if env is None else env
    warnings: list[str] = []

    if args.cmd == "check":
        decision, warnings = check(args.mode, args.force == "true", client)
        verdict = "건너뜀" if decision.skip else "실행"
        for warning in warnings:
            print(f"::warning::band_guard: {warning}")
        print(f"band_guard: {verdict} — {decision.reason}")
        _append(env, "GITHUB_OUTPUT", f"skip={'true' if decision.skip else 'false'}\n")
        _append(env, "GITHUB_STEP_SUMMARY", f"### band 건너뛰기 판정\n\n- 모드: `{args.mode}`\n- 판정: **{verdict}** — {decision.reason}\n\n")
        return 0

    if args.mode != "candidate":
        print("band_guard: 운영 모드는 운영 latest 사본이 필요 없다")
        return 0
    message = snapshot(client, args.dest, warnings)
    for warning in warnings:
        print(f"::warning::band_guard: {warning}")
    print(f"band_guard: {message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
