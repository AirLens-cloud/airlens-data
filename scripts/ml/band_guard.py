#!/usr/bin/env python3
"""band 추론 job 의 "이미 처리한 예보 발행이면 건너뛴다" 판정, 운영 latest 사본 받기, 다음 관측 슬롯 기다리기 (stdlib 전용, 토큰 없음).

band job 은 예보 수집(forecast-collect) 직후와 백업 스케줄에 모두 돈다. 같은 예보 발행에 대한 발행물이 이미 있는데
추론(수십 분)을 다시 돌릴 이유가 없다. 건너뛰기는 **최적화일 뿐**이라, 판정을 못 하면(네트워크 오류·모양이 틀린
응답) 경고만 남기고 그대로 돈다 — 발행물을 놓치는 쪽이 더 나쁘다.

판정 기준은 예보 발행 시각 하나다.

  issue       공개 `aq-data/forecast.json` 의 `generated_at` (band 엔진이 쓰는 issue 시각과 같은 값)
  production  공개 운영 latest 의 `issue_time` 이 issue 와 같은 시각이면 건너뛴다
  candidate   후보 발행물(`band_<issue 의 UTC %Y%m%d%H>.json`)이 이미 있으면 건너뛴다
  --force true 는 어느 쪽이든 건너뛰지 않는다.

"이미 있다" 는 파일이 응답한다(200)는 뜻이 아니라 본문이 그 issue 의 band 발행물이라는 뜻이다 — JSON 객체이고 `schema_version`
이 `w3-band-v1` 이고 `issue_time` 의 STAMP 가 같을 때만. 점검 페이지 같은 200 응답이나 다른 issue 의 파일 때문에 후보 발행이
빠지면 안 된다. 아니면 경고를 남기고 돈다. snapshot 이 "같은 issue 의 사본이 이미 있나" 를 보는 것도 같은 확인이다.

시각 문자열은 출처마다 모양이 다르다(`...Z` ↔ `...+00:00`). 문자열이 아니라 파싱한 시각으로 비교한다. 파서는
`publish_targets.parse_utc` 하나다 — snapshot 이 받은 `issue_time` 은 발행 매핑(`publish_targets.py maps`)도 받는다.
공개 저장소 조회(재시도·404·오류 문구)는 `compare_candidate.HttpSource` 를 그대로 쓴다 — 조회 코드를 두 벌 두지 않는다.

CLI (워크플로가 쓴다):
    band_guard.py check    --mode M --force true|false   → $GITHUB_OUTPUT 에 `skip=true|false`
    band_guard.py snapshot --mode M --dest FILE          → 후보 모드에서만: 지금의 운영 latest 원문을 FILE 로 저장
                                                            (같은 issue 의 사본이 이미 있거나, 받지 못하거나, 계약을 어기면 쓰지 않는다)
    band_guard.py wait-slot                              → 예보 issue 의 다음 관측 슬롯이 올라올 때까지 기다린다 (마감이 있고, 실패하지 않는다)

운영 band 는 latest 하나만 덮어써서 이력이 없다. 후보 모드가 후보 발행물과 같은 시점의 운영 latest 사본을 같이
남겨야(`snapshot` → `publish_targets.py maps --prod-snapshot`) 나중에 같은 issue 끼리 짝지어 비교할 수 있다.

band 엔진은 issue 를 시 단위로 내린 H 까지를 컨텍스트 마지막 시각으로 쓴다. H 시 관측은 공개 관측 슬롯 H 에는 일부만 있고 슬롯
H+1 이 올라와야 대부분 들어오는데, band 는 예보 수집 직후에 돌아 그 일부만으로 마지막 시각을 만든다. `wait-slot` 이 슬롯 H+1
(`openaq-shadow/openaq-<UTC %Y%m%d%H>.json.gz`)이 올라올 때까지 기다린다. 마감은 H+1 시 30분이고 시작 뒤 `MAX_WAIT` 가 상한이다.
기다림도 **최적화일 뿐이라** 못 기다리면(마감까지 슬롯이 없거나, 예보 발행 시각을 읽지 못하거나, 그 시각으로 슬롯을 정할 수
없으면) 경고만 남기고 0 으로 끝나 band 는 그대로 돈다. 슬롯이 이미 있으면 바로 끝난다. 여기서도 200 응답이 곧 "있다" 는 아니다 —
본문이 gzip 이어야 한다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import compare_candidate as cc
import publish_targets as pt

FORECAST_PATH = "aq-data/forecast.json"  # 예보 수집이 덮어쓰는 최신 예보
BAND_SCHEMA = "w3-band-v1"  # 사본으로 남길 운영 latest 의 계약 버전
SLOT_DIR = "openaq-shadow"  # 공개 dataset 의 OpenAQ 관측 슬롯 폴더 — 슬롯 파일은 `openaq-<UTC %Y%m%d%H>.json.gz`
SLOT_WAIT_UNTIL = timedelta(hours=1, minutes=30)  # issue 를 시 단위로 내린 시각부터의 마감 — 슬롯 H+1 은 보통 H+1 시 16~19분에 온다
MAX_WAIT = timedelta(minutes=90)  # 시작 시각부터의 상한 — issue 시각이 이상해도 이보다 오래 기다리지 않는다
POLL_INTERVAL_S = 120
GZIP_MAGIC = b"\x1f\x8b"


@dataclass(frozen=True)
class Decision:
    skip: bool
    reason: str


@dataclass(frozen=True)
class Facts:
    """조회로 알아낸 사실. 모르면 None / False — `decide` 는 모르는 쪽을 "돌린다"로 처리한다."""

    issue: datetime | None
    published_issue: datetime | None = None  # production 전용: 운영 latest 의 issue_time
    candidate_exists: bool = False  # candidate 전용: 같은 issue 의 후보 발행물이 실제로 있다(본문 확인)


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
    return Decision(False, f"후보 발행물 band_{pt.stamp_of(facts.issue)}.json 이 없거나 그 issue 의 발행물이 아니다 — 예보 발행 {issue} 을(를) 돌린다")


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


def _band_stamp(doc: dict) -> str:
    """band 발행물(`w3-band-v1`)이면 `issue_time` 의 STAMP. 아니면 ValueError."""
    if doc.get("schema_version") != BAND_SCHEMA:
        raise ValueError(f"schema_version 이 {BAND_SCHEMA} 가 아니다")
    try:
        return pt.utc_stamp(doc.get("issue_time"))
    except ValueError as exc:
        raise ValueError(f"issue_time 를 읽을 수 없다 ({exc})") from None


def _has_issue(client: cc.HttpSource, rel_path: str, stamp: str, warnings: list[str]) -> bool:
    """`rel_path` 에 issue `stamp` 의 band 발행물이 실제로 있나. 없거나 읽지 못했거나 본문이 그 발행물이 아니면 False.

    404 는 조용히 False 다. 그 밖의 이유(200 인데 JSON 이 아님, 다른 schema·issue, 조회 실패)는 경고를 모은다.
    """
    try:
        doc = client.read_json(rel_path)
        if doc is None:
            return False
        found = _band_stamp(doc)
    except cc.SourceError as exc:
        warnings.append(str(exc))
        return False
    except ValueError as exc:
        warnings.append(f"{rel_path}: band 발행물 모양이 아니다 ({exc})")
        return False
    if found != stamp:
        warnings.append(f"{rel_path}: issue {found} 의 발행물이다 — issue {stamp} 의 발행물로 보지 않는다")
        return False
    return True


def observe(mode: str, client: cc.HttpSource, warnings: list[str]) -> Facts:
    """판정에 필요한 사실을 공개 저장소에서 읽는다. 실패는 경고로 모으고 모르는 값으로 둔다."""
    issue = _read_instant(client, FORECAST_PATH, "generated_at", warnings)
    if issue is None:
        return Facts(None)
    if mode == "production":
        return Facts(issue, published_issue=_read_instant(client, pt.BAND_LATEST, "issue_time", warnings, missing_ok=True))
    stamp = pt.stamp_of(issue)
    return Facts(issue, candidate_exists=_has_issue(client, pt.band_history_dest(stamp), stamp, warnings))


def check(mode: str, force: bool, client: cc.HttpSource) -> tuple[Decision, list[str]]:
    """(판정, 경고들). `--force` 면 공개 저장소를 조회하지 않는다."""
    warnings: list[str] = []
    facts = Facts(None) if force else observe(mode, client, warnings)
    return decide(mode, force, facts), warnings


def snapshot(client: cc.HttpSource, dest: Path, warnings: list[str], validate: Callable[[dict], str | None]) -> str:
    """지금의 운영 latest 원문을 dest 로 저장한다(후보 모드). 결과 설명 한 줄을 돌려준다.

    같은 issue 의 사본이 이미 있으면(본문이 그 issue 의 band 발행물일 때만) 쓰지 않는다. 운영 latest 를 못 받거나 band 모양이
    아니거나 계약 검증(`validate`)에 걸리면 쓰지 않고 경고만 남긴다 — 사본은 비교용 보조 자료라 후보 발행을 막지 않는다.
    계약 검증은 발행기의 `batch --schema auto` 와 같은 검증기다: 거르지 않으면 계약을 어긴 사본 하나가 batch 전체
    (후보 발행물까지)를 멈춘다.
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
        stamp = _band_stamp(doc)
    except ValueError as exc:
        warnings.append(f"{pt.BAND_LATEST}: band 발행물 모양이 아니다 ({exc})")
        return "운영 latest 가 band 발행물 모양이 아니라 사본을 남기지 않는다"
    error = validate(doc)
    if error is not None:
        warnings.append(f"{pt.BAND_LATEST}: band 계약을 어겼다 ({error!r:.300})")  # repr: 줄바꿈이 섞여도 로그 한 줄, 길이도 제한
        return "운영 latest 가 band 계약을 어겨 사본을 남기지 않는다"
    if _has_issue(client, pt.band_prod_copy_dest(stamp), stamp, warnings):
        return f"issue {stamp} 의 운영 사본이 이미 있다 — 쓰지 않는다"
    # 확인을 못 했거나 본문이 그 발행물이 아니면 쓴다. 같은 issue 의 운영 latest 는 덮어써도 비교에 해가 없다
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return f"issue {stamp} 의 운영 latest 를 {dest.name} 로 저장했다"


def wait_plan(issue: datetime, start: datetime) -> tuple[str, datetime]:
    """(기다릴 관측 슬롯의 STAMP, 마감). 순수 함수.

    band 엔진은 issue 를 시 단위로 내린 H 까지를 컨텍스트 마지막 시각으로 쓴다. H 시 관측은 슬롯 H 에는 일부만 있고 슬롯 H+1 이
    와야 대부분 들어오므로 H+1 을 기다린다. 마감 = H 에서 `SLOT_WAIT_UNTIL` 뒤와 시작에서 `MAX_WAIT` 뒤 중 이른 쪽이다 — issue 가
    먼 미래여도 시작 뒤 `MAX_WAIT` 를 넘겨 기다리지 않는다. `issue` 는 tz 가 있는 시각이고, UTC 로 바꾼 뒤 시 단위로 내린다
    (반 시간 오프셋이면 순서가 결과를 바꾼다). H+1 시가 `datetime` 의 표현 범위를 넘는 issue(9999-12-31 23 시대)는 OverflowError 다.
    """
    hour = issue.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return pt.stamp_of(hour + timedelta(hours=1)), min(hour + SLOT_WAIT_UNTIL, start + MAX_WAIT)


def _slot_present(client: cc.HttpSource, rel_path: str, warnings: list[str]) -> bool:
    """관측 슬롯 `rel_path` 가 실제로 올라와 있나. 404 는 조용히 False 다.

    200 응답이 곧 "있다" 는 아니다(점검 페이지 등) — 본문이 gzip 이어야 한다. gzip 이 아니거나 조회에 실패하면 경고를 모으고
    False 다: 아직 없는 것으로 보고 마감까지 계속 기다린다.
    """
    try:
        # ponytail: 슬롯이 올라와 있으면 gz(약 1.1 MB)를 통째로 한 번 받는다(없을 때는 404 응답뿐). 있는지만 보면 되지만 HttpSource 에
        # HEAD 가 없다 — upgrade: HEAD 조회
        body = client.read_bytes(rel_path)
    except cc.SourceError as exc:
        warnings.append(str(exc))
        return False
    if body is None:
        return False
    if not body.startswith(GZIP_MAGIC):
        warnings.append(f"{rel_path}: gzip 이 아니다 — 관측 슬롯으로 보지 않는다")
        return False
    return True


def wait_for_slot(
    client: cc.HttpSource,
    warnings: list[str],
    *,
    now: Callable[[], datetime],
    sleep: Callable[[float], None],
    interval_s: float = POLL_INTERVAL_S,
    log: Callable[[str], None] | None = None,
) -> str:
    """예보 issue 의 다음 관측 슬롯(H+1)이 올라올 때까지 기다린다(`wait_plan` 의 마감까지). 결과 설명 한 줄을 돌려준다.

    기다림은 **최적화일 뿐이라** 못 기다려도 band 를 막지 않는다. 예보 발행 시각을 모르거나 그 시각으로 슬롯을 정할 수 없으면(표현
    범위를 넘는 값) 기다리지 않고, 마감까지 슬롯이 없으면 경고를 모으고 끝낸다. 시작할 때 이미 마감이 지났으면 한 번만 확인한다.
    읽기 실패(`cc.SourceError`)는 "아직 없음" 으로 보고 계속 기다린다 — 그 밖의 예외는 버그라 삼키지 않는다. `log` 로는 잠들기 전에
    한 번만 한 줄 알린다(결과·경고는 끝에 나오므로, 그 줄이 없으면 기다리는 동안 로그가 빈다). 슬롯이 이미 있거나 시작할 때 이미
    마감이 지났으면 자지 않으므로 아무것도 남기지 않는다.
    """
    start = now()  # 대기 상한(MAX_WAIT)이 예보 발행 시각을 읽는 시간까지 세게 한다 — 잡 timeout 의 여유가 그만큼 남는다
    issue = _read_instant(client, FORECAST_PATH, "generated_at", warnings)
    if issue is None:
        return "예보 발행 시각을 몰라 기다리지 않는다"
    try:
        stamp, deadline = wait_plan(issue, start)
    except OverflowError:  # 원격이 준 값이다: 9999-12-31 23 시대는 H+1 시가 표현 범위를 넘는다. 경고에는 파싱한 시각만 싣는다
        warnings.append(f"예보 발행 시각 {issue.isoformat()} 의 다음 관측 슬롯을 정할 수 없다 — 기다리지 않고 돈다")
        return "예보 발행 시각으로 기다릴 슬롯을 정할 수 없어 기다리지 않는다"
    rel_path = f"{SLOT_DIR}/openaq-{stamp}.json.gz"
    announced = False
    while True:
        present = _slot_present(client, rel_path, warnings)
        current = now()
        waited = int((current - start).total_seconds() // 60)
        if present:
            return f"관측 슬롯 {stamp} 이 있다 — {waited}분 기다렸다"
        remaining = (deadline - current).total_seconds()
        if remaining <= 0:
            warnings.append(f"관측 슬롯 {stamp} 이 마감 {deadline.isoformat()} 까지 올라오지 않았다 — 더 기다리지 않고 돈다")
            return f"관측 슬롯 {stamp} 이 마감까지 올라오지 않았다 — {waited}분 기다린 뒤 그대로 돈다"
        if log and not announced:
            log(f"관측 슬롯 {stamp} 이 아직 없다 — 마감 {deadline.isoformat()} 까지 {interval_s:.0f}초마다 확인한다")
            announced = True
        sleep(min(interval_s, remaining))


def _append(env: Mapping[str, str], key: str, text: str) -> None:
    """GitHub Actions 가 알려 준 파일(`$GITHUB_OUTPUT`·`$GITHUB_STEP_SUMMARY`)에 덧붙인다. 환경변수가 없으면 건너뛴다."""
    path = env.get(key)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)


def main(
    argv: list[str] | None = None,
    *,
    client: cc.HttpSource | None = None,
    env: Mapping[str, str] | None = None,
    validate: Callable[[dict], str | None] | None = None,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="이번 실행을 건너뛸지 판정한다")
    p_check.add_argument("--mode", required=True, choices=pt.MODES)
    p_check.add_argument("--force", required=True, choices=("true", "false"))

    p_snap = sub.add_parser("snapshot", help="후보 모드: 지금의 운영 latest 원문을 FILE 로 저장한다")
    p_snap.add_argument("--mode", required=True, choices=pt.MODES)
    p_snap.add_argument("--dest", required=True, type=Path)

    sub.add_parser("wait-slot", help="예보 issue 의 다음 관측 슬롯이 올라올 때까지 기다린다(마감이 있고, 못 기다려도 실패하지 않는다)")

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

    if args.cmd == "wait-slot":
        message = wait_for_slot(
            client, warnings, now=now or (lambda: datetime.now(timezone.utc)), sleep=sleep or time.sleep,
            log=lambda line: print(f"band_guard: {line}", flush=True),  # flush: 파이프에서는 끝날 때까지 버퍼에 남는다
        )
        for warning in dict.fromkeys(warnings):  # 같은 조회 실패가 폴링마다 쌓여도 한 줄
            print(f"::warning::band_guard: {warning}")
        print(f"band_guard: {message}")
        _append(env, "GITHUB_STEP_SUMMARY", f"### band 관측 슬롯 대기\n\n- 결과: {message}\n\n")
        return 0

    if args.mode != "candidate":
        print("band_guard: 운영 모드는 운영 latest 사본이 필요 없다")
        return 0
    message = snapshot(client, args.dest, warnings, validate or cc.contract_validator())
    for warning in warnings:
        print(f"::warning::band_guard: {warning}")
    print(f"band_guard: {message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
