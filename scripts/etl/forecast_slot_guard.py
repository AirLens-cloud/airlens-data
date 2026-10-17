#!/usr/bin/env python3
"""forecast-collect 의 "같은 6시간 칸에 이미 수집했으면 건너뛴다" 판정 (stdlib 전용, 토큰 없음).

예보 수집(`forecast-collect.yml`)은 두 길로 시작한다. E2 VM 의 타이머가 6시간마다 `workflow_dispatch` 로 정시에 시작시키고
(`e2-sidecar/`), 같은 주기의 `schedule` 이 백업으로 남아 있다. GitHub 스케줄은 몇 시간씩 늦게 와서 백업 run 은 정시 수집이 끝난
뒤에 도착한다. 같은 칸을 두 번 수집할 이유가 없으니 첫 잡이 이 판정으로 건너뛴다.

판정 기준은 예보 발행 시각 하나다.

  issue   공개 `aq-data/forecast.json` 의 `generated_at` (수집이 끝난 시각)
  slot    UTC 로 바꾼 시각의 시를 6 의 배수로 내린 값 — 칸의 경계는 UTC 00/06/12/18 시
  skip    force 가 아니고, issue 가 지금보다 뒤가 아니고, slot(issue) == slot(지금)이고, 그 예보의 archive 가 있다(아래)

"마지막 수집 뒤 N시간" 이 아니라 칸으로 정한 이유: N시간 기준은 그 칸에 수집이 없어도 늦게 도착한 백업 run 직후의 다음 정시 dispatch 를
막는다. 칸 기준은 같은 칸에 이미 수집이 있을 때만 건너뛴다 — 정시 dispatch 가 건너뛰는 때는 백업이 그 칸의 :50 전에 먼저 수집했고
그 칸의 archive 까지 올라갔을 때뿐이다.

칸이 같아도 그 칸의 **forecast archive 가 있어야** 건너뛴다. 워크플로는 forecast.json 을 먼저 올리고 archive
(`aq-data/forecast-archive/forecast_<STAMP>.json`)를 그다음에 올린다. archive 업로드가 실패한 run 은 실패로 끝나는데 forecast.json 의 발행 시각은
이미 그 칸 안이라, 칸만 보면 늦게 오는 백업이 건너뛰어 그 칸의 archive 가 영영 빠진다. 그래서 칸 규칙이 건너뛰라고 할 때만 archive 를 HEAD 로 묻는다.
STAMP 후보는 발행 시각의 UTC `%Y%m%dT%H` 와 그다음 시다(워크플로가 업로드 뒤에 `date -u` 로 정해서 그 사이 시가 넘어갈 수 있다). 하나라도 있으면
(리다이렉트를 따라가 200) 건너뛰고, 둘 다 없으면(404) 수집해서 그 칸의 archive 를 메운다. 그 밖의 상태·네트워크 오류·시간 초과는 있는지 모르는
것이라 경고하고 수집한다. `force` 이거나 칸이 다르면 HEAD 를 보내지 않는다.
AIFS archive 는 보지 않는다 — AIFS 는 fail-soft 단계라 산출물이 없어도 정상일 수 있어서, AIFS 업로드만 실패한 칸은 여전히 건너뛴다.

건너뛰기는 **최적화일 뿐이라** 판정을 못 하면 경고만 남기고 수집한다(skip=false) — 수집을 놓치는 쪽이 더 나쁘다. 못 하는 경우는
조회 실패·200 이 아닌 응답·크기 초과·JSON 이 아님·`generated_at` 이 없거나 시간대가 없거나 읽을 수 없음이다. issue 가 지금보다 뒤인
값도 믿지 않고 수집한다. 경고와 로그에는 원격이 준 글자를 싣지 않는다 — 고정 문구와 파싱한 시각의 isoformat 뿐이라, 원격 값이 로그에
자기 줄이나 `::명령::` 을 만들 수 없다.

CLI (워크플로가 쓴다):
    forecast_slot_guard.py --force true|false   → $GITHUB_OUTPUT 에 `skip=true|false`, $GITHUB_STEP_SUMMARY 에 요약
`--force true` 는 공개 저장소를 조회하지 않고 건너뛰지도 않는다. 종료 코드는 늘 0 이다 — 예상 밖 예외(버그)는 삼키지 않는다.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

LIVE_BASE = "https://huggingface.co/datasets/Robeedau/airlens-live/resolve/main"  # 공개 데이터셋 — 워크플로의 HF_LIVE_BASE 와 같다
FORECAST_URL = f"{LIVE_BASE}/aq-data/forecast.json"  # 예보 수집이 덮어쓰는 공개 최신 예보
ARCHIVE_URL = f"{LIVE_BASE}/aq-data/forecast-archive/forecast_{{stamp}}.json"  # 발행 시각별 사본 — 워크플로의 archive 단계가 올린다
STAMP_FORMAT = "%Y%m%dT%H"  # archive 파일명의 STAMP — 워크플로의 `date -u +%Y%m%dT%H` 와 같다
USER_AGENT = "airlens-data-forecast-slot-guard"
SLOT_HOURS = 6  # 수집 주기이자 칸의 길이
MAX_BYTES = 4 * 1024 * 1024  # 응답 상한 — forecast.json 은 약 180 KB(75개 도시, 10-17 실측)라 20배가 넘는 여유다
TIMEOUT_S = 30


class FetchError(Exception):
    """조회 실패. 메시지는 고정 문구·상태 코드·예외 이름뿐이라(원격이 준 글자를 담지 않는다) 그대로 경고에 실린다."""


@dataclass(frozen=True)
class Decision:
    skip: bool
    reason: str


def fetch_forecast(url: str = FORECAST_URL) -> bytes:
    """공개 forecast.json 본문. 못 받으면 FetchError — 200 이 아니거나, 본문이 `MAX_BYTES` 를 넘거나, 네트워크 오류다."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310 — 고정 https URL
            if response.status != 200:  # 204·206 처럼 성공이어도 본문이 온전한 예보라는 보장이 없다
                raise FetchError(f"HTTP {response.status}")
            body = response.read(MAX_BYTES + 1)  # 상한보다 한 바이트 더 읽어 넘었는지만 본다
    except urllib.error.HTTPError as exc:
        raise FetchError(f"HTTP {exc.code}") from None  # 사유 문구는 원격이 준 글자라 싣지 않는다
    except (OSError, http.client.HTTPException) as exc:  # URLError·타임아웃·SSLError·IncompleteRead 포함
        raise FetchError(type(exc).__name__) from None
    if len(body) > MAX_BYTES:
        raise FetchError("응답 크기 상한 초과")
    return body


class _KeepMethodRedirect(urllib.request.HTTPRedirectHandler):
    """리다이렉트를 따라가도 요청 방식을 그대로 둔다 — 기본 처리기는 HEAD 를 GET 으로 바꿔 따라가(3.11 에서 확인) 본문을 받게 된다."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.method = req.get_method()
        return redirected


def head_exists(url: str) -> bool:
    """공개 URL 에 파일이 있는지 HEAD 로 묻는다(본문은 읽지 않는다). 리다이렉트를 따라간 뒤 200 이면 True, 404 면 False.

    그 밖의 상태·네트워크 오류·시간 초과는 있는지 모르는 것이라 FetchError — 메시지는 `fetch_forecast` 와 같이 상태 코드나 예외 이름뿐이다.
    """
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    opener = urllib.request.build_opener(_KeepMethodRedirect)
    try:
        with opener.open(request, timeout=TIMEOUT_S) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        exc.close()  # 오류 응답도 연결을 쥐고 있다
        if exc.code == 404:
            return False
        raise FetchError(f"HTTP {exc.code}") from None  # 사유 문구는 원격이 준 글자라 싣지 않는다
    except (OSError, http.client.HTTPException) as exc:  # URLError·타임아웃·SSLError 포함
        raise FetchError(type(exc).__name__) from None
    if status != 200:  # 204·206 처럼 성공이어도 그 파일이 있다는 보장이 없다
        raise FetchError(f"HTTP {status}")
    return True


def parse_issue(body: bytes, warnings: list[str]) -> datetime | None:
    """본문의 `generated_at` 을 UTC 시각으로. 못 읽으면 고정 문구 경고를 모으고 None.

    시간대가 없는 값은 읽지 않는다 — 어느 시계의 시각인지 모르는 값으로 칸을 정하면 엉뚱한 칸에서 건너뛴다.
    """
    # ponytail: 본문 전체를 받아 파싱한다(약 180 KB). `generated_at` 하나만 필요하다 — upgrade: 앞부분만 읽기
    try:
        doc = json.loads(body)
    except (ValueError, RecursionError):  # 깊게 중첩된 본문은 ValueError 가 아니라 RecursionError 다
        warnings.append("forecast.json 이 JSON 이 아니다")
        return None
    if not isinstance(doc, dict):
        warnings.append("forecast.json 이 JSON 객체가 아니다")
        return None
    value = doc.get("generated_at")
    if not isinstance(value, str):
        warnings.append("forecast.json 에 문자열 generated_at 이 없다")
        return None
    try:
        issue = datetime.fromisoformat(value)
    except ValueError:
        warnings.append("forecast.json 의 generated_at 이 ISO 시각이 아니다")
        return None
    if issue.utcoffset() is None:
        warnings.append("forecast.json 의 generated_at 에 시간대가 없다")
        return None
    try:
        return issue.astimezone(timezone.utc)
    except OverflowError:  # 범위 밖 시각(`0001-01-01T00:00:00+14:00`)은 UTC 로 바꾸다 OverflowError
        warnings.append("forecast.json 의 generated_at 을 UTC 로 바꿀 수 없다")
        return None


def slot_of(moment: datetime) -> datetime:
    """`moment`(tz 있음)가 속한 6시간 칸의 시작(UTC). UTC 로 바꾼 뒤 시를 내린다 — 오프셋 시각을 먼저 내리면 칸이 어긋난다."""
    utc = moment.astimezone(timezone.utc)
    return utc.replace(hour=utc.hour - utc.hour % SLOT_HOURS, minute=0, second=0, microsecond=0)


def decide(force: bool, issue: datetime | None, now: datetime) -> Decision:
    """칸 규칙으로 건너뛸지 정한다. 순수 함수 — 조회는 하지 않는다(그 칸의 archive 는 `confirm_archive` 가 따로 본다). `issue`·`now` 는 tz 있는 시각이고, 모르면(issue 가 None) 수집한다."""
    if force:
        return Decision(False, "force=true — 같은 6시간 칸에 이미 수집됐어도 다시 수집한다")
    if issue is None:
        return Decision(False, "예보 발행 시각을 알 수 없어 판정 없이 수집한다")
    if issue > now:
        return Decision(False, "예보 발행 시각이 지금보다 뒤라 믿지 않고 수집한다")
    slot = slot_of(now).isoformat()
    if slot_of(issue) == slot_of(now):
        return Decision(True, f"6시간 칸 {slot} 에 이미 수집됐다 — 예보 발행 {issue.isoformat()}")
    return Decision(False, f"6시간 칸 {slot} 에는 아직 수집이 없다 — 마지막 예보 발행 {issue.isoformat()}")


def archive_stamps(issue: datetime) -> tuple[str, str]:
    """`issue` 의 archive 파일명에 붙었을 STAMP 후보 — 발행한 시(UTC)와 그다음 시. 워크플로가 업로드 뒤에 STAMP 를 정해서 그 사이 시가 넘어갈 수 있다."""
    utc = issue.astimezone(timezone.utc)
    return utc.strftime(STAMP_FORMAT), (utc + timedelta(hours=1)).strftime(STAMP_FORMAT)


def confirm_archive(decision: Decision, issue: datetime, head: Callable[[str], bool], warnings: list[str]) -> Decision:
    """칸 규칙이 건너뛰라고 한 `decision` 을 그 예보의 archive 가 있을 때만 그대로 둔다. 둘 다 없으면 수집으로 바꾸고, 있는지 모르면 경고하고 수집한다."""
    unknown: FetchError | None = None
    for stamp in archive_stamps(issue):
        try:
            if head(ARCHIVE_URL.format(stamp=stamp)):
                return decision  # 하나라도 있으면 올라간 것이다 — 앞 후보를 모르고 지나쳤어도
        except FetchError as exc:
            if unknown is None:
                unknown = exc
    slot = slot_of(issue).isoformat()
    if unknown is not None:
        warnings.append(f"forecast archive 가 있는지 확인하지 못했다 ({unknown})")
        return Decision(False, f"6시간 칸 {slot} 에 예보는 있지만 archive 가 있는지 확인하지 못해 다시 수집한다 — 예보 발행 {issue.isoformat()}")
    return Decision(False, f"6시간 칸 {slot} 에 예보는 있지만 archive 가 없다 — 앞 run 이 archive 전에 실패했을 수 있어 다시 수집한다 — 예보 발행 {issue.isoformat()}")


def check(force: bool, fetch: Callable[[], bytes], now: Callable[[], datetime], head: Callable[[str], bool]) -> tuple[Decision, list[str]]:
    """(판정, 경고들). `force` 면 공개 저장소를 조회하지 않는다. `now` 는 조회가 끝난 뒤에 읽는다 — 조회가 느려도 issue 가 앞서 보이지 않게.

    `head` 는 archive URL 을 받아 있으면 True, 없으면 False, 모르면 FetchError. 칸 규칙이 건너뛰라고 할 때만 부른다.
    """
    warnings: list[str] = []
    issue = None
    if not force:
        try:
            issue = parse_issue(fetch(), warnings)
        except FetchError as exc:
            warnings.append(f"forecast.json 을 읽지 못했다 ({exc})")
    current = now()
    if issue is not None and issue > current:
        warnings.append(f"예보 발행 시각 {issue.isoformat()} 이 지금 {current.isoformat()} 보다 뒤다")
    decision = decide(force, issue, current)
    if decision.skip:  # 같은 칸에 예보가 있다 — 그 칸의 archive 까지 있어야 건너뛴다
        decision = confirm_archive(decision, issue, head, warnings)
    return decision, warnings


def _append(env: Mapping[str, str], key: str, text: str) -> None:
    """GitHub Actions 가 알려 준 파일(`$GITHUB_OUTPUT`·`$GITHUB_STEP_SUMMARY`)에 덧붙인다. 환경변수가 없으면 건너뛴다."""
    path = env.get(key)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)


def main(
    argv: list[str] | None = None,
    *,
    fetch: Callable[[], bytes] | None = None,
    env: Mapping[str, str] | None = None,
    now: Callable[[], datetime] | None = None,
    head: Callable[[str], bool] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", required=True, choices=("true", "false"))
    args = parser.parse_args(argv)
    env = os.environ if env is None else env

    decision, warnings = check(args.force == "true", fetch or fetch_forecast, now or (lambda: datetime.now(timezone.utc)), head or head_exists)
    verdict = "건너뜀" if decision.skip else "수집"
    for warning in warnings:
        print(f"::warning::forecast_slot_guard: {warning}")
    print(f"forecast_slot_guard: {verdict} — {decision.reason}")
    _append(env, "GITHUB_OUTPUT", f"skip={'true' if decision.skip else 'false'}\n")
    _append(env, "GITHUB_STEP_SUMMARY", f"### forecast 수집 가드\n\n- 판정: **{verdict}** — {decision.reason}\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
