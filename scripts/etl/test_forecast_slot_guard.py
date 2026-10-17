"""forecast_slot_guard 시험 — 판정은 순수 함수, 조회는 가짜 함수·가짜 응답이라 외부 네트워크가 0 이다 (AAA). HEAD 의 리다이렉트·시간 초과만 127.0.0.1 임시 서버로 본다."""
from __future__ import annotations

import http.client
import http.server
import json
import re
import socket
import subprocess
import sys
import threading
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Self

import pytest

import forecast_slot_guard as g

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/forecast-collect.yml"
SCRIPT = Path(g.__file__).resolve()
MARKER = "REMOTE-MARKER"  # 원격이 준 글자가 출력에 새는지 찾는 표지

NOW = datetime(2026, 10, 17, 3, 15, tzinfo=timezone.utc)  # 칸 [00:00, 06:00) 의 한가운데


def at(text: str) -> datetime:
    return datetime.fromisoformat(text)


def doc(**fields: object) -> bytes:
    return json.dumps(fields).encode()


def forecast(generated_at: object = "2026-10-17T00:55:12Z") -> bytes:
    return doc(generated_at=generated_at, cities=[])


def serving(body: bytes):
    return lambda: body


def failing(exc: BaseException):
    def fetch() -> bytes:
        raise exc

    return fetch


def untouched() -> bytes:
    raise AssertionError("조회하면 안 된다")


def archive_url(stamp: str) -> str:
    return f"https://huggingface.co/datasets/Robeedau/airlens-live/resolve/main/aq-data/forecast-archive/forecast_{stamp}.json"


def untouched_head(url: str) -> bool:
    raise AssertionError(f"HEAD 를 보내면 안 된다: {url}")


class FakeHead:
    """`head` 대역 — STAMP 별로 정해 둔 결과(있음·없음·예외)만 받고(그 밖의 URL 은 시험이 실패한다), 물은 STAMP 를 순서대로 기록한다."""

    def __init__(self, outcomes: dict[str, bool | BaseException]) -> None:
        self.outcomes = {archive_url(stamp): outcome for stamp, outcome in outcomes.items()}
        self.calls: list[str] = []

    def __call__(self, url: str) -> bool:
        self.calls.append(url)
        outcome = self.outcomes[url]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def stamps(self) -> list[str]:
        return [url.rsplit("_", 1)[1].removesuffix(".json") for url in self.calls]


# ── 칸 ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("moment", "slot"),
    [
        ("2026-10-17T00:00:00+00:00", "2026-10-17T00:00:00+00:00"),
        ("2026-10-17T05:59:59.999999+00:00", "2026-10-17T00:00:00+00:00"),
        ("2026-10-17T06:00:00+00:00", "2026-10-17T06:00:00+00:00"),
        ("2026-10-17T11:59:59+00:00", "2026-10-17T06:00:00+00:00"),
        ("2026-10-17T12:00:00+00:00", "2026-10-17T12:00:00+00:00"),
        ("2026-10-17T17:59:59+00:00", "2026-10-17T12:00:00+00:00"),
        ("2026-10-17T18:00:00+00:00", "2026-10-17T18:00:00+00:00"),
        ("2026-10-17T23:59:59+00:00", "2026-10-17T18:00:00+00:00"),
        ("2026-10-17T08:55:00+09:00", "2026-10-16T18:00:00+00:00"),  # 한국 시각 08:55 = UTC 전날 23:55
        ("2026-10-17T11:35:00+05:30", "2026-10-17T06:00:00+00:00"),  # UTC 06:05 (오프셋 시각의 시를 먼저 내리면 00:30Z 로 어긋난다)
        ("2026-10-17T02:30:00-03:00", "2026-10-17T00:00:00+00:00"),  # UTC 05:30
    ],
)
def test_slot_is_the_utc_hour_floored_to_a_multiple_of_six(moment, slot):
    # Act
    result = g.slot_of(at(moment))

    # Assert
    assert result.isoformat() == slot


@pytest.mark.parametrize(
    ("issue", "now", "skip"),
    [
        ("2026-10-17T05:59:59+00:00", "2026-10-17T06:00:00+00:00", False),  # 칸 경계 직전 ↔ 직후는 다른 칸
        ("2026-10-17T06:00:00+00:00", "2026-10-17T11:59:59+00:00", True),  # 같은 칸의 처음과 끝
        ("2026-10-17T00:00:00+00:00", "2026-10-17T05:59:59+00:00", True),
        ("2026-10-17T06:00:00+00:00", "2026-10-17T06:00:00+00:00", True),  # 같은 순간(미래가 아니다)
        ("2026-10-16T18:00:00+00:00", "2026-10-16T23:59:59+00:00", True),
        ("2026-10-16T23:59:59+00:00", "2026-10-17T00:00:00+00:00", False),  # 자정을 넘기면 다른 칸
        ("2026-10-16T21:10:03+00:00", "2026-10-17T00:50:00+00:00", False),  # 늦은 백업이 21:10 에 끝났다 → 00:50 정시 dispatch 는 수집한다
        ("2026-10-17T00:20:00+00:00", "2026-10-17T00:50:00+00:00", True),  # 제때 온 백업이 00:20 에 먼저 수집했다 → 같은 칸이라 00:50 정시 dispatch 는 건너뛴다
        ("2026-10-17T06:05:00+00:00", "2026-10-17T06:50:00+00:00", True),  # 앞 칸 백업이 6시간 가까이 늦어 06:05 에 수집했다 → 06:50 정시 dispatch 는 건너뛴다
        ("2026-10-17T00:55:12+00:00", "2026-10-17T03:15:00+00:00", True),  # 정시 dispatch 가 수집한 뒤 늦게 온 백업은 건너뛴다
        ("2026-10-16T06:10:00+00:00", "2026-10-17T06:10:00+00:00", False),  # 시각이 같아도 날짜가 다르면 다른 칸이다
        ("2026-10-17T08:55:00+09:00", "2026-10-17T00:10:00+00:00", False),  # UTC 전날 23:55 ↔ 00:10 — 오프셋을 UTC 로 바꿔 비교한다
        ("2026-10-17T09:55:00+09:00", "2026-10-17T03:00:00+00:00", True),  # UTC 00:55 ↔ 03:00
        ("2026-10-17T11:35:00+05:30", "2026-10-17T07:00:00+00:00", True),  # UTC 06:05 ↔ 07:00
    ],
)
def test_skip_only_when_the_issue_is_in_the_same_utc_slot_as_now(issue, now, skip):
    # Act
    decision = g.decide(False, at(issue), at(now))

    # Assert
    assert decision.skip is skip


def test_the_decision_reason_names_the_slot_and_the_issue():
    # Act
    skipped = g.decide(False, at("2026-10-17T00:55:12+00:00"), NOW)
    run = g.decide(False, at("2026-10-16T21:10:03+00:00"), NOW)

    # Assert
    assert skipped.reason == "6시간 칸 2026-10-17T00:00:00+00:00 에 이미 수집됐다 — 예보 발행 2026-10-17T00:55:12+00:00"
    assert run.reason == "6시간 칸 2026-10-17T00:00:00+00:00 에는 아직 수집이 없다 — 마지막 예보 발행 2026-10-16T21:10:03+00:00"


def test_force_never_skips_even_when_the_slot_is_already_collected():
    # Act
    decision = g.decide(True, at("2026-10-17T00:55:12+00:00"), NOW)

    # Assert
    assert decision.skip is False and "force" in decision.reason


def test_an_unknown_issue_means_collect():
    # Act
    decision = g.decide(False, None, NOW)

    # Assert
    assert decision.skip is False and "알 수 없어" in decision.reason


@pytest.mark.parametrize("ahead", [timedelta(seconds=1), timedelta(hours=1), timedelta(days=400)])
def test_an_issue_later_than_now_is_not_trusted_even_inside_the_same_slot(ahead):
    # Act — 같은 칸 안의 1초 뒤여도 믿지 않는다(시계가 어긋난 값으로 건너뛰지 않는다)
    decision, warnings = g.check(False, serving(forecast((NOW + ahead).isoformat())), lambda: NOW, untouched_head)

    # Assert
    assert decision.skip is False and "뒤라" in decision.reason
    assert len(warnings) == 1 and "보다 뒤다" in warnings[0]


# ── check: 조회 + 판정 ───────────────────────────────────────────────────────
def test_check_skips_when_the_fetched_issue_is_in_the_current_slot():
    # Act
    decision, warnings = g.check(False, serving(forecast("2026-10-17T00:55:12Z")), lambda: NOW, FakeHead({"20261017T00": True}))

    # Assert
    assert (decision.skip, warnings) == (True, [])


def test_check_collects_for_an_issue_from_the_previous_slot_without_a_warning():
    # Act
    decision, warnings = g.check(False, serving(forecast("2026-10-16T21:10:03Z")), lambda: NOW, untouched_head)

    # Assert
    assert (decision.skip, warnings) == (False, [])


def test_check_reads_the_clock_after_the_fetch():
    # Arrange — 조회가 칸 경계를 넘겨 끝난다: 시작할 때 읽은 시계였다면 issue(05:59:58)가 같은 칸이라 건너뛴다
    clock = {"now": at("2026-10-17T05:59:59+00:00")}

    def slow_fetch() -> bytes:
        clock["now"] = at("2026-10-17T06:00:03+00:00")
        return forecast("2026-10-17T05:59:58Z")

    # Act
    decision, warnings = g.check(False, slow_fetch, lambda: clock["now"], untouched_head)

    # Assert
    assert (decision.skip, warnings) == (False, [])


def test_force_does_not_touch_the_network():
    # Act
    decision, warnings = g.check(True, untouched, lambda: NOW, untouched_head)

    # Assert
    assert (decision.skip, warnings) == (False, [])


UNREADABLE = [
    (g.FetchError("HTTP 500"), "HTTP 500"),
    (g.FetchError("URLError"), "URLError"),
    (g.FetchError("응답 크기 상한 초과"), "상한"),
    (b"", "JSON 이 아니다"),
    (b"<html>maintenance</html>", "JSON 이 아니다"),
    (b"\xff\xfe\x00", "JSON 이 아니다"),
    (b"[" * 200_000, "JSON 이 아니다"),  # 깊게 중첩된 본문은 ValueError 가 아니라 RecursionError 다
    (b"[]", "JSON 객체가 아니다"),
    (b"null", "JSON 객체가 아니다"),
    (b'"2026-10-17T00:55:12Z"', "JSON 객체가 아니다"),
    (doc(cities=[]), "generated_at"),
    (doc(generated_at=None), "generated_at"),
    (doc(generated_at=12), "generated_at"),
    (doc(generated_at=["2026-10-17T00:55:12Z"]), "generated_at"),
    (forecast(""), "ISO 시각이 아니다"),
    (forecast("garbage"), "ISO 시각이 아니다"),
    (forecast("2026-10-17T00:55:12Z "), "ISO 시각이 아니다"),
    (forecast("2026-10-17T00:55:12"), "시간대가 없다"),
    (forecast("2026-10-17"), "시간대가 없다"),
    (forecast("0001-01-01T00:00:00+14:00"), "UTC 로 바꿀 수 없다"),  # 범위 밖 시각은 UTC 로 바꾸다 OverflowError
]
UNREADABLE_IDS = [
    "http-500", "network-error", "too-large", "empty", "html", "not-utf8", "deeply-nested", "array", "null", "string",
    "no-generated-at", "null-generated-at", "number", "list", "empty-string", "garbage", "trailing-space", "no-timezone",
    "date-only", "out-of-range",
]


@pytest.mark.parametrize(("result", "expected"), UNREADABLE, ids=UNREADABLE_IDS)
def test_an_unreadable_forecast_means_collect_with_one_warning(result, expected):
    # Arrange
    fetch = failing(result) if isinstance(result, BaseException) else serving(result)

    # Act
    decision, warnings = g.check(False, fetch, lambda: NOW, untouched_head)

    # Assert — 모르면 수집한다
    assert decision.skip is False
    assert len(warnings) == 1 and expected in warnings[0]


def test_unexpected_errors_from_the_fetch_are_not_swallowed():
    # Act / Assert — FetchError 만 "조회 실패" 다. 그 밖의 예외는 버그라 드러나야 한다
    with pytest.raises(RuntimeError):
        g.check(False, failing(RuntimeError("bug")), lambda: NOW, untouched_head)


def test_the_generated_at_format_the_collector_writes_is_readable():
    # Arrange — 수집기가 쓰는 형식을 워크플로 본문에서 읽는다(수집기가 형식을 바꾸면 가드가 조용히 못 읽게 되는 것을 막는다)
    fmt = re.search(r'"generated_at": datetime\.now\(timezone\.utc\)\.strftime\("([^"]+)"\)', WORKFLOW.read_text(encoding="utf-8"))
    assert fmt, "수집기의 generated_at 형식을 못 찾았다 — 추출이 깨졌다(vacuous pass 방지)"
    written = datetime(2026, 10, 17, 0, 55, 12, tzinfo=timezone.utc).strftime(fmt.group(1))

    # Act
    warnings: list[str] = []
    issue = g.parse_issue(forecast(written), warnings)

    # Assert
    assert issue == datetime(2026, 10, 17, 0, 55, 12, tzinfo=timezone.utc) and warnings == []


def test_parse_issue_returns_utc_whatever_offset_the_value_has():
    # Act
    warnings: list[str] = []
    issue = g.parse_issue(forecast("2026-10-17T09:55:12+09:00"), warnings)

    # Assert
    assert issue is not None and issue.isoformat() == "2026-10-17T00:55:12+00:00" and warnings == []


# ── archive: 칸이 같아도 그 칸의 forecast archive 가 있어야 건너뛴다 ──────────────
SAME_SLOT_ISSUE = "2026-10-17T00:55:12Z"  # NOW 와 같은 칸
STAMP_H, STAMP_H1 = "20261017T00", "20261017T01"  # SAME_SLOT_ISSUE 의 archive STAMP 후보 — 발행한 시와 그다음 시


def check_same_slot(head, issue=SAME_SLOT_ISSUE, now=NOW):
    return g.check(False, serving(forecast(issue)), lambda: now, head)


@pytest.mark.parametrize(
    ("issue", "stamps"),
    [
        ("2026-10-17T00:55:12+00:00", ("20261017T00", "20261017T01")),
        ("2026-10-17T05:59:58+00:00", ("20261017T05", "20261017T06")),  # 칸 경계 직전 — 다음 시는 다음 칸의 시다
        ("2026-10-16T23:59:59+00:00", ("20261016T23", "20261017T00")),  # 자정을 넘는 다음 시
        ("2026-10-31T23:30:00+00:00", ("20261031T23", "20261101T00")),  # 월말
        ("2026-12-31T23:30:00+00:00", ("20261231T23", "20270101T00")),  # 연말
        ("2026-10-17T09:55:12+09:00", ("20261017T00", "20261017T01")),  # 한국 시각 09시 = UTC 00시
        ("2026-10-17T00:30:00+09:00", ("20261016T15", "20261016T16")),  # 한국 시각 00:30 = UTC 전날 15:30
        ("2026-10-17T11:35:00+05:30", ("20261017T06", "20261017T07")),  # UTC 06:05
        ("2026-10-17T02:30:00-03:00", ("20261017T05", "20261017T06")),  # UTC 05:30
    ],
)
def test_the_archive_stamps_are_the_utc_hour_of_the_issue_and_the_next_hour(issue, stamps):
    # Act
    result = g.archive_stamps(at(issue))

    # Assert
    assert result == stamps


def test_the_slot_is_skipped_when_the_archive_of_the_issue_hour_exists():
    # Arrange
    head = FakeHead({STAMP_H: True})

    # Act
    decision, warnings = check_same_slot(head)

    # Assert — 첫 후보가 있으면 둘째는 묻지 않는다
    assert (decision.skip, warnings) == (True, [])
    assert head.stamps == [STAMP_H]


def test_the_slot_is_skipped_when_only_the_archive_of_the_next_hour_exists():
    # Arrange — 업로드 뒤 시가 넘어가 STAMP 가 다음 시가 된 run
    head = FakeHead({STAMP_H: False, STAMP_H1: True})

    # Act
    decision, warnings = check_same_slot(head)

    # Assert
    assert (decision.skip, warnings) == (True, [])
    assert head.stamps == [STAMP_H, STAMP_H1]


def test_the_slot_is_collected_again_when_the_archive_exists_under_neither_stamp():
    # Arrange — 앞 run 이 forecast.json 만 올리고 archive 업로드에서 실패했다
    head = FakeHead({STAMP_H: False, STAMP_H1: False})

    # Act
    decision, warnings = check_same_slot(head)

    # Assert — 없다는 것은 정상 판정이라 경고가 아니다
    assert decision.skip is False and warnings == []
    assert head.stamps == [STAMP_H, STAMP_H1]
    assert "archive 가 없다" in decision.reason and "다시 수집한다" in decision.reason


UNKNOWN_ARCHIVE = [
    ({STAMP_H: g.FetchError("HTTP 500"), STAMP_H1: False}, "HTTP 500"),
    ({STAMP_H: False, STAMP_H1: g.FetchError("URLError")}, "URLError"),
    ({STAMP_H: g.FetchError("TimeoutError"), STAMP_H1: g.FetchError("HTTP 503")}, "TimeoutError"),  # 둘 다 모르면 경고는 하나다(첫 오류)
]


@pytest.mark.parametrize(("outcomes", "shown"), UNKNOWN_ARCHIVE, ids=["first-unknown", "second-unknown", "both-unknown"])
def test_an_archive_that_cannot_be_checked_means_collect_with_one_warning(outcomes, shown):
    # Arrange
    head = FakeHead(outcomes)

    # Act
    decision, warnings = check_same_slot(head)

    # Assert — 모르면 수집한다
    assert decision.skip is False and "확인하지 못해" in decision.reason
    assert warnings == [f"forecast archive 가 있는지 확인하지 못했다 ({shown})"]
    assert head.stamps == [STAMP_H, STAMP_H1]


def test_an_archive_found_under_the_second_stamp_skips_even_if_the_first_could_not_be_checked():
    # Arrange
    head = FakeHead({STAMP_H: g.FetchError("HTTP 500"), STAMP_H1: True})

    # Act
    decision, warnings = check_same_slot(head)

    # Assert — 하나라도 있다고 확인되면 올라간 것이다
    assert (decision.skip, warnings) == (True, [])


def test_the_collect_reasons_name_the_slot_and_the_issue():
    # Act
    absent, _ = check_same_slot(FakeHead({STAMP_H: False, STAMP_H1: False}))
    unknown, _ = check_same_slot(FakeHead({STAMP_H: g.FetchError("HTTP 500"), STAMP_H1: False}))

    # Assert
    assert absent.reason == "6시간 칸 2026-10-17T00:00:00+00:00 에 예보는 있지만 archive 가 없다 — 앞 run 이 archive 전에 실패했을 수 있어 다시 수집한다 — 예보 발행 2026-10-17T00:55:12+00:00"
    assert unknown.reason == "6시간 칸 2026-10-17T00:00:00+00:00 에 예보는 있지만 archive 가 있는지 확인하지 못해 다시 수집한다 — 예보 발행 2026-10-17T00:55:12+00:00"


def test_force_sends_no_head_even_when_the_slot_already_has_a_forecast():
    # Act
    decision, warnings = g.check(True, serving(forecast(SAME_SLOT_ISSUE)), lambda: NOW, untouched_head)

    # Assert
    assert (decision.skip, warnings) == (False, [])


@pytest.mark.parametrize("issue", ["2026-10-16T21:10:03Z", "2026-10-16T23:59:59Z", "2026-10-10T00:55:12Z"], ids=["previous-slot", "just-before-the-slot", "a-week-ago"])
def test_a_forecast_from_another_slot_sends_no_head(issue):
    # Act — 칸 규칙이 이미 수집하라고 한 경우라 archive 는 물을 필요가 없다
    decision, warnings = check_same_slot(untouched_head, issue)

    # Assert
    assert (decision.skip, warnings) == (False, [])


def test_the_stamps_asked_for_an_offset_issue_are_utc_based():
    # Arrange — 한국 시각 09:55 = UTC 00:55. 현지 시로 만들면 T09·T10 을 묻는다
    head = FakeHead({STAMP_H: False, STAMP_H1: False})

    # Act
    check_same_slot(head, "2026-10-17T09:55:12+09:00")

    # Assert
    assert head.stamps == [STAMP_H, STAMP_H1]


def test_the_stamps_asked_just_before_midnight_roll_over_to_the_next_day():
    # Arrange — 발행 23:59:30, 지금 23:59:50: 같은 칸이다. archive STAMP 는 자정을 넘겨 다음 날 00시일 수 있다
    head = FakeHead({"20261016T23": False, "20261017T00": True})

    # Act
    decision, warnings = check_same_slot(head, "2026-10-16T23:59:30Z", at("2026-10-16T23:59:50+00:00"))

    # Assert
    assert (decision.skip, warnings) == (True, [])
    assert head.stamps == ["20261016T23", "20261017T00"]


def test_the_stamps_asked_at_the_end_of_a_month_roll_over_to_the_next_month():
    # Arrange
    head = FakeHead({"20261031T23": False, "20261101T00": True})

    # Act
    decision, _ = check_same_slot(head, "2026-10-31T23:58:00Z", at("2026-10-31T23:59:00+00:00"))

    # Assert
    assert decision.skip is True and head.stamps == ["20261031T23", "20261101T00"]


def test_unexpected_errors_from_the_head_are_not_swallowed():
    # Arrange
    def head(url: str) -> bool:
        raise RuntimeError("bug")

    # Act / Assert — FetchError 만 "모름" 이다. 그 밖의 예외는 버그라 드러나야 한다
    with pytest.raises(RuntimeError):
        check_same_slot(head)


# ── 워크플로와의 약속 ────────────────────────────────────────────────────────
def test_the_guard_reads_the_file_the_collector_publishes():
    # Arrange
    text = WORKFLOW.read_text(encoding="utf-8")
    base = re.search(r'HF_LIVE_BASE = "([^"]+)"', text)
    assert base, "수집기의 HF_LIVE_BASE 를 못 찾았다 — 추출이 깨졌다(vacuous pass 방지)"

    # Act / Assert — 읽는 곳과 쓰는 곳이 어긋나면 가드가 엉뚱한 파일을 보고 판정한다
    assert g.FORECAST_URL == f"{base.group(1)}/aq-data/forecast.json"
    assert "--dest aq-data/forecast.json" in text


def test_the_guard_asks_for_the_file_the_archive_step_uploads():
    # Arrange
    text = WORKFLOW.read_text(encoding="utf-8")
    base = re.search(r'HF_LIVE_BASE = "([^"]+)"', text)
    stamps = re.findall(r"STAMP=\$\((date[^)]*)\)", text)
    dest = re.search(r'--dest "(aq-data/forecast-archive/forecast_\$\{STAMP\}\.json)"', text)
    assert base and stamps and dest, "워크플로의 archive 단계를 못 찾았다 — 추출이 깨졌다(vacuous pass 방지)"

    # Act / Assert — 가드가 묻는 이름이 워크플로가 올리는 이름과 어긋나면 늘 "없음" 으로 읽혀 건너뛰지 못한다
    assert set(stamps) == {f"date -u +{g.STAMP_FORMAT}"}  # UTC 시각의 STAMP 다 — 가드도 UTC 시로 계산한다
    assert g.ARCHIVE_URL == f"{base.group(1)}/" + dest.group(1).replace("${STAMP}", "{stamp}")
    assert g.ARCHIVE_URL.format(stamp=STAMP_H) == archive_url(STAMP_H)


def test_the_slot_length_is_the_collection_cadence_of_the_schedule():
    # Arrange
    cron = re.search(r"- cron: '([^']+)'", WORKFLOW.read_text(encoding="utf-8"))
    assert cron, "schedule cron 을 못 찾았다 — 추출이 깨졌다(vacuous pass 방지)"

    # Act
    hours = re.fullmatch(r"\d+ \*/(\d+) \* \* \*", cron.group(1))

    # Assert — 칸이 수집 주기와 다르면 "같은 칸" 이 한 번의 수집 단위가 아니다
    assert hours and int(hours.group(1)) == g.SLOT_HOURS


def test_the_response_cap_leaves_room_for_a_payload_shaped_like_the_real_one():
    # Arrange — 75개 도시 × 48시간
    hourly = [{"time": f"2026-10-17T{h % 24:02d}:00:00Z", "pm25": 12.3} for h in range(48)]
    city = {"name": "City-000", "lat": 37.5665, "lon": 126.978, "country_code": "KR", "source": "open-meteo", "hourly": hourly}
    payload = json.dumps({"generated_at": "2026-10-17T00:55:12Z", "model_version": "open-meteo-cams-v1", "source": "open-meteo", "cities": [city] * 75})

    # Act / Assert — 도시가 늘어도 상한에 걸려 매번 수집만 하게 되지 않는 여유
    assert g.MAX_BYTES >= 5 * len(payload)


# ── fetch_forecast: 응답 → 본문 또는 FetchError ──────────────────────────────
class FakeResponse:
    """`urlopen` 이 돌려주는 응답 대역. `read(n)` 은 실제처럼 최대 n 바이트를 준다."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body, self.status = body, status
        self.read_amounts: list[int] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self, amount: int = -1) -> bytes:
        self.read_amounts.append(amount)
        return self.body if amount < 0 else self.body[:amount]


def stub_urlopen(monkeypatch, outcome):
    """`urlopen` 을 바꿔치우고 (요청, timeout) 호출 기록을 돌려준다. outcome 이 예외면 던지고, 아니면 돌려준다."""
    calls: list[tuple[object, object]] = []

    def fake(request, timeout=None):
        calls.append((request, timeout))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(g.urllib.request, "urlopen", fake)
    return calls


def test_fetch_returns_the_body_of_a_200_response_with_a_timeout_and_a_bounded_read(monkeypatch):
    # Arrange
    response = FakeResponse(b'{"generated_at": "2026-10-17T00:55:12Z"}')
    calls = stub_urlopen(monkeypatch, response)

    # Act
    body = g.fetch_forecast()

    # Assert
    request, timeout = calls[0]
    assert body == response.body
    assert request.full_url == g.FORECAST_URL and request.get_header("User-agent") == "airlens-data-forecast-slot-guard"
    assert timeout == g.TIMEOUT_S
    assert response.read_amounts == [g.MAX_BYTES + 1]  # 상한보다 한 바이트 더 읽어 넘었는지만 본다


@pytest.mark.parametrize("status", [201, 204, 206])
def test_fetch_rejects_a_success_status_that_is_not_200(monkeypatch, status):
    # Arrange
    stub_urlopen(monkeypatch, FakeResponse(b"{}", status=status))

    # Act / Assert
    with pytest.raises(g.FetchError) as caught:
        g.fetch_forecast()
    assert str(caught.value) == f"HTTP {status}"


@pytest.mark.parametrize("code", [404, 429, 500, 503])
def test_fetch_reports_only_the_status_code_of_an_http_error(monkeypatch, code):
    # Arrange — 사유 문구는 원격이 준 글자다
    stub_urlopen(monkeypatch, urllib.error.HTTPError(g.FORECAST_URL, code, f"x\n::error::{MARKER}", {}, None))

    # Act / Assert
    with pytest.raises(g.FetchError) as caught:
        g.fetch_forecast()
    assert str(caught.value) == f"HTTP {code}"


NETWORK_ERRORS = [
    urllib.error.URLError(f"{MARKER}\n::error::x"),
    TimeoutError(MARKER),
    ConnectionResetError(MARKER),
    http.client.IncompleteRead(b"abc"),
    http.client.BadStatusLine(f"{MARKER}\n::error::x"),
]
NETWORK_ERROR_IDS = ["url-error", "timeout", "reset", "incomplete-read", "bad-status-line"]


@pytest.mark.parametrize("error", NETWORK_ERRORS, ids=NETWORK_ERROR_IDS)
def test_fetch_reports_only_the_exception_name_of_a_network_error(monkeypatch, error):
    # Arrange
    stub_urlopen(monkeypatch, error)

    # Act / Assert
    with pytest.raises(g.FetchError) as caught:
        g.fetch_forecast()
    assert str(caught.value) == type(error).__name__ and MARKER not in str(caught.value)


def test_fetch_accepts_a_body_of_exactly_the_cap_and_rejects_one_byte_more(monkeypatch):
    # Arrange
    monkeypatch.setattr(g, "MAX_BYTES", 10)
    stub_urlopen(monkeypatch, FakeResponse(b"x" * 10))

    # Act
    exact = g.fetch_forecast()
    stub_urlopen(monkeypatch, FakeResponse(b"x" * 11))

    # Assert
    assert exact == b"x" * 10
    with pytest.raises(g.FetchError) as caught:
        g.fetch_forecast()
    assert str(caught.value) == "응답 크기 상한 초과"


def test_fetch_does_not_swallow_unexpected_errors(monkeypatch):
    # Arrange
    stub_urlopen(monkeypatch, RuntimeError("bug"))

    # Act / Assert
    with pytest.raises(RuntimeError):
        g.fetch_forecast()


# ── head_exists: HEAD 응답 → 있음·없음·모름 ──────────────────────────────────
ARCHIVE = archive_url(STAMP_H)


def stub_opener(monkeypatch, outcome):
    """`build_opener` 를 바꿔치우고 (요청, timeout) 호출 기록을 돌려준다. outcome 이 예외면 던지고, 아니면 돌려준다."""
    calls: list[tuple[object, object]] = []

    class FakeOpener:
        def open(self, request, timeout=None):
            calls.append((request, timeout))
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    monkeypatch.setattr(g.urllib.request, "build_opener", lambda *handlers: FakeOpener())
    return calls


def test_head_asks_with_the_head_method_and_a_timeout_and_never_reads_the_body(monkeypatch):
    # Arrange
    response = FakeResponse(b"not-read")
    calls = stub_opener(monkeypatch, response)

    # Act
    exists = g.head_exists(ARCHIVE)

    # Assert
    request, timeout = calls[0]
    assert exists is True
    assert request.get_method() == "HEAD" and request.full_url == ARCHIVE
    assert request.get_header("User-agent") == "airlens-data-forecast-slot-guard"
    assert timeout == g.TIMEOUT_S
    assert response.read_amounts == []  # 본문은 읽지 않는다


def test_head_means_absent_for_a_404(monkeypatch):
    # Arrange — 사유 문구는 원격이 준 글자다
    stub_opener(monkeypatch, urllib.error.HTTPError(ARCHIVE, 404, f"x\n::error::{MARKER}", {}, None))

    # Act / Assert
    assert g.head_exists(ARCHIVE) is False


@pytest.mark.parametrize("status", [201, 202, 204, 206])
def test_head_does_not_take_a_success_status_other_than_200_for_present(monkeypatch, status):
    # Arrange
    stub_opener(monkeypatch, FakeResponse(b"", status=status))

    # Act / Assert
    with pytest.raises(g.FetchError) as caught:
        g.head_exists(ARCHIVE)
    assert str(caught.value) == f"HTTP {status}"


@pytest.mark.parametrize("code", [301, 302, 307, 400, 401, 403, 410, 429, 500, 502, 503])
def test_head_reports_only_the_status_code_of_any_other_http_error(monkeypatch, code):
    # Arrange
    stub_opener(monkeypatch, urllib.error.HTTPError(ARCHIVE, code, f"x\n::error::{MARKER}", {}, None))

    # Act / Assert — 404 만 "없음" 이다. 그 밖의 오류는 있는지 모른다
    with pytest.raises(g.FetchError) as caught:
        g.head_exists(ARCHIVE)
    assert str(caught.value) == f"HTTP {code}"


@pytest.mark.parametrize("error", NETWORK_ERRORS, ids=NETWORK_ERROR_IDS)
def test_head_reports_only_the_exception_name_of_a_network_error(monkeypatch, error):
    # Arrange
    stub_opener(monkeypatch, error)

    # Act / Assert
    with pytest.raises(g.FetchError) as caught:
        g.head_exists(ARCHIVE)
    assert str(caught.value) == type(error).__name__ and MARKER not in str(caught.value)


def test_head_does_not_swallow_unexpected_errors(monkeypatch):
    # Arrange
    stub_opener(monkeypatch, RuntimeError("bug"))

    # Act / Assert
    with pytest.raises(RuntimeError):
        g.head_exists(ARCHIVE)


@pytest.fixture
def loopback(monkeypatch):
    # 개발 머신의 프록시 설정이 루프백 요청을 가로채지 않게 한다(NO_PROXY 가 있으면 환경 프록시만 쓰고 시스템 프록시는 안 본다).
    for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")


@pytest.fixture
def local_server(loopback):
    """127.0.0.1 의 임시 HTTP 서버 — (기본 주소, 경로별 응답 `routes[경로] = (상태, 이동할 곳)`, 받은 (방식, 경로) 기록). 밖으로 나가는 연결은 없다."""
    routes: dict[str, tuple[int, str | None]] = {}
    seen: list[tuple[str, str]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self):
            seen.append((self.command, self.path))
            status, location = routes.get(self.path, (404, None))
            self.send_response(status)
            if location:
                self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = do_HEAD = answer

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)  # 종료 때 기다리는 시간이 짧도록
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", routes, seen
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_head_follows_a_redirect_with_head_all_the_way(local_server):
    # Arrange — 실제 공개 저장소는 있는 파일의 HEAD 에 307 로 응답한다. urllib 의 기본 처리기는 3.11 에서 이동한 뒤 GET 으로 바꿔 본문을 받는다
    base, routes, seen = local_server
    routes.update({"/archive": (307, "/final"), "/final": (200, None)})

    # Act
    exists = g.head_exists(f"{base}/archive")

    # Assert
    assert exists is True
    assert seen == [("HEAD", "/archive"), ("HEAD", "/final")]


@pytest.mark.parametrize(
    ("routes", "expected"),
    [
        ({"/archive": (200, None)}, True),
        ({"/archive": (404, None)}, False),
        ({"/archive": (307, "/final"), "/final": (404, None)}, False),
        ({"/archive": (500, None)}, "HTTP 500"),
        ({"/archive": (307, "/final"), "/final": (503, None)}, "HTTP 503"),
        ({"/archive": (204, None)}, "HTTP 204"),
        ({"/archive": (302, "/archive")}, "HTTP 302"),  # 끝없는 이동은 urllib 이 몇 번 따라가다 멈춘다
    ],
    ids=["found", "missing", "redirect-to-missing", "server-error", "redirect-to-server-error", "no-content", "redirect-loop"],
)
def test_head_over_a_real_socket_gives_present_absent_or_unknown(local_server, routes, expected):
    # Arrange
    base, served, _ = local_server
    served.update(routes)

    # Act
    if isinstance(expected, bool):
        result = g.head_exists(f"{base}/archive")
    else:
        with pytest.raises(g.FetchError) as caught:
            g.head_exists(f"{base}/archive")
        result = str(caught.value)

    # Assert
    assert result == expected


def test_head_gives_up_on_a_server_that_never_answers_once_the_timeout_has_passed(loopback, monkeypatch):
    # Arrange — 연결은 받고 응답은 하지 않는 소켓
    monkeypatch.setattr(g, "TIMEOUT_S", 0.3)
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(1)

        # Act / Assert
        with pytest.raises(g.FetchError) as caught:
            g.head_exists(f"http://127.0.0.1:{silent.getsockname()[1]}/archive")
    assert str(caught.value) == "TimeoutError"


# ── main: GITHUB_OUTPUT · 요약 · 종료 코드 ───────────────────────────────────
def run_main(argv, tmp_path, capsys, *, fetch=untouched, now=lambda: NOW, head=untouched_head, with_files=True):
    out_file, summary_file = tmp_path / "output.txt", tmp_path / "summary.md"
    env = {"GITHUB_OUTPUT": str(out_file), "GITHUB_STEP_SUMMARY": str(summary_file)} if with_files else {}
    code = g.main(argv, fetch=fetch, env=env, now=now, head=head)
    stdout = capsys.readouterr().out

    def read(path: Path) -> str | None:
        return path.read_text(encoding="utf-8") if path.exists() else None

    return code, stdout, read(out_file), read(summary_file)


def test_main_writes_skip_true_and_a_summary_when_the_slot_is_already_collected(tmp_path, capsys):
    # Act
    code, stdout, output, summary = run_main(["--force", "false"], tmp_path, capsys, fetch=serving(forecast("2026-10-17T00:55:12Z")), head=FakeHead({"20261017T00": True}))

    # Assert
    result = "6시간 칸 2026-10-17T00:00:00+00:00 에 이미 수집됐다 — 예보 발행 2026-10-17T00:55:12+00:00"
    assert code == 0
    assert output == "skip=true\n"
    assert stdout == f"forecast_slot_guard: 건너뜀 — {result}\n"
    assert summary == f"### forecast 수집 가드\n\n- 판정: **건너뜀** — {result}\n\n"


def test_main_writes_skip_false_and_a_summary_when_the_slot_has_no_collection_yet(tmp_path, capsys):
    # Act
    code, stdout, output, summary = run_main(["--force", "false"], tmp_path, capsys, fetch=serving(forecast("2026-10-16T21:10:03Z")))

    # Assert
    result = "6시간 칸 2026-10-17T00:00:00+00:00 에는 아직 수집이 없다 — 마지막 예보 발행 2026-10-16T21:10:03+00:00"
    assert code == 0
    assert output == "skip=false\n"
    assert stdout == f"forecast_slot_guard: 수집 — {result}\n"
    assert summary == f"### forecast 수집 가드\n\n- 판정: **수집** — {result}\n\n"


def test_main_appends_to_existing_output_and_summary_files(tmp_path, capsys):
    # Arrange
    (tmp_path / "output.txt").write_text("earlier=1\n", encoding="utf-8")
    (tmp_path / "summary.md").write_text("# earlier\n", encoding="utf-8")

    # Act
    _, _, output, summary = run_main(["--force", "false"], tmp_path, capsys, fetch=serving(forecast("2026-10-16T21:10:03Z")))

    # Assert
    assert output == "earlier=1\nskip=false\n"
    assert summary.startswith("# earlier\n### forecast 수집 가드\n")


def test_main_force_true_collects_without_any_request(tmp_path, capsys):
    # Act
    code, stdout, output, summary = run_main(["--force", "true"], tmp_path, capsys)

    # Assert
    assert code == 0 and output == "skip=false\n"
    assert stdout == "forecast_slot_guard: 수집 — force=true — 같은 6시간 칸에 이미 수집됐어도 다시 수집한다\n"
    assert "force=true" in summary


def test_main_prints_warnings_as_annotations_and_still_exits_zero(tmp_path, capsys):
    # Act
    code, stdout, output, _ = run_main(["--force", "false"], tmp_path, capsys, fetch=failing(g.FetchError("HTTP 503")))

    # Assert — 판정을 못 해도 실패하지 않고 수집한다
    assert code == 0 and output == "skip=false\n"
    assert stdout.splitlines() == [
        "::warning::forecast_slot_guard: forecast.json 을 읽지 못했다 (HTTP 503)",
        "forecast_slot_guard: 수집 — 예보 발행 시각을 알 수 없어 판정 없이 수집한다",
    ]


def test_main_collects_again_and_says_why_when_the_slot_has_no_archive(tmp_path, capsys):
    # Act
    code, stdout, output, summary = run_main(
        ["--force", "false"], tmp_path, capsys, fetch=serving(forecast(SAME_SLOT_ISSUE)), head=FakeHead({STAMP_H: False, STAMP_H1: False}),
    )

    # Assert
    result = "6시간 칸 2026-10-17T00:00:00+00:00 에 예보는 있지만 archive 가 없다 — 앞 run 이 archive 전에 실패했을 수 있어 다시 수집한다 — 예보 발행 2026-10-17T00:55:12+00:00"
    assert code == 0
    assert output == "skip=false\n"
    assert stdout == f"forecast_slot_guard: 수집 — {result}\n"
    assert summary == f"### forecast 수집 가드\n\n- 판정: **수집** — {result}\n\n"


def test_main_warns_once_and_collects_when_the_archive_cannot_be_checked(tmp_path, capsys):
    # Act
    code, stdout, output, summary = run_main(
        ["--force", "false"], tmp_path, capsys,
        fetch=serving(forecast(SAME_SLOT_ISSUE)), head=FakeHead({STAMP_H: g.FetchError("HTTP 503"), STAMP_H1: g.FetchError("HTTP 503")}),
    )

    # Assert — 모르면 실패하지 않고 수집한다
    result = "6시간 칸 2026-10-17T00:00:00+00:00 에 예보는 있지만 archive 가 있는지 확인하지 못해 다시 수집한다 — 예보 발행 2026-10-17T00:55:12+00:00"
    assert code == 0 and output == "skip=false\n"
    assert stdout.splitlines() == [
        "::warning::forecast_slot_guard: forecast archive 가 있는지 확인하지 못했다 (HTTP 503)",
        f"forecast_slot_guard: 수집 — {result}",
    ]
    assert summary == f"### forecast 수집 가드\n\n- 판정: **수집** — {result}\n\n"


@pytest.mark.parametrize(
    "error",
    [urllib.error.HTTPError(ARCHIVE, 503, f"x\n::error::{MARKER}", {}, None), *NETWORK_ERRORS],
    ids=["http-error", *NETWORK_ERROR_IDS],
)
def test_main_never_prints_remote_text_from_a_failed_archive_check(tmp_path, capsys, monkeypatch, error):
    # Arrange — 실제 head_exists 를 쓰고 그 아래 응답만 바꿔치운다: 사유 문구·예외 메시지는 원격이 준 글자다
    stub_opener(monkeypatch, error)

    # Act
    code, stdout, output, summary = run_main(["--force", "false"], tmp_path, capsys, fetch=serving(forecast(SAME_SLOT_ISSUE)), head=g.head_exists)

    # Assert — 경고 한 줄 + 결과 한 줄뿐이다
    lines = stdout.splitlines()
    assert code == 0 and output == "skip=false\n" and len(lines) == 2
    assert lines[0].startswith("::warning::forecast_slot_guard: forecast archive 가 있는지 확인하지 못했다 (")
    assert MARKER not in stdout and MARKER not in summary


def test_main_asks_the_real_head_function_by_default(tmp_path, monkeypatch):
    # Arrange
    head = FakeHead({STAMP_H: True})
    monkeypatch.setattr(g, "head_exists", head)
    out_file = tmp_path / "output.txt"

    # Act — head 를 주지 않으면 모듈의 head_exists 를 쓴다
    code = g.main(["--force", "false"], fetch=serving(forecast(SAME_SLOT_ISSUE)), env={"GITHUB_OUTPUT": str(out_file)}, now=lambda: NOW)

    # Assert
    assert code == 0 and out_file.read_text(encoding="utf-8") == "skip=true\n" and head.stamps == [STAMP_H]


def test_main_works_without_the_github_files(tmp_path, capsys):
    # Act — 로컬 실행: 환경변수가 없으면 출력 파일을 건드리지 않는다
    code, stdout, output, summary = run_main(["--force", "false"], tmp_path, capsys, fetch=serving(forecast("2026-10-17T00:55:12Z")), head=FakeHead({"20261017T00": True}), with_files=False)

    # Assert
    assert code == 0 and (output, summary) == (None, None)
    assert stdout.startswith("forecast_slot_guard: 건너뜀")


@pytest.mark.parametrize("argv", [[], ["--force"], ["--force", "maybe"], ["--force", "True"], ["--force", ""], ["--force", "true", "extra"], ["bogus"]])
def test_bad_arguments_exit_with_a_usage_error(argv):
    # Act / Assert
    with pytest.raises(SystemExit) as caught:
        g.main(argv, fetch=untouched, env={}, now=lambda: NOW, head=untouched_head)
    assert caught.value.code == 2


def test_main_does_not_swallow_unexpected_errors():
    # Act / Assert
    with pytest.raises(RuntimeError):
        g.main(["--force", "false"], fetch=failing(RuntimeError("bug")), env={}, now=lambda: NOW, head=untouched_head)


def test_main_uses_the_real_utc_clock_by_default(tmp_path, capsys):
    # Arrange — 지금보다 5시간 뒤의 issue: 시험을 도는 시각·칸 경계와 무관하게 "미래" 다. 시계가 tz 없는 값이면 비교에서 TypeError 가 난다
    ahead = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
    out_file = tmp_path / "output.txt"

    # Act
    code = g.main(["--force", "false"], fetch=serving(forecast(ahead)), env={"GITHUB_OUTPUT": str(out_file)}, head=untouched_head)

    # Assert
    stdout = capsys.readouterr().out
    assert code == 0 and out_file.read_text(encoding="utf-8") == "skip=false\n"
    assert "보다 뒤다" in stdout and "뒤라 믿지 않고" in stdout


@pytest.mark.parametrize(
    "generated_at",
    [
        f"{MARKER}\n::error::x",
        f"%0A::set-env name=A::b{MARKER}",
        f"2026-10-17T00:55:12Z ::error::{MARKER}",
        f"2026-10-17T00:55:12Z\r::error::{MARKER}",
        f"2026-10-17T00:55:12Z{MARKER}",
        f"2026-10-17T00:55:12{MARKER}+00:00",
    ],
    ids=["newline", "escaped-newline", "line-separator", "carriage-return", "trailing-text", "inner-text"],
)
def test_main_never_lets_a_remote_timestamp_start_a_log_line_of_its_own(tmp_path, capsys, generated_at):
    # Act — forecast.json 은 원격이 준 값이다. 줄바꿈 문자나 `::명령::` 모양이 섞여도 러너가 워크플로 명령으로 읽을 줄이 생기면 안 된다
    code, stdout, output, summary = run_main(["--force", "false"], tmp_path, capsys, fetch=serving(forecast(generated_at)))

    # Assert — 경고 한 줄 + 결과 한 줄뿐이다(`splitlines` 는 `\r`·U+2028 도 줄 끝으로 본다). 원격이 준 글자는 어디에도 없다
    lines = stdout.splitlines()
    assert code == 0 and output == "skip=false\n" and len(lines) == 2
    assert lines[0].startswith("::warning::forecast_slot_guard: ") and lines[1].startswith("forecast_slot_guard: ")
    assert MARKER not in stdout and MARKER not in summary
    assert summary == "### forecast 수집 가드\n\n- 판정: **수집** — 예보 발행 시각을 알 수 없어 판정 없이 수집한다\n\n"  # 줄이 늘지 않았다


def test_main_prints_nothing_from_the_other_fields_of_a_valid_forecast(tmp_path, capsys):
    # Arrange — 읽을 수 있는 issue 가 있어도 나머지 필드(도시 이름 등)는 출력에 실리지 않는다
    body = doc(generated_at="2026-10-17T00:55:12Z", cities=[{"name": f"x\n::error::{MARKER}"}], note=f"\n::error::{MARKER}")

    # Act
    _, stdout, _, summary = run_main(["--force", "false"], tmp_path, capsys, fetch=serving(body), head=FakeHead({"20261017T00": True}))

    # Assert
    assert len(stdout.splitlines()) == 1 and MARKER not in stdout and MARKER not in summary


def test_the_script_runs_as_a_program_and_force_true_needs_no_network(tmp_path):
    # Arrange
    out_file = tmp_path / "output.txt"

    # Act — 격리 모드(-I): 이 디렉터리의 다른 파일을 import 하지 않아도 도는지(표준 라이브러리뿐인지)도 본다
    done = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--force", "true"],
        env={"GITHUB_OUTPUT": str(out_file)}, capture_output=True, text=True, timeout=60, check=False,
    )

    # Assert
    assert done.returncode == 0 and done.stderr == ""
    assert done.stdout == "forecast_slot_guard: 수집 — force=true — 같은 6시간 칸에 이미 수집됐어도 다시 수집한다\n"
    assert out_file.read_text(encoding="utf-8") == "skip=false\n"
