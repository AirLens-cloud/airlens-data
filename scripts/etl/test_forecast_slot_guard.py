"""forecast_slot_guard 시험 — 판정은 순수 함수, 조회는 가짜 함수·가짜 응답이라 네트워크가 0 이다 (AAA)."""
from __future__ import annotations

import http.client
import json
import re
import subprocess
import sys
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
    decision, warnings = g.check(False, serving(forecast((NOW + ahead).isoformat())), lambda: NOW)

    # Assert
    assert decision.skip is False and "뒤라" in decision.reason
    assert len(warnings) == 1 and "보다 뒤다" in warnings[0]


# ── check: 조회 + 판정 ───────────────────────────────────────────────────────
def test_check_skips_when_the_fetched_issue_is_in_the_current_slot():
    # Act
    decision, warnings = g.check(False, serving(forecast("2026-10-17T00:55:12Z")), lambda: NOW)

    # Assert
    assert (decision.skip, warnings) == (True, [])


def test_check_collects_for_an_issue_from_the_previous_slot_without_a_warning():
    # Act
    decision, warnings = g.check(False, serving(forecast("2026-10-16T21:10:03Z")), lambda: NOW)

    # Assert
    assert (decision.skip, warnings) == (False, [])


def test_check_reads_the_clock_after_the_fetch():
    # Arrange — 조회가 칸 경계를 넘겨 끝난다: 시작할 때 읽은 시계였다면 issue(05:59:58)가 같은 칸이라 건너뛴다
    clock = {"now": at("2026-10-17T05:59:59+00:00")}

    def slow_fetch() -> bytes:
        clock["now"] = at("2026-10-17T06:00:03+00:00")
        return forecast("2026-10-17T05:59:58Z")

    # Act
    decision, warnings = g.check(False, slow_fetch, lambda: clock["now"])

    # Assert
    assert (decision.skip, warnings) == (False, [])


def test_force_does_not_touch_the_network():
    # Act
    decision, warnings = g.check(True, untouched, lambda: NOW)

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
    decision, warnings = g.check(False, fetch, lambda: NOW)

    # Assert — 모르면 수집한다
    assert decision.skip is False
    assert len(warnings) == 1 and expected in warnings[0]


def test_unexpected_errors_from_the_fetch_are_not_swallowed():
    # Act / Assert — FetchError 만 "조회 실패" 다. 그 밖의 예외는 버그라 드러나야 한다
    with pytest.raises(RuntimeError):
        g.check(False, failing(RuntimeError("bug")), lambda: NOW)


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


# ── 워크플로와의 약속 ────────────────────────────────────────────────────────
def test_the_guard_reads_the_file_the_collector_publishes():
    # Arrange
    text = WORKFLOW.read_text(encoding="utf-8")
    base = re.search(r'HF_LIVE_BASE = "([^"]+)"', text)
    assert base, "수집기의 HF_LIVE_BASE 를 못 찾았다 — 추출이 깨졌다(vacuous pass 방지)"

    # Act / Assert — 읽는 곳과 쓰는 곳이 어긋나면 가드가 엉뚱한 파일을 보고 판정한다
    assert g.FORECAST_URL == f"{base.group(1)}/aq-data/forecast.json"
    assert "--dest aq-data/forecast.json" in text


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


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError(f"{MARKER}\n::error::x"),
        TimeoutError(MARKER),
        ConnectionResetError(MARKER),
        http.client.IncompleteRead(b"abc"),
        http.client.BadStatusLine(f"{MARKER}\n::error::x"),
    ],
    ids=["url-error", "timeout", "reset", "incomplete-read", "bad-status-line"],
)
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


# ── main: GITHUB_OUTPUT · 요약 · 종료 코드 ───────────────────────────────────
def run_main(argv, tmp_path, capsys, *, fetch=untouched, now=lambda: NOW, with_files=True):
    out_file, summary_file = tmp_path / "output.txt", tmp_path / "summary.md"
    env = {"GITHUB_OUTPUT": str(out_file), "GITHUB_STEP_SUMMARY": str(summary_file)} if with_files else {}
    code = g.main(argv, fetch=fetch, env=env, now=now)
    stdout = capsys.readouterr().out

    def read(path: Path) -> str | None:
        return path.read_text(encoding="utf-8") if path.exists() else None

    return code, stdout, read(out_file), read(summary_file)


def test_main_writes_skip_true_and_a_summary_when_the_slot_is_already_collected(tmp_path, capsys):
    # Act
    code, stdout, output, summary = run_main(["--force", "false"], tmp_path, capsys, fetch=serving(forecast("2026-10-17T00:55:12Z")))

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


def test_main_works_without_the_github_files(tmp_path, capsys):
    # Act — 로컬 실행: 환경변수가 없으면 출력 파일을 건드리지 않는다
    code, stdout, output, summary = run_main(["--force", "false"], tmp_path, capsys, fetch=serving(forecast("2026-10-17T00:55:12Z")), with_files=False)

    # Assert
    assert code == 0 and (output, summary) == (None, None)
    assert stdout.startswith("forecast_slot_guard: 건너뜀")


@pytest.mark.parametrize("argv", [[], ["--force"], ["--force", "maybe"], ["--force", "True"], ["--force", ""], ["--force", "true", "extra"], ["bogus"]])
def test_bad_arguments_exit_with_a_usage_error(argv):
    # Act / Assert
    with pytest.raises(SystemExit) as caught:
        g.main(argv, fetch=untouched, env={}, now=lambda: NOW)
    assert caught.value.code == 2


def test_main_does_not_swallow_unexpected_errors():
    # Act / Assert
    with pytest.raises(RuntimeError):
        g.main(["--force", "false"], fetch=failing(RuntimeError("bug")), env={}, now=lambda: NOW)


def test_main_uses_the_real_utc_clock_by_default(tmp_path, capsys):
    # Arrange — 지금보다 5시간 뒤의 issue: 시험을 도는 시각·칸 경계와 무관하게 "미래" 다. 시계가 tz 없는 값이면 비교에서 TypeError 가 난다
    ahead = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
    out_file = tmp_path / "output.txt"

    # Act
    code = g.main(["--force", "false"], fetch=serving(forecast(ahead)), env={"GITHUB_OUTPUT": str(out_file)})

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
    _, stdout, _, summary = run_main(["--force", "false"], tmp_path, capsys, fetch=serving(body))

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
