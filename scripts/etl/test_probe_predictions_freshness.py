"""probe_predictions_freshness.py 단위 테스트 (AAA). 네트워크 없음 — 문서/슬롯 fixture 와 가짜 urlopen.

핵심 회귀:
1. 발행 워크플로가 스케줄에서 밀리거나 꺼지면 run 이 **생성조차 되지 않아** Actions 탭이 조용하다.
   판정은 run 이 아니라 *발행물*을 봐야 하고, 워크플로가 켜져 있는지도 따로 봐야 한다.
2. history 슬롯 시각은 관측 슬롯에서 오므로 고정 격자에 정렬되지 않는다
   (실측 08,14,18,23 / 05,11,16,21,23). 격자 기반 결손 판정은 오탐이라 **간격**으로 센다.
3. 예측 밴드는 웹과 같은 12시간 기준으로 본다(경계값은 정상 — 웹의 `age > STALE`).
4. 이 레포에 아직 없는 워크플로(404)는 "미배포" 이지 결손이 아니다. 조회 실패는 프로브 고장(probe_error)이다.
"""
from __future__ import annotations

import io
import json
import urllib.error
from datetime import datetime, timedelta, timezone

import pytest

import probe_predictions_freshness as m

UTC = timezone.utc
NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
MAX_AGE = timedelta(hours=4)
CADENCE = timedelta(hours=3)
REPO = "example-org/example-repo"
TOKEN = "unit-test-token-value"


def _latest(generated_at: str | None = "2026-09-03T11:45:44Z") -> dict:
    doc = {"model_version": "v2.0", "count": 1200}
    if generated_at is not None:
        doc["generated_at"] = generated_at
    return doc


def _slots(*hours_ago: float) -> list[datetime]:
    return [NOW - timedelta(hours=h) for h in hours_ago]


def _band(hours_ago: float | None = 3.0) -> m.BandFetch:
    doc: dict = {"schema_version": "w3-band-v1"}
    if hours_ago is not None:
        doc["generated_at"] = (NOW - timedelta(hours=hours_ago)).isoformat()  # 정확한 나이 — 경계값 시험용
    return m.BandFetch(doc, False)


def _classify(**kwargs) -> dict:
    return m.classify(_latest(), _slots(1, 7, 13), NOW, MAX_AGE, CADENCE, **kwargs)


# ── grid_latest / history (이전 전의 판정 그대로) ────────────────────────────
def test_fresh_publication_has_no_problems():
    # Arrange — latest 16분 전, 최신 슬롯 1시간 전
    # Act
    verdict = m.classify(_latest(), _slots(1, 7, 13), NOW, MAX_AGE, CADENCE)

    # Assert
    assert verdict["problems"] == []


def test_stale_latest_is_flagged_when_older_than_max_age():
    # Arrange — 발행이 멈춰 발행물이 굳은 상태
    latest = _latest("2026-09-03T02:45:00Z")  # 9.25h 전

    # Act
    verdict = m.classify(latest, _slots(1), NOW, MAX_AGE, CADENCE)

    # Assert
    assert any(p.startswith("stale_latest") for p in verdict["problems"])
    assert verdict["metrics"]["latest_age_h"] == 9.2


def test_latest_exactly_at_the_limit_is_fresh_and_just_over_is_stale():
    # Arrange — 허용 나이 경계(4h)는 정상, 넘으면 stale
    at_limit = _latest((NOW - timedelta(hours=4)).isoformat())
    over = _latest((NOW - timedelta(hours=4, minutes=1)).isoformat())

    # Act
    ok = m.classify(at_limit, _slots(1), NOW, MAX_AGE, CADENCE)
    stale = m.classify(over, _slots(1), NOW, MAX_AGE, CADENCE)

    # Assert
    assert ok["problems"] == []
    assert [p.split(" ")[0] for p in stale["problems"]] == ["stale_latest"]


def test_history_gap_is_flagged_even_when_latest_is_fresh():
    # Arrange — 이 프로브의 두 번째 존재 이유: latest 는 갱신되는데 history 적재만 죽은 경로.
    # 최신 슬롯 7h 전 = 주기 3h 의 2배 초과.
    # Act
    verdict = m.classify(_latest(), _slots(7, 13), NOW, MAX_AGE, CADENCE)

    # Assert
    assert not any(p.startswith("stale_latest") for p in verdict["problems"])
    assert any(p.startswith("history_gap") for p in verdict["problems"])
    assert verdict["metrics"]["consecutive_missed"] == 1


def test_one_missed_slot_is_tolerated():
    # Arrange — 4h 공백 = 슬롯 1개 결손. 상시 발생하므로 경보 대상이 아니다
    # (2*cadence 초과부터가 "연속 결손").
    # Act
    verdict = m.classify(_latest(), _slots(4, 10), NOW, MAX_AGE, CADENCE)

    # Assert
    assert verdict["problems"] == []
    assert verdict["metrics"]["consecutive_missed"] == 0


def test_gap_of_exactly_two_cadences_is_tolerated_and_beyond_it_is_flagged():
    # Arrange — "연속 결손" 은 발행 주기의 2배를 *넘을 때* 부터다 (6h 는 정상, 6.1h 는 이상)
    # Act
    at_limit = m.classify(_latest(), _slots(6), NOW, MAX_AGE, CADENCE)
    over = m.classify(_latest(), _slots(6.1), NOW, MAX_AGE, CADENCE)

    # Assert
    assert at_limit["problems"] == []
    assert [p.split(" ")[0] for p in over["problems"]] == ["history_gap"]


def test_empty_history_is_flagged():
    # Arrange / Act
    verdict = m.classify(_latest(), [], NOW, MAX_AGE, CADENCE)

    # Assert
    assert any(p.startswith("history_empty") for p in verdict["problems"])


def test_fetch_error_does_not_masquerade_as_empty_history():
    # Arrange — HF 를 못 읽은 것과 "슬롯이 없다" 는 다른 사실이다.
    # 조용히 통과시키면 프로브 자체의 고장이 정상으로 위장된다.
    errors = ["probe_error — `predictions/history/` 를 나열하지 못했다 (URLError: timeout)"]

    # Act
    verdict = m.classify(None, [], NOW, MAX_AGE, CADENCE, fetch_errors=errors)

    # Assert
    assert any(p.startswith("probe_error") for p in verdict["problems"])
    assert not any(p.startswith("history_empty") for p in verdict["problems"])


def test_missing_generated_at_is_flagged_as_malformed():
    # Arrange / Act
    verdict = m.classify(_latest(generated_at=None), _slots(1), NOW, MAX_AGE, CADENCE)

    # Assert
    assert any(p.startswith("malformed") for p in verdict["problems"])


def test_coverage_counts_only_the_last_24h():
    # Arrange — 30일 보존이라 오래된 슬롯이 함께 나열된다. 커버리지는 최근 24h 만 센다.
    slots = _slots(1, 7, 13, 19, 30, 55)

    # Act
    verdict = m.classify(_latest(), slots, NOW, MAX_AGE, CADENCE)

    # Assert
    assert verdict["metrics"]["coverage_24h"] == 4
    assert verdict["metrics"]["expected_24h"] == 8


def test_parse_slot_reads_the_published_filename_shape():
    # Arrange / Act / Assert — 실측 경로 그대로
    assert m.parse_slot("aq-data/predictions/history/grid_2026090311.json") == datetime(
        2026, 9, 3, 11, tzinfo=UTC
    )


def test_parse_slot_rejects_names_outside_the_convention():
    # Arrange / Act / Assert — backfill·임시 파일이 슬롯으로 오인되면 결손이 묻힌다
    assert m.parse_slot("aq-data/predictions/history/README.md") is None
    assert m.parse_slot("aq-data/predictions/history/grid_2026093199.json") is None


def test_classify_signature_without_new_arguments_checks_neither_band_nor_workflows():
    # Arrange / Act — 기존 호출 형태(위치 인자 5개)는 그대로 동작한다
    verdict = m.classify(_latest(), _slots(1), NOW, MAX_AGE, CADENCE)

    # Assert
    assert verdict["metrics"]["band_age_h"] is None
    assert verdict["metrics"]["workflows"] == {}


# ── 예측 밴드 (웹과 같은 12h 기준) ───────────────────────────────────────────
def test_fresh_band_has_no_problems_and_reports_its_age():
    # Act
    verdict = _classify(band=_band(3.0))

    # Assert
    assert verdict["problems"] == []
    assert verdict["metrics"]["band_age_h"] == 3.0
    assert verdict["metrics"]["band_generated_at"] == "2026-09-03T09:00:00Z"


def test_band_timestamp_in_the_published_shape_is_parsed():
    # Arrange — 실제 발행물의 형태: 마이크로초 + `+00:00`
    band = m.BandFetch({"generated_at": "2026-09-03T08:25:13.178973+00:00"}, False)

    # Act
    verdict = _classify(band=band)

    # Assert
    assert verdict["problems"] == []
    assert verdict["metrics"]["band_age_h"] == 3.6
    assert verdict["metrics"]["band_generated_at"] == "2026-09-03T08:25:13Z"


def test_band_exactly_at_the_limit_is_fresh_and_just_over_is_stale():
    # Arrange — 웹은 `ageMs > STALE` 일 때만 오래됨 배지를 단다. 경계값은 정상.
    at_limit, over = _band(12.0), _band(12.01)

    # Act / Assert
    assert _classify(band=at_limit)["problems"] == []
    problems = _classify(band=over)["problems"]
    assert len(problems) == 1 and problems[0].startswith("stale_band")
    assert "허용 12h" in problems[0]


def test_band_limit_is_configurable_for_the_probe_but_defaults_to_the_web_value():
    # Arrange / Act / Assert
    assert m.BAND_MAX_AGE == timedelta(hours=12)
    assert _classify(band=_band(5.0), band_max_age=timedelta(hours=4))["problems"][0].startswith("stale_band")


def test_band_without_generated_at_is_malformed_and_names_the_band_file():
    # Arrange / Act
    problems = _classify(band=_band(None))["problems"]

    # Assert — grid_latest 의 malformed 와 구분된다
    assert len(problems) == 1
    assert problems[0].startswith("malformed") and "forecast-band" in problems[0]


def test_missing_band_file_is_a_publication_gap_not_a_probe_error():
    # Arrange / Act
    problems = _classify(band=m.BandFetch(None, True))["problems"]

    # Assert — 404: 발행물의 결손이다. probe_error 가 아니라서 프로브 스텝을 red 로 만들지 않는다.
    assert len(problems) == 1 and problems[0].startswith("band_missing")


def test_band_fetch_failure_is_only_a_probe_error_never_a_missing_band():
    # Arrange
    errors = ["probe_error — `forecast-band/latest.json` 을 읽지 못했다 (URLError: timeout)"]

    # Act
    problems = _classify(band=m.BandFetch(None, False), fetch_errors=errors)["problems"]

    # Assert
    assert problems == errors


# ── 워크플로 상태 ────────────────────────────────────────────────────────────
def test_active_workflows_have_no_problems():
    # Act
    verdict = _classify(workflow_states={"ml-predict.yml": "active", "band-forecast.yml": "active"})

    # Assert
    assert verdict["problems"] == []


@pytest.mark.parametrize("state", ["disabled_inactivity", "disabled_manually", "disabled_fork", "deleted"])
def test_workflow_that_is_not_active_is_flagged_with_its_state(state):
    # Arrange / Act
    problems = _classify(workflow_states={"ml-predict.yml": state})["problems"]

    # Assert
    assert len(problems) == 1
    assert problems[0].startswith("workflow_inactive")
    assert "`ml-predict.yml`" in problems[0] and state in problems[0]


def test_workflow_not_deployed_yet_is_not_a_problem_but_is_shown():
    # Arrange — 이 레포에 아직 없는 워크플로(404 → None)
    verdict = _classify(workflow_states={"ml-predict.yml": None, "band-forecast.yml": "active"})

    # Act
    report = m.render_report(verdict, NOW)

    # Assert
    assert verdict["problems"] == []
    assert "| 워크플로 `ml-predict.yml` | 미배포 |" in report
    assert "| 워크플로 `band-forecast.yml` | active |" in report


def test_each_inactive_workflow_is_reported_separately():
    # Arrange / Act
    problems = _classify(workflow_states={"a.yml": "disabled_manually", "b.yml": "active", "c.yml": "deleted"})["problems"]

    # Assert
    assert [p.split("`")[1] for p in problems] == ["a.yml", "c.yml"]


# ── 감시 대상 목록 ───────────────────────────────────────────────────────────
def test_parse_workflow_list_reads_default_trimmed_and_deduplicated():
    # Act
    names, bad = m.parse_workflow_list(" ml-predict.yml , band-forecast.yml,,ml-predict.yml ")

    # Assert
    assert names == ["ml-predict.yml", "band-forecast.yml"]
    assert bad == []
    assert m.parse_workflow_list(m.DEFAULT_WATCHED)[0] == ["ml-predict.yml", "band-forecast.yml"]


@pytest.mark.parametrize("item", ["../x.yml", "a/b.yml", "x.txt", "no-extension", ".hidden.yml", "a b.yml", "x.yml?y=1"])
def test_parse_workflow_list_rejects_names_that_could_alter_the_request_path(item):
    # Act
    names, bad = m.parse_workflow_list(f"ok.yml,{item}")

    # Assert
    assert names == ["ok.yml"]
    assert bad == [item]


def test_parse_workflow_list_empty_means_no_workflow_checks():
    assert m.parse_workflow_list("") == ([], [])
    assert m.parse_workflow_list(" , ") == ([], [])


# ── 조회 함수 (가짜 urlopen) ─────────────────────────────────────────────────
class FakeResp:
    def __init__(self, body: bytes, link: str = "") -> None:
        self._body = body
        self.headers = {"Link": link} if link else {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResp":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid/secret", code, "msg", {}, io.BytesIO(b"secret body"))


@pytest.fixture
def fake_urlopen(monkeypatch):
    """결과를 차례로 돌려주고 요청 객체를 기록하는 urlopen."""
    state: dict = {"outcomes": [], "requests": []}

    def fake(req, timeout=None):
        state["requests"].append(req)
        outcome = state["outcomes"].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(m.urllib.request, "urlopen", fake)
    return state


def test_workflow_state_reads_state_field_from_the_repo_workflow_endpoint(fake_urlopen):
    # Arrange
    fake_urlopen["outcomes"] = [FakeResp(json.dumps({"id": 1, "state": "active"}).encode())]

    # Act
    state, error = m.fetch_workflow_state("ml-predict.yml", REPO, TOKEN, 5)

    # Assert
    req = fake_urlopen["requests"][0]
    assert (state, error) == ("active", None)
    assert req.full_url == f"https://api.github.com/repos/{REPO}/actions/workflows/ml-predict.yml"


def test_workflow_state_sends_the_token_only_as_an_unredirected_header(fake_urlopen):
    # Arrange
    fake_urlopen["outcomes"] = [FakeResp(b'{"state": "active"}'), FakeResp(b'{"state": "active"}')]

    # Act
    m.fetch_workflow_state("a.yml", REPO, TOKEN, 5)
    m.fetch_workflow_state("a.yml", REPO, None, 5)

    # Assert — 토큰이 있으면 리다이렉트를 따라가지 않는 헤더로, 없으면 익명 조회
    with_token, without_token = fake_urlopen["requests"]
    assert with_token.unredirected_hdrs.get("Authorization") == f"Bearer {TOKEN}"
    assert "Authorization" not in with_token.headers
    assert "Authorization" not in without_token.unredirected_hdrs


def test_workflow_state_404_means_not_deployed_and_is_not_an_error(fake_urlopen):
    # Arrange
    fake_urlopen["outcomes"] = [_http_error(404)]

    # Act / Assert
    assert m.fetch_workflow_state("band-forecast.yml", REPO, TOKEN, 5) == (None, None)


@pytest.mark.parametrize("code", [401, 403, 500])
def test_workflow_state_other_http_errors_are_probe_errors_without_url_or_token(code, fake_urlopen):
    # Arrange
    fake_urlopen["outcomes"] = [_http_error(code)]

    # Act
    state, error = m.fetch_workflow_state("ml-predict.yml", REPO, TOKEN, 5)

    # Assert
    assert state is None
    assert error is not None and error.startswith("probe_error")
    assert f"HTTPError: {code}" in error
    assert TOKEN not in error and "example.invalid" not in error and "secret body" not in error


def test_workflow_state_network_error_reports_class_only(fake_urlopen):
    # Arrange
    fake_urlopen["outcomes"] = [urllib.error.URLError(f"cannot reach host with {TOKEN}")]

    # Act
    _state, error = m.fetch_workflow_state("ml-predict.yml", REPO, TOKEN, 5)

    # Assert
    assert error is not None and "URLError" in error
    assert TOKEN not in error


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"id": 1}', b'{"state": 5}'])
def test_workflow_state_unreadable_body_is_a_probe_error(body, fake_urlopen):
    # Arrange
    fake_urlopen["outcomes"] = [FakeResp(body)]

    # Act
    state, error = m.fetch_workflow_state("ml-predict.yml", REPO, TOKEN, 5)

    # Assert
    assert state is None
    assert error is not None and error.startswith("probe_error")


def test_workflow_states_collects_states_and_errors_separately(fake_urlopen):
    # Arrange
    fake_urlopen["outcomes"] = [FakeResp(b'{"state": "active"}'), _http_error(404), _http_error(500)]

    # Act
    states, errors = m.fetch_workflow_states(["a.yml", "b.yml", "c.yml"], REPO, TOKEN, 5)

    # Assert — 실패한 c.yml 은 사전에 없다(판정에 "미배포" 로 섞이지 않는다)
    assert states == {"a.yml": "active", "b.yml": None}
    assert len(errors) == 1 and "c.yml" in errors[0]


def test_fetch_band_returns_document_on_success(fake_urlopen):
    # Arrange
    fake_urlopen["outcomes"] = [FakeResp(b'{"generated_at": "2026-09-03T09:00:00Z"}')]

    # Act
    band, error = m.fetch_band("https://example.invalid/band.json", 5)

    # Assert
    assert error is None
    assert band == m.BandFetch({"generated_at": "2026-09-03T09:00:00Z"}, False)


def test_fetch_band_404_is_missing_not_an_error(fake_urlopen):
    # Arrange
    fake_urlopen["outcomes"] = [_http_error(404)]

    # Act / Assert
    assert m.fetch_band("https://example.invalid/band.json", 5) == (m.BandFetch(None, True), None)


@pytest.mark.parametrize("outcome", [_http_error(500), urllib.error.URLError("down"), TimeoutError("slow")])
def test_fetch_band_failures_are_probe_errors(outcome, fake_urlopen):
    # Arrange
    fake_urlopen["outcomes"] = [outcome]

    # Act
    band, error = m.fetch_band("https://example.invalid/band.json", 5)

    # Assert
    assert band == m.BandFetch(None, False)
    assert error is not None and error.startswith("probe_error") and "forecast-band" in error


@pytest.mark.parametrize("body", [b"[1, 2]", b"null", b"<html>"])
def test_fetch_band_and_latest_reject_non_object_documents(body, fake_urlopen):
    # Arrange
    fake_urlopen["outcomes"] = [FakeResp(body), FakeResp(body)]

    # Act
    band, band_error = m.fetch_band("https://example.invalid/band.json", 5)
    latest, latest_error = m.fetch_latest("https://example.invalid/latest.json", 5)

    # Assert — 모양이 틀린 문서가 classify 에서 AttributeError 로 터지지 않고 probe_error 가 된다
    assert band.doc is None and band_error and band_error.startswith("probe_error")
    assert latest is None and latest_error and latest_error.startswith("probe_error")


# ── main(): 종료 코드 · 출력 ─────────────────────────────────────────────────
@pytest.fixture
def main_env(tmp_path, monkeypatch):
    """네트워크 없이 main() 을 돌리는 환경. 반환값은 (출력 파일 경로 dict, fetch 호출 기록)."""
    for key in ("PREDICTIONS_MAX_AGE_HOURS", "PREDICTIONS_CADENCE_HOURS", "BAND_MAX_AGE_HOURS", "WATCHED_WORKFLOWS",
                "FAIL_ON_STALE", "GITHUB_REPOSITORY", "GH_TOKEN", "PREDICTIONS_LATEST_URL", "PREDICTIONS_HISTORY_URL",
                "BAND_LATEST_URL"):
        monkeypatch.delenv(key, raising=False)
    paths = {"output": tmp_path / "out", "report": tmp_path / "report.md", "summary": tmp_path / "summary.md"}
    monkeypatch.setenv("GITHUB_OUTPUT", str(paths["output"]))
    monkeypatch.setenv("PREDICTIONS_REPORT_PATH", str(paths["report"]))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(paths["summary"]))
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    calls: dict = {"workflow_args": None}

    class FixedNow(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    monkeypatch.setattr(m, "datetime", FixedNow)
    monkeypatch.setattr(m, "fetch_latest", lambda url, timeout: (_latest(), None))
    monkeypatch.setattr(m, "fetch_slots", lambda url, timeout: (_slots(1, 7), None))
    monkeypatch.setattr(m, "fetch_band", lambda url, timeout: (_band(3.0), None))

    def fake_states(names, repo, token, timeout):
        calls["workflow_args"] = (list(names), repo, token)
        return {n: "active" for n in names}, []

    monkeypatch.setattr(m, "fetch_workflow_states", fake_states)
    return paths, calls


def test_main_healthy_returns_zero_and_writes_all_outputs(main_env, capsys):
    # Arrange
    paths, calls = main_env

    # Act
    code = m.main()

    # Assert
    assert code == 0
    assert "healthy=true" in paths["output"].read_text(encoding="utf-8")
    assert "✅ 발행 정상" in paths["report"].read_text(encoding="utf-8")
    assert "✅ 발행 정상" in paths["summary"].read_text(encoding="utf-8")
    assert calls["workflow_args"] == (["ml-predict.yml", "band-forecast.yml"], REPO, None)
    capsys.readouterr()


def test_main_staleness_alone_exits_zero_because_the_issue_is_the_alert_channel(main_env, monkeypatch, capsys):
    # Arrange
    paths, _ = main_env
    monkeypatch.setattr(m, "fetch_band", lambda url, timeout: (_band(20.0), None))

    # Act
    code = m.main()

    # Assert
    assert code == 0
    assert "healthy=false" in paths["output"].read_text(encoding="utf-8")
    assert "stale_band" in paths["report"].read_text(encoding="utf-8")
    capsys.readouterr()


def test_main_exits_one_when_the_probe_itself_could_not_read(main_env, monkeypatch, capsys):
    # Arrange
    monkeypatch.setattr(m, "fetch_slots", lambda url, timeout: ([], "probe_error — `predictions/history/` 를 나열하지 못했다 (URLError: x)"))

    # Act / Assert
    assert m.main() == 1
    capsys.readouterr()


def test_main_exits_one_for_staleness_only_when_fail_on_stale_is_true(main_env, monkeypatch, capsys):
    # Arrange
    monkeypatch.setattr(m, "fetch_band", lambda url, timeout: (_band(20.0), None))
    monkeypatch.setenv("FAIL_ON_STALE", "TRUE")

    # Act / Assert
    assert m.main() == 1
    capsys.readouterr()


def test_main_inactive_workflow_is_an_alert_not_a_probe_failure(main_env, monkeypatch, capsys):
    # Arrange
    monkeypatch.setattr(m, "fetch_workflow_states", lambda names, repo, token, timeout: ({"ml-predict.yml": "disabled_manually"}, []))

    # Act
    code = m.main()

    # Assert
    assert code == 0
    assert "workflow_inactive" in main_env[0]["report"].read_text(encoding="utf-8")
    capsys.readouterr()


def test_main_workflow_lookup_failure_exits_one(main_env, monkeypatch, capsys):
    # Arrange
    monkeypatch.setattr(m, "fetch_workflow_states", lambda names, repo, token, timeout: ({}, ["probe_error — 워크플로 `ml-predict.yml` 의 상태를 읽지 못했다 (HTTPError: 403)"]))

    # Act / Assert
    assert m.main() == 1
    capsys.readouterr()


def test_main_watched_workflows_env_replaces_the_default_list(main_env, monkeypatch, capsys):
    # Arrange — 롤백 등으로 한 워크플로를 일부러 끌 때 목록에서 빼는 경로
    monkeypatch.setenv("WATCHED_WORKFLOWS", "band-forecast.yml")
    _, calls = main_env

    # Act
    code = m.main()

    # Assert
    assert code == 0
    assert calls["workflow_args"][0] == ["band-forecast.yml"]
    capsys.readouterr()


def test_main_empty_watched_workflows_skips_the_state_check_entirely(main_env, monkeypatch, capsys):
    # Arrange
    monkeypatch.setenv("WATCHED_WORKFLOWS", "")
    _, calls = main_env

    # Act
    code = m.main()

    # Assert
    assert code == 0
    assert calls["workflow_args"] is None
    capsys.readouterr()


def test_main_without_repository_env_skips_the_state_check(main_env, monkeypatch, capsys):
    # Arrange — 로컬 실행
    monkeypatch.delenv("GITHUB_REPOSITORY")
    _, calls = main_env

    # Act
    code = m.main()

    # Assert
    assert code == 0
    assert calls["workflow_args"] is None
    capsys.readouterr()


def test_main_passes_the_token_from_gh_token(main_env, monkeypatch, capsys):
    # Arrange
    monkeypatch.setenv("GH_TOKEN", TOKEN)
    _, calls = main_env

    # Act
    m.main()

    # Assert
    assert calls["workflow_args"][2] == TOKEN
    capsys.readouterr()


def test_main_invalid_watched_name_is_reported_as_probe_error_without_echoing_it(main_env, monkeypatch, capsys):
    # Arrange
    monkeypatch.setenv("WATCHED_WORKFLOWS", "ml-predict.yml,../secret-name.yml")

    # Act
    code = m.main()

    # Assert
    out = capsys.readouterr().out
    assert code == 1
    assert "WATCHED_WORKFLOWS" in out
    assert "secret-name" not in out


def test_main_malformed_repository_env_skips_the_state_check(main_env, monkeypatch, capsys):
    # Arrange — 요청 경로에 들어가는 값이라 모양이 틀리면 쓰지 않는다
    monkeypatch.setenv("GITHUB_REPOSITORY", "not a repo/../x")
    _, calls = main_env

    # Act
    m.main()

    # Assert
    assert calls["workflow_args"] is None
    capsys.readouterr()


# ── 리포트 ───────────────────────────────────────────────────────────────────
def test_report_points_to_the_actions_tab_when_unhealthy():
    # Arrange — 이슈를 여는 사람이 "스케줄에서 밀리면 run 이 안 생긴다" 를 모르면
    # Actions 탭이 조용한 것을 정상으로 읽는다. 본문이 그 순서를 들고 있어야 한다.
    verdict = m.classify(_latest("2026-09-03T02:00:00Z"), _slots(10), NOW, MAX_AGE, CADENCE)

    # Act
    report = m.render_report(verdict, NOW)

    # Assert
    assert "Actions 탭" in report
    assert "keepalive" in report
    assert "자동으로 닫힌다" in report


def test_report_is_green_when_healthy():
    # Arrange / Act
    report = m.render_report(m.classify(_latest(), _slots(1), NOW, MAX_AGE, CADENCE), NOW)

    # Assert
    assert "✅ 발행 정상" in report
    assert "점검 순서" not in report


def test_report_shows_band_rows_and_dashes_when_nothing_was_measured():
    # Arrange
    with_band = m.render_report(_classify(band=_band(3.0)), NOW)
    without = m.render_report(m.classify(None, [], NOW, MAX_AGE, CADENCE, fetch_errors=["probe_error — x"]), NOW)

    # Assert
    assert "| 예측 밴드 `generated_at` | 2026-09-03T09:00:00Z |" in with_band
    assert "| 밴드 나이 | 3.0 h |" in with_band
    assert "| 예측 밴드 `generated_at` | — |" in without
    assert "| 밴드 나이 | — |" in without
