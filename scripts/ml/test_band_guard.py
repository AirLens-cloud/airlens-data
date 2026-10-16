"""band_guard 시험 — 판정은 순수 함수, 조회는 가짜 응답이라 네트워크가 0 이다."""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone

import pytest

import band_guard as bg
import compare_candidate as cc
import publish_targets as pt

ISSUE_Z = "2026-10-15T06:15:42Z"
ISSUE = datetime(2026, 10, 15, 6, 15, 42, tzinfo=timezone.utc)
OLDER = datetime(2026, 10, 15, 0, 15, 40, tzinfo=timezone.utc)
STAMP = "2026101506"
CANDIDATE_PATH = pt.band_history_dest(STAMP)
PROD_COPY_PATH = pt.band_prod_copy_dest(STAMP)


# ── 판정 (순수 함수) ─────────────────────────────────────────────────────────
def decide(mode, facts, force=False):
    return bg.decide(mode, force, facts)


def test_production_skips_only_when_the_published_issue_is_the_same_instant():
    # Act
    same = decide("production", bg.Facts(ISSUE, published_issue=ISSUE))
    older = decide("production", bg.Facts(ISSUE, published_issue=OLDER))
    newer = decide("production", bg.Facts(OLDER, published_issue=ISSUE))

    # Assert — 더 새로운 운영 latest(`>=`)도 건너뛰지 않는다: 같은 발행임을 확인한 경우만 건너뛴다
    assert (same.skip, older.skip, newer.skip) == (True, False, False)
    assert "이미" in same.reason


def test_production_runs_when_there_is_no_published_latest_yet():
    # Act
    decision = decide("production", bg.Facts(ISSUE, published_issue=None))

    # Assert
    assert decision.skip is False
    assert "운영 latest" in decision.reason


def test_production_ignores_a_candidate_file_that_exists():
    # Act — 운영 판정은 운영 latest 만 본다
    decision = decide("production", bg.Facts(ISSUE, published_issue=OLDER, candidate_exists=True))

    # Assert
    assert decision.skip is False


def test_candidate_skips_when_its_file_exists_and_ignores_the_production_issue():
    # Act
    exists = decide("candidate", bg.Facts(ISSUE, candidate_exists=True))
    absent = decide("candidate", bg.Facts(ISSUE, candidate_exists=False))
    absent_but_published = decide("candidate", bg.Facts(ISSUE, published_issue=ISSUE, candidate_exists=False))

    # Assert — 후보는 운영 latest 가 같은 발행이어도 후보 파일이 없으면 돈다
    assert (exists.skip, absent.skip, absent_but_published.skip) == (True, False, False)
    assert f"band_{STAMP}.json" in exists.reason


@pytest.mark.parametrize("mode", ["production", "candidate"])
def test_force_never_skips(mode):
    # Act
    decision = decide(mode, bg.Facts(ISSUE, published_issue=ISSUE, candidate_exists=True), force=True)

    # Assert
    assert decision.skip is False
    assert "force" in decision.reason


@pytest.mark.parametrize("mode", ["production", "candidate"])
def test_unknown_issue_means_run(mode):
    # Act
    decision = decide(mode, bg.Facts(None, published_issue=ISSUE, candidate_exists=True))

    # Assert — 모르면 돌린다
    assert decision.skip is False


@pytest.mark.parametrize("mode", ["", "prod", "Candidate", "both"])
def test_unknown_mode_is_a_programming_error(mode):
    # Act / Assert
    with pytest.raises(ValueError):
        decide(mode, bg.Facts(ISSUE))


# ── check: 사실 수집 + 판정 ──────────────────────────────────────────────────
class FakeClient(cc.HttpSource):
    """공개 저장소 대역(`read_json` 은 `HttpSource` 것 그대로). files = 경로 → bytes | 예외 (없으면 404).
    조회한 경로를 기록한다. 재시도·상태 코드·헤더는 test_compare_candidate.py 가 `HttpSource` 로 시험한다."""

    def __init__(self, files: dict[str, object] | None = None) -> None:
        super().__init__(sleep=lambda _s: None)
        self.files = files or {}
        self.paths: list[str] = []

    def read_bytes(self, rel_path: str) -> bytes | None:
        self.paths.append(rel_path)
        value = self.files.get(rel_path)
        if isinstance(value, BaseException):
            raise value
        return value  # type: ignore[return-value]


def doc(**fields: object) -> bytes:
    return json.dumps(fields).encode()


def forecast(generated_at: object = ISSUE_Z) -> bytes:
    return doc(generated_at=generated_at, cities=[])


def band_doc(issue_time: object = "2026-10-15T06:15:42+00:00", schema_version: object = "w3-band-v1") -> dict:
    """계약(`w3-band-v1`)을 지키는 가장 작은 발행물."""
    return {
        "schema_version": schema_version, "generated_at": "2026-10-15T06:40:00+00:00", "model": "timesfm-test + CQR", "issue_time": issue_time,
        "cities": [{"name": "city-00", "horizons": [{"lead_hours": 1, "valid_time": "2026-10-15T07:00:00+00:00", "p10": 1.0, "p50": 2.0, "p90": 3.0}]}],
        "uncertainty": {"method": "CQR conformal", "picp80_claim_by_horizon": {}, "provisional_horizons": []},
        "dqss": {"grade": None, "status": "unscored"},
    }


def band(issue_time: object = "2026-10-15T06:15:42+00:00", schema_version: object = "w3-band-v1") -> bytes:
    return json.dumps(band_doc(issue_time, schema_version)).encode()


def test_the_helper_payload_satisfies_the_real_band_contract():
    # Act / Assert — 아래 snapshot 시험의 입력이 계약에서 벗어나 있으면 시험이 아무것도 보증하지 못한다
    assert cc.contract_validator()(band_doc()) is None


def test_production_check_skips_when_latest_has_the_same_issue_written_with_a_different_suffix():
    # Arrange — forecast.json 은 `Z`, band latest 는 `+00:00`: 문자열 비교였다면 영원히 건너뛰지 못한다
    client = FakeClient({bg.FORECAST_PATH: forecast("2026-10-15T06:15:42Z"), pt.BAND_LATEST: band("2026-10-15T06:15:42+00:00")})

    # Act
    decision, warnings = bg.check("production", False, client)

    # Assert
    assert decision.skip is True and warnings == []
    assert client.paths == [bg.FORECAST_PATH, pt.BAND_LATEST]


def test_production_check_runs_for_a_newer_forecast_issue():
    # Arrange
    client = FakeClient({bg.FORECAST_PATH: forecast(), pt.BAND_LATEST: band("2026-10-15T00:15:40+00:00")})

    # Act
    decision, warnings = bg.check("production", False, client)

    # Assert
    assert (decision.skip, warnings) == (False, [])


def test_production_check_without_a_published_latest_runs_without_a_warning():
    # Arrange — 아직 한 번도 발행하지 않았다면 정상 상황이다
    client = FakeClient({bg.FORECAST_PATH: forecast()})

    # Act
    decision, warnings = bg.check("production", False, client)

    # Assert
    assert (decision.skip, warnings) == (False, [])


@pytest.mark.parametrize(
    ("latest", "expected"),
    [
        (b"<html>", "JSON 아님"),
        (doc(schema_version="w3-band-v1"), "issue_time 를 읽을 수 없다"),
        (doc(issue_time="garbage"), "issue_time 를 읽을 수 없다"),
        (cc.SourceError("aq-data/forecast-band/latest.json: HTTPError status=500"), "status=500"),
    ],
)
def test_production_check_with_an_unreadable_latest_runs_and_warns(latest, expected):
    # Arrange
    client = FakeClient({bg.FORECAST_PATH: forecast(), pt.BAND_LATEST: latest})

    # Act
    decision, warnings = bg.check("production", False, client)

    # Assert
    assert decision.skip is False
    assert len(warnings) == 1 and expected in warnings[0]


@pytest.mark.parametrize(
    ("forecast_file", "expected"),
    [
        (None, "파일이 없다"),
        (b"<html>", "JSON 아님"),
        (doc(cities=[]), "generated_at 를 읽을 수 없다"),
        (forecast("garbage"), "generated_at 를 읽을 수 없다"),
        (forecast(12), "generated_at 를 읽을 수 없다"),
        (cc.SourceError("aq-data/forecast.json: URLError"), "URLError"),
    ],
)
@pytest.mark.parametrize("mode", ["production", "candidate"])
def test_unreadable_forecast_means_run_with_a_warning_and_no_further_requests(mode, forecast_file, expected):
    # Arrange — 이 파일 하나를 못 읽으면 어떤 판정도 할 수 없다
    client = FakeClient({bg.FORECAST_PATH: forecast_file, pt.BAND_LATEST: band(), CANDIDATE_PATH: band()})

    # Act
    decision, warnings = bg.check(mode, False, client)

    # Assert
    assert decision.skip is False
    assert len(warnings) == 1 and expected in warnings[0]
    assert client.paths == [bg.FORECAST_PATH]


@pytest.mark.parametrize(
    "issue_time",
    ["2026-10-15T06:15:42+00:00", "2026-10-15T06:15:42Z", "2026-10-15T15:15:42+09:00", "2026-10-15T06:40:00+00:00"],
    ids=["offset-zero", "z-suffix", "kst-offset", "same-hour-other-minute"],
)
def test_candidate_check_skips_when_the_stamped_file_is_that_issues_band_payload(issue_time):
    # Arrange — 판정 단위는 issue_time 의 UTC 시(STAMP)다. 표기가 달라도, 같은 시 안의 다른 분이어도 같은 발행으로 본다
    client = FakeClient({bg.FORECAST_PATH: forecast(), CANDIDATE_PATH: band(issue_time)})

    # Act
    decision, warnings = bg.check("candidate", False, client)

    # Assert — 후보 모드는 운영 latest 를 조회하지 않고, 후보 파일은 본문을 읽어 확인한다
    assert (decision.skip, warnings) == (True, [])
    assert client.paths == [bg.FORECAST_PATH, CANDIDATE_PATH]


@pytest.mark.parametrize(
    ("candidate_file", "expected"),
    [
        (b"<html>maintenance</html>", "JSON 아님"),  # 200 인데 점검 페이지
        (b"", "JSON 아님"),
        (b"[]", "JSON 객체 아님"),
        (b"\x80abc", "JSON 아님"),
        (doc(schema_version="w3-band-v1"), "issue_time"),  # schema 는 맞는데 issue_time 이 없다
        (band(issue_time="garbage"), "garbage"),
        (band(issue_time=None), "issue_time"),
        (band(schema_version="w3-band-v0"), "schema_version"),
        (band(schema_version=None), "schema_version"),
        (doc(issue_time="2026-10-15T06:15:42+00:00"), "schema_version"),  # issue 는 맞는데 band 발행물이 아니다
        (band("2026-10-15T00:15:40+00:00"), "issue 2026101500 의 발행물이다"),  # 다른 issue
        (band("2026-10-15T06:15:42+09:00"), "issue 2026101421 의 발행물이다"),  # 같은 시각 표기라도 UTC 로 바꾸면 다른 시
    ],
    ids=["html", "empty", "array", "not-utf8", "no-issue-time", "garbage-issue-time", "null-issue-time", "other-schema", "null-schema",
         "no-schema", "other-issue", "other-hour-in-utc"],
)
def test_candidate_check_runs_and_warns_when_the_stamped_file_is_not_that_issues_band_payload(candidate_file, expected):
    # Arrange — 파일이 응답해도(200) 그 issue 의 발행물이 아니면 "있다" 가 아니다: 건너뛰면 그 issue 의 후보 발행이 빠진다
    client = FakeClient({bg.FORECAST_PATH: forecast(), CANDIDATE_PATH: candidate_file})

    # Act
    decision, warnings = bg.check("candidate", False, client)

    # Assert
    assert decision.skip is False
    assert len(warnings) == 1 and warnings[0].startswith(f"{CANDIDATE_PATH}: ") and expected in warnings[0]
    assert "발행물이 아니다" in decision.reason


def test_candidate_check_runs_when_the_stamped_file_is_missing():
    # Arrange
    client = FakeClient({bg.FORECAST_PATH: forecast()})

    # Act
    decision, warnings = bg.check("candidate", False, client)

    # Assert
    assert (decision.skip, warnings) == (False, [])
    assert client.paths == [bg.FORECAST_PATH, CANDIDATE_PATH]


def test_candidate_check_runs_and_warns_when_the_lookup_fails():
    # Arrange
    client = FakeClient({bg.FORECAST_PATH: forecast(), CANDIDATE_PATH: cc.SourceError(f"{CANDIDATE_PATH}: TimeoutError")})

    # Act
    decision, warnings = bg.check("candidate", False, client)

    # Assert
    assert decision.skip is False
    assert warnings == [f"{CANDIDATE_PATH}: TimeoutError"]


def test_candidate_path_uses_the_utc_hour_of_the_forecast_issue():
    # Arrange — 한국 시각 15:15 = UTC 06:15 → STAMP 2026101506
    client = FakeClient({bg.FORECAST_PATH: forecast("2026-10-15T15:15:42+09:00")})

    # Act
    bg.check("candidate", False, client)

    # Assert
    assert client.paths[-1] == f"aq-data/_candidate/forecast-band/history/band_{STAMP}.json"


@pytest.mark.parametrize("mode", ["production", "candidate"])
def test_force_does_not_touch_the_network(mode):
    # Arrange
    client = FakeClient({bg.FORECAST_PATH: forecast(), pt.BAND_LATEST: band(), CANDIDATE_PATH: band()})

    # Act
    decision, warnings = bg.check(mode, True, client)

    # Assert
    assert (decision.skip, warnings, client.paths) == (False, [], [])


def test_paths_come_from_publish_targets():
    # Assert — 경로 문자열이 두 곳에 있으면 한쪽만 고쳐졌을 때 판정이 조용히 어긋난다
    assert pt.BAND_LATEST == "aq-data/forecast-band/latest.json"
    assert CANDIDATE_PATH == pt.band_targets("candidate", STAMP)[0].dest
    assert PROD_COPY_PATH == pt.band_targets("candidate", STAMP, STAMP)[1].dest


# ── main: GITHUB_OUTPUT · 요약 · 종료 코드 ───────────────────────────────────
def run_main(argv, client, tmp_path, capsys, *, with_files=True, **inject):
    out_file, summary_file = tmp_path / "output.txt", tmp_path / "summary.md"
    env = {"GITHUB_OUTPUT": str(out_file), "GITHUB_STEP_SUMMARY": str(summary_file)} if with_files else {}
    code = bg.main(argv, client=client, env=env, **inject)
    stdout = capsys.readouterr().out
    read = lambda p: p.read_text(encoding="utf-8") if p.exists() else None  # noqa: E731
    return code, stdout, read(out_file), read(summary_file)


@pytest.mark.parametrize(
    ("files", "skip"),
    [({bg.FORECAST_PATH: forecast(), CANDIDATE_PATH: band()}, "true"), ({bg.FORECAST_PATH: forecast()}, "false")],
)
def test_check_command_writes_the_skip_output_and_a_summary(tmp_path, capsys, files, skip):
    # Act
    code, stdout, output, summary = run_main(["check", "--mode", "candidate", "--force", "false"], FakeClient(files), tmp_path, capsys)

    # Assert
    assert code == 0
    assert output == f"skip={skip}\n"
    assert "### band 건너뛰기 판정" in summary and "`candidate`" in summary
    assert stdout.startswith("band_guard: ") and "::warning::" not in stdout


def test_check_command_appends_to_an_existing_output_file(tmp_path, capsys):
    # Arrange
    (tmp_path / "output.txt").write_text("earlier=1\n", encoding="utf-8")

    # Act
    _, _, output, _ = run_main(["check", "--mode", "candidate", "--force", "false"], FakeClient({bg.FORECAST_PATH: forecast()}), tmp_path, capsys)

    # Assert
    assert output == "earlier=1\nskip=false\n"


def test_check_command_prints_warnings_as_annotations_and_still_exits_zero(tmp_path, capsys):
    # Arrange
    client = FakeClient({bg.FORECAST_PATH: cc.SourceError("aq-data/forecast.json: URLError")})

    # Act
    code, stdout, output, _ = run_main(["check", "--mode", "production", "--force", "false"], client, tmp_path, capsys)

    # Assert — 판정을 못 해도 실패하지 않고 돈다
    assert code == 0 and output == "skip=false\n"
    assert "::warning::band_guard: aq-data/forecast.json: URLError" in stdout


def test_check_command_force_true_runs_without_any_request(tmp_path, capsys):
    # Arrange
    client = FakeClient({bg.FORECAST_PATH: forecast(), pt.BAND_LATEST: band()})

    # Act
    code, _, output, summary = run_main(["check", "--mode", "production", "--force", "true"], client, tmp_path, capsys)

    # Assert
    assert (code, output, client.paths) == (0, "skip=false\n", [])
    assert "force=true" in summary


def test_check_command_works_without_the_github_files(tmp_path, capsys):
    # Act — 로컬 실행: 환경변수가 없으면 출력 파일을 건드리지 않는다
    code, stdout, output, summary = run_main(["check", "--mode", "candidate", "--force", "false"], FakeClient({bg.FORECAST_PATH: forecast()}), tmp_path, capsys, with_files=False)

    # Assert
    assert code == 0 and (output, summary) == (None, None)
    assert "band_guard: 실행" in stdout


@pytest.mark.parametrize("argv", [["check", "--mode", "candidate", "--force", "maybe"], ["check", "--mode", "staging", "--force", "false"], ["check", "--mode", "candidate"], ["snapshot", "--mode", "candidate"], ["wait-slot", "--mode", "candidate"], ["bogus"], []])
def test_bad_arguments_exit_with_usage_error(argv):
    # Act / Assert
    with pytest.raises(SystemExit) as caught:
        bg.main(argv, client=FakeClient(), env={})
    assert caught.value.code == 2


# ── snapshot ────────────────────────────────────────────────────────────────
def test_snapshot_saves_the_published_latest_byte_for_byte(tmp_path, capsys):
    # Arrange — 따옴표·공백·순서까지 그대로여야 "그대로 옮긴 사본" 이다(키 순서를 뒤집고 들여쓰고 줄바꿈으로 끝낸 원문)
    raw = json.dumps(dict(reversed(band_doc().items())), indent=1).encode() + b"\n"
    assert raw != band()
    client = FakeClient({pt.BAND_LATEST: raw})
    dest = tmp_path / "prod" / "prod_latest.json"

    # Act
    code, stdout, _, _ = run_main(["snapshot", "--mode", "candidate", "--dest", str(dest)], client, tmp_path, capsys)

    # Assert
    assert code == 0 and dest.read_bytes() == raw
    assert f"issue {STAMP}" in stdout and "저장했다" in stdout and str(tmp_path) not in stdout
    assert client.paths == [pt.BAND_LATEST, PROD_COPY_PATH]


def test_snapshot_does_not_write_when_the_copy_for_that_issue_already_exists(tmp_path, capsys):
    # Arrange
    client = FakeClient({pt.BAND_LATEST: band(), PROD_COPY_PATH: band()})
    dest = tmp_path / "prod_latest.json"

    # Act
    code, stdout, _, _ = run_main(["snapshot", "--mode", "candidate", "--dest", str(dest)], client, tmp_path, capsys)

    # Assert
    assert code == 0 and not dest.exists()
    assert "이미 있다" in stdout and "::warning::" not in stdout


@pytest.mark.parametrize(
    ("copy_file", "expected"),
    [
        (b"<html>maintenance</html>", "JSON 아님"),  # 200 인데 점검 페이지
        (b"[]", "JSON 객체 아님"),
        (band(schema_version="w3-band-v0"), "schema_version"),
        (band(issue_time="garbage"), "garbage"),
        (band("2026-10-15T00:15:40+00:00"), "issue 2026101500 의 발행물이다"),  # 다른 issue 의 사본
    ],
    ids=["html", "array", "other-schema", "garbage-issue-time", "other-issue"],
)
def test_snapshot_writes_when_the_file_at_the_copy_path_is_not_that_issues_band_payload(tmp_path, capsys, copy_file, expected):
    # Arrange — 사본 경로가 응답해도 그 issue 의 발행물이 아니면 "이미 있다" 가 아니다: 건너뛰면 그 issue 는 짝이 영영 만들어지지 않는다
    client = FakeClient({pt.BAND_LATEST: band(), PROD_COPY_PATH: copy_file})
    dest = tmp_path / "prod_latest.json"

    # Act
    code, stdout, _, _ = run_main(["snapshot", "--mode", "candidate", "--dest", str(dest)], client, tmp_path, capsys)

    # Assert
    assert code == 0 and dest.read_bytes() == band()
    warnings = [line for line in stdout.splitlines() if line.startswith("::warning::")]
    assert len(warnings) == 1 and warnings[0].startswith(f"::warning::band_guard: {PROD_COPY_PATH}: ") and expected in warnings[0]
    assert "저장했다" in stdout


def test_snapshot_in_production_mode_does_nothing_and_never_touches_the_network(tmp_path, capsys):
    # Arrange
    client = FakeClient({pt.BAND_LATEST: band()})
    dest = tmp_path / "prod_latest.json"

    # Act
    code, stdout, _, _ = run_main(["snapshot", "--mode", "production", "--dest", str(dest)], client, tmp_path, capsys)

    # Assert
    assert (code, client.paths, dest.exists()) == (0, [], False)
    assert "필요 없다" in stdout


@pytest.mark.parametrize(
    ("latest", "expected"),
    [
        (None, "아직 없다"),
        (cc.SourceError(f"{pt.BAND_LATEST}: HTTPError status=500"), "받지 못해"),
        (b"<html>", "받지 못해"),
        (band(schema_version="w3-band-v0"), "모양이 아니라"),
        (band(schema_version=None), "모양이 아니라"),
        (band(issue_time="garbage"), "모양이 아니라"),
        (band(issue_time=None), "모양이 아니라"),
    ],
)
def test_snapshot_without_a_usable_latest_leaves_no_file_and_never_fails_the_step(tmp_path, capsys, latest, expected):
    # Arrange
    client = FakeClient({pt.BAND_LATEST: latest})
    dest = tmp_path / "prod_latest.json"

    # Act
    code, stdout, _, _ = run_main(["snapshot", "--mode", "candidate", "--dest", str(dest)], client, tmp_path, capsys)

    # Assert — 사본은 보조 자료다: 못 받아도 후보 발행을 막지 않는다
    assert code == 0 and not dest.exists()
    assert expected in stdout


@pytest.mark.parametrize(
    "break_it",
    [
        lambda d: d.update(cities=[]),  # 도시가 하나도 없다
        lambda d: d["cities"][0]["horizons"][0].update(p10=-1.0),  # 음수 농도
        lambda d: d.pop("uncertainty"),  # 필수 필드 누락
    ],
    ids=["no-cities", "negative-p10", "missing-uncertainty"],
)
def test_snapshot_does_not_keep_a_copy_that_breaks_the_band_contract_and_still_exits_zero(tmp_path, capsys, break_it):
    # Arrange — 모양(schema_version·issue_time)은 맞지만 계약을 어긴 운영 latest. 남기면 발행기의 계약 검증이 batch 전체를 멈춘다.
    broken = band_doc()
    break_it(broken)
    client = FakeClient({pt.BAND_LATEST: json.dumps(broken).encode()})
    dest = tmp_path / "prod_latest.json"

    # Act
    code, stdout, _, _ = run_main(["snapshot", "--mode", "candidate", "--dest", str(dest)], client, tmp_path, capsys)

    # Assert — 스텝은 성공이고(후보 발행을 막지 않는다) 사본 파일은 없다. 같은 issue 의 사본 존재 확인까지 가지 않는다
    assert code == 0 and not dest.exists()
    assert f"::warning::band_guard: {pt.BAND_LATEST}: band 계약을 어겼다" in stdout
    assert "계약을 어겨 사본을 남기지 않는다" in stdout
    assert client.paths == [pt.BAND_LATEST]


def test_snapshot_warning_for_a_contract_violation_is_one_bounded_line(tmp_path, capsys):
    # Arrange — 위반 설명에 줄바꿈이 섞여 있어도 로그에서 워크플로 명령(`::...`)을 흉내 낼 수 없고, 길이도 짧다
    client = FakeClient({pt.BAND_LATEST: band()})
    hostile = "x\n::error::injected" + "y" * 500

    # Act
    code = bg.main(["snapshot", "--mode", "candidate", "--dest", str(tmp_path / "x.json")], client=client, env={}, validate=lambda _doc: hostile)

    # Assert
    commands = [line for line in capsys.readouterr().out.splitlines() if line.startswith("::")]
    assert code == 0
    assert len(commands) == 1 and commands[0].startswith("::warning::band_guard: ") and len(commands[0]) < 500


def test_snapshot_still_writes_when_the_copy_lookup_fails_and_says_so(tmp_path, capsys):
    # Arrange
    client = FakeClient({pt.BAND_LATEST: band(), PROD_COPY_PATH: cc.SourceError(f"{PROD_COPY_PATH}: TimeoutError")})
    dest = tmp_path / "prod_latest.json"

    # Act
    code, stdout, _, _ = run_main(["snapshot", "--mode", "candidate", "--dest", str(dest)], client, tmp_path, capsys)

    # Assert
    assert code == 0 and dest.read_bytes() == band()
    assert f"::warning::band_guard: {PROD_COPY_PATH}: TimeoutError" in stdout


def test_snapshot_names_the_copy_after_the_published_issue_not_the_forecast_issue(tmp_path, capsys):
    # Arrange — 운영 latest 의 issue 는 한 발행 전(00시)이다. 사본 경로는 이 issue 를 따라야 나중에 같은 issue 끼리 짝지어진다.
    client = FakeClient({pt.BAND_LATEST: band("2026-10-15T00:15:40+00:00")})

    # Act
    run_main(["snapshot", "--mode", "candidate", "--dest", str(tmp_path / "x.json")], client, tmp_path, capsys)

    # Assert
    assert client.paths[-1] == "aq-data/_candidate/forecast-band/prod/band_2026101500.json"


@pytest.mark.parametrize("issue_time", ["2026-10-15T06:15:42Z", "2026-10-15T06:15:42+00:00", "2026-10-15T15:15:42+09:00", "2026-10-15T06:15:42"])
def test_a_copy_that_snapshot_saves_is_accepted_by_the_publish_mapping_too(tmp_path, capsys, issue_time):
    # Arrange — snapshot 이 받은 사본을 발행 매핑이 거부하면 batch 전체가 멈춰 후보 발행이 막힌다. 두 쪽이 같은 파서를 쓴다.
    dest = tmp_path / "prod" / "prod_latest.json"
    run_main(["snapshot", "--mode", "candidate", "--dest", str(dest)], FakeClient({pt.BAND_LATEST: band(issue_time)}), tmp_path, capsys)
    artifact = tmp_path / "pub"
    artifact.mkdir()
    (artifact / pt.BAND_FILE).write_bytes(band())

    # Act
    code = pt.main(["maps", "--mode", "candidate", "--dir", str(artifact), "--group", "band", "--prod-snapshot", str(dest)])

    # Assert
    assert dest.is_file() and code == 0
    assert capsys.readouterr().out.splitlines()[-1].endswith(f"={PROD_COPY_PATH}")


# ── wait-slot ───────────────────────────────────────────────────────────────
# 예보 issue(ISSUE_Z = 06:15:42Z)의 시는 06 시 → 기다리는 슬롯은 07 시(`openaq-…2026101507.json.gz`), 마감은 07:30Z.
# band 는 예보 수집 직후(06:20Z)에 시작한다. 시계·sleep 은 가짜라 실제로 자지 않고 네트워크도 0 이다.
SLOT_STAMP = "2026101507"
SLOT_PATH = "openaq-shadow/openaq-2026101507.json.gz"  # 경로 문자열은 일부러 리터럴로 둔다(상수를 거치면 오타가 시험에서도 같이 숨는다)
SLOT_DEADLINE = datetime(2026, 10, 15, 7, 30, tzinfo=timezone.utc)
STARTED = datetime(2026, 10, 15, 6, 20, tzinfo=timezone.utc)
GZ = gzip.compress(b"[]")


class FakeClock:
    """가짜 시계 — `sleep` 이 시각을 앞으로 돌린다(실제로는 자지 않는다). 잠든 시간(초)을 순서대로 기록한다."""

    def __init__(self, at: datetime = STARTED) -> None:
        self.at = at
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self.at

    def sleep(self, seconds: float) -> None:
        assert len(self.sleeps) < 500, "대기가 끝나지 않는다"  # 마감 판정이 깨져도 시험이 멈추지 않고 실패하게 한다(정상은 많아야 수십 번)
        self.sleeps.append(seconds)
        self.at += timedelta(seconds=seconds)


class SlotClient(FakeClient):
    """관측 슬롯 경로를 조회한 차례대로 `replies` 를 돌려주는 대역(None = 404, 예외는 던진다, 다 쓰면 마지막 응답을 되풀이한다).
    그 밖의 경로(forecast.json)는 `FakeClient` 와 같다."""

    def __init__(self, replies: list[object], files: dict[str, object] | None = None, slot_path: str = SLOT_PATH) -> None:
        super().__init__({bg.FORECAST_PATH: forecast()} if files is None else files)
        self.replies = list(replies)
        self.slot_path = slot_path

    def read_bytes(self, rel_path: str) -> bytes | None:
        if rel_path != self.slot_path:
            return super().read_bytes(rel_path)
        self.paths.append(rel_path)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, BaseException):
            raise reply
        return reply  # type: ignore[return-value]

    def polls(self) -> int:
        return self.paths.count(self.slot_path)


def wait(client: SlotClient, clock: FakeClock | None = None, **kwargs) -> tuple[str, list[str], FakeClock]:
    clock = clock or FakeClock()
    warnings: list[str] = []
    message = bg.wait_for_slot(client, warnings, now=clock.now, sleep=clock.sleep, **kwargs)
    return message, warnings, clock


@pytest.mark.parametrize(
    ("issue", "stamp"),
    [
        ("2026-10-16T16:42:00+00:00", "2026101617"),  # 시 중간 → 다음 시
        ("2026-10-16T16:00:00+00:00", "2026101617"),  # 정각이어도 슬롯 H 가 아니라 H+1 이다
        ("2026-10-16T23:30:00+00:00", "2026101700"),  # 날짜가 넘어간다
        ("2026-10-17T01:42:00+09:00", "2026101617"),  # 한국 시각 01:42 = UTC 16:42
        ("2026-10-17T07:42:00+05:30", "2026101703"),  # 반 시간 오프셋: UTC 02:12 → 03 시(오프셋 시각을 먼저 내리면 02 시가 되어 어긋난다)
    ],
)
def test_wait_plan_names_the_slot_after_the_utc_hour_of_the_issue(issue, stamp):
    # Arrange
    moment = datetime.fromisoformat(issue)

    # Act
    planned, _deadline = bg.wait_plan(moment, moment)

    # Assert
    assert planned == stamp


@pytest.mark.parametrize("issue", ["2026-10-16T16:00:00+00:00", "2026-10-16T16:42:17+00:00", "2026-10-16T16:59:59+00:00"])
def test_wait_plan_deadline_is_half_past_the_slot_hour_whatever_the_minute_of_the_issue(issue):
    # Arrange
    moment = datetime.fromisoformat(issue)

    # Act
    _stamp, deadline = bg.wait_plan(moment, moment)

    # Assert — 슬롯 H+1 은 보통 H+1 시 16~19분에 온다: 마감 = H+1 시 30분
    assert deadline == datetime(2026, 10, 16, 17, 30, tzinfo=timezone.utc)


def test_wait_plan_deadline_never_goes_past_the_max_wait_after_the_start():
    # Arrange — issue 가 시작보다 한참 뒤인 이상한 값(H+1 시 30분은 21:30)
    start = datetime(2026, 10, 16, 16, 50, tzinfo=timezone.utc)
    odd_issue = datetime(2026, 10, 16, 20, 10, tzinfo=timezone.utc)

    # Act
    _stamp, deadline = bg.wait_plan(odd_issue, start)

    # Assert
    assert deadline == datetime(2026, 10, 16, 18, 20, tzinfo=timezone.utc)


def test_wait_returns_at_once_when_the_slot_is_already_there():
    # Arrange — 백업 schedule · force · 늦은 run 은 슬롯이 이미 올라와 있다
    client = SlotClient([GZ])

    # Act
    message, warnings, clock = wait(client)

    # Assert
    assert message == f"관측 슬롯 {SLOT_STAMP} 이 있다 — 0분 기다렸다"
    assert (warnings, clock.sleeps) == ([], [])
    assert client.paths == [bg.FORECAST_PATH, SLOT_PATH]


def test_wait_sleeps_one_interval_per_miss_until_the_slot_appears():
    # Arrange — 세 번 없다가 네 번째 조회에서 올라온다
    client = SlotClient([None, None, None, GZ])

    # Act
    message, warnings, clock = wait(client)

    # Assert
    assert clock.sleeps == [120, 120, 120] and client.polls() == 4
    assert message == f"관측 슬롯 {SLOT_STAMP} 이 있다 — 6분 기다렸다"
    assert warnings == []


def test_wait_polls_at_the_interval_it_is_given():
    # Arrange
    client = SlotClient([None, GZ])

    # Act
    _message, _warnings, clock = wait(client, interval_s=30)

    # Assert
    assert clock.sleeps == [30]


def test_wait_stops_at_the_deadline_and_warns_once_when_the_slot_never_comes():
    # Arrange — 06:20 에 시작, 마감 07:30
    client = SlotClient([None])

    # Act
    message, warnings, clock = wait(client)

    # Assert — 70분을 120초씩 35번 자고, 마감 시각에 한 번 더 조회한 뒤 멈춘다. forecast.json 은 한 번만 읽는다
    assert clock.sleeps == [120] * 35 and clock.at == SLOT_DEADLINE
    assert client.polls() == 36 and client.paths.count(bg.FORECAST_PATH) == 1
    assert len(warnings) == 1 and "기다리지 않고 돈다" in warnings[0]
    assert SLOT_STAMP in warnings[0] and SLOT_DEADLINE.isoformat() in warnings[0]
    assert message == f"관측 슬롯 {SLOT_STAMP} 이 마감까지 올라오지 않았다 — 70분 기다린 뒤 그대로 돈다"


def test_wait_cuts_the_last_sleep_to_the_time_left_so_it_never_passes_the_deadline():
    # Arrange — 06:20:30 에 시작: 마감까지 4170초 = 120 × 34 + 90
    started = STARTED + timedelta(seconds=30)

    # Act
    _message, _warnings, clock = wait(SlotClient([None]), FakeClock(started))

    # Assert
    assert clock.sleeps == [120] * 34 + [90]
    assert sum(clock.sleeps) == (SLOT_DEADLINE - started).total_seconds()


def test_wait_never_waits_longer_than_max_wait_even_when_the_issue_looks_like_the_future():
    # Arrange — generated_at 이 시작보다 5시간 뒤: 그대로 H+1 시 30분(12:30)까지 기다리면 6시간 10분이다
    slot_path = "openaq-shadow/openaq-2026101512.json.gz"
    client = SlotClient([None], {bg.FORECAST_PATH: forecast("2026-10-15T11:20:00Z")}, slot_path=slot_path)

    # Act
    _message, warnings, clock = wait(client)

    # Assert — 시작 06:20 + 90분
    assert clock.at == datetime(2026, 10, 15, 7, 50, tzinfo=timezone.utc)
    assert sum(clock.sleeps) == 90 * 60 and max(clock.sleeps) <= 120
    assert client.polls() == 46 and len(warnings) == 1


def test_wait_checks_only_once_when_the_deadline_has_already_passed_at_the_start():
    # Arrange — 늦은 run: 시작(08:00)이 마감(07:30)보다 뒤다
    client = SlotClient([None])

    # Act
    message, warnings, clock = wait(client, FakeClock(datetime(2026, 10, 15, 8, 0, tzinfo=timezone.utc)))

    # Assert
    assert clock.sleeps == [] and client.polls() == 1
    assert len(warnings) == 1 and "기다리지 않고 돈다" in warnings[0] and "그대로 돈다" in message


def test_wait_is_quiet_when_a_late_run_finds_the_slot_already_there():
    # Arrange — 마감이 한참 지난 시작이어도 슬롯이 있으면 경고 없이 끝난다(백업 schedule 이 늦게 도는 흔한 경우)
    client = SlotClient([GZ])

    # Act
    message, warnings, clock = wait(client, FakeClock(datetime(2026, 10, 15, 9, 0, tzinfo=timezone.utc)))

    # Assert
    assert (warnings, clock.sleeps) == ([], []) and "이 있다" in message


@pytest.mark.parametrize(
    ("forecast_file", "expected"),
    [
        (None, "파일이 없다"),
        (b"<html>", "JSON 아님"),
        (doc(cities=[]), "generated_at 를 읽을 수 없다"),
        (forecast("garbage"), "generated_at 를 읽을 수 없다"),
        (cc.SourceError("aq-data/forecast.json: URLError"), "URLError"),
    ],
)
def test_wait_does_not_wait_when_the_forecast_issue_is_unknown(forecast_file, expected):
    # Arrange — 마감을 정할 수 없다. 기다림은 최적화일 뿐이라 경고만 남기고 슬롯은 조회하지 않는다
    client = SlotClient([GZ], {bg.FORECAST_PATH: forecast_file})

    # Act
    message, warnings, clock = wait(client)

    # Assert
    assert "기다리지 않는다" in message and clock.sleeps == []
    assert len(warnings) == 1 and expected in warnings[0]
    assert client.paths == [bg.FORECAST_PATH]


@pytest.mark.parametrize("issue", ["9999-12-31T23:00:00Z", "9999-12-31T23:30:00Z", "9999-12-31T23:59:59Z"])
def test_wait_does_not_wait_when_the_next_hour_is_beyond_the_range_a_datetime_can_hold(issue):
    # Arrange — 9999-12-31 23 시대: H+1 시가 표현 범위를 넘어 슬롯을 정할 수 없다. 원격이 준 값이라 예외로 band 를 막으면 안 된다
    client = SlotClient([GZ], {bg.FORECAST_PATH: forecast(issue)})

    # Act
    message, warnings, clock = wait(client)

    # Assert — 자지도 슬롯을 조회하지도 않고, 경고는 파싱한 시각만 담은 한 줄이다
    assert message == "예보 발행 시각으로 기다릴 슬롯을 정할 수 없어 기다리지 않는다"
    assert clock.sleeps == [] and client.polls() == 0 and client.paths == [bg.FORECAST_PATH]
    assert warnings == [f"예보 발행 시각 {issue.replace('Z', '+00:00')} 의 다음 관측 슬롯을 정할 수 없다 — 기다리지 않고 돈다"]


def test_wait_treats_a_failed_poll_as_not_there_yet_and_keeps_waiting():
    # Arrange
    error = cc.SourceError(f"{SLOT_PATH}: HTTPError status=503")
    client = SlotClient([error, None, GZ])

    # Act
    message, warnings, clock = wait(client)

    # Assert
    assert clock.sleeps == [120, 120] and "이 있다" in message
    assert warnings == [str(error)]


@pytest.mark.parametrize("body", [b"<html>maintenance</html>", b"", b"{}", b"\x1f"], ids=["html", "empty", "json", "half-magic"])
def test_wait_does_not_take_a_200_response_that_is_not_gzip_for_the_slot(body):
    # Arrange — 점검 페이지 같은 200 응답을 "있다" 로 보면 슬롯이 없는데 기다림이 끝난다
    client = SlotClient([body, GZ])

    # Act
    message, warnings, clock = wait(client)

    # Assert
    assert clock.sleeps == [120] and "이 있다" in message
    assert len(warnings) == 1 and warnings[0].startswith(f"{SLOT_PATH}: ") and "gzip 이 아니다" in warnings[0]


def test_wait_does_not_swallow_unexpected_errors():
    # Arrange — 읽기 실패(SourceError)만 "아직 없음" 이다. 그 밖의 예외는 버그라 드러나야 한다
    client = SlotClient([RuntimeError("bug")])

    # Act / Assert
    with pytest.raises(RuntimeError):
        wait(client)


def test_wait_slot_command_prints_the_result_and_a_summary_and_writes_no_output(tmp_path, capsys):
    # Arrange
    clock = FakeClock()
    client = SlotClient([None, GZ])

    # Act
    code, stdout, output, summary = run_main(["wait-slot"], client, tmp_path, capsys, now=clock.now, sleep=clock.sleep)

    # Assert — 다음 잡이 따르는 값(skip 같은 것)은 없다: GITHUB_OUTPUT 에는 쓰지 않는다. 기다리기 시작할 때 한 줄, 끝에 결과 한 줄
    result = f"관측 슬롯 {SLOT_STAMP} 이 있다 — 2분 기다렸다"
    assert code == 0 and output is None
    assert stdout.splitlines() == [
        f"band_guard: 관측 슬롯 {SLOT_STAMP} 이 아직 없다 — 마감 {SLOT_DEADLINE.isoformat()} 까지 120초마다 확인한다",
        f"band_guard: {result}",
    ]
    assert summary == f"### band 관측 슬롯 대기\n\n- 결과: {result}\n\n"


def test_wait_logs_one_line_when_it_starts_sleeping_and_nothing_when_it_does_not_sleep():
    # Arrange — 결과·경고는 끝에 나오므로, 기다리는 동안(최대 90분) 로그가 비지 않게 시작할 때 한 줄만 남긴다
    waiting, present, late = [], [], []

    # Act
    wait(SlotClient([None, None, GZ]), log=waiting.append)
    wait(SlotClient([GZ]), log=present.append)
    wait(SlotClient([None]), FakeClock(datetime(2026, 10, 15, 8, 0, tzinfo=timezone.utc)), log=late.append)

    # Assert — 자기 전에 한 번만. 이미 있거나 마감이 지나 자지 않으면 남기지 않는다(결과 줄로 충분하다)
    assert waiting == [f"관측 슬롯 {SLOT_STAMP} 이 아직 없다 — 마감 {SLOT_DEADLINE.isoformat()} 까지 120초마다 확인한다"]
    assert (present, late) == ([], [])


def test_wait_counts_the_forecast_read_inside_the_max_wait():
    # Arrange — forecast.json 을 읽는 데 10분이 걸린다(재시도 등). issue 가 미래라 마감은 시작 + MAX_WAIT 로 정해진다
    clock = FakeClock()
    slot_path = "openaq-shadow/openaq-2026101512.json.gz"

    class SlowForecast(SlotClient):
        def read_bytes(self, rel_path: str) -> bytes | None:
            if rel_path == bg.FORECAST_PATH:
                clock.at += timedelta(minutes=10)
            return super().read_bytes(rel_path)

    client = SlowForecast([None], {bg.FORECAST_PATH: forecast("2026-10-15T11:20:00Z")}, slot_path=slot_path)

    # Act
    wait(client, clock)

    # Assert — 상한은 읽기 전에 잰 시작(06:20)부터다: 07:50 에 멈춘다(읽은 뒤부터 쟀다면 08:00)
    assert clock.at == STARTED + bg.MAX_WAIT


def test_wait_slot_command_exits_zero_and_warns_when_the_forecast_cannot_be_read(tmp_path, capsys):
    # Arrange
    client = SlotClient([GZ], {bg.FORECAST_PATH: cc.SourceError("aq-data/forecast.json: URLError")})
    clock = FakeClock()

    # Act
    code, stdout, _, _ = run_main(["wait-slot"], client, tmp_path, capsys, now=clock.now, sleep=clock.sleep)

    # Assert — 기다릴 수 없어도 실패하지 않는다(band 를 막지 않는다)
    assert code == 0
    assert stdout.splitlines() == ["::warning::band_guard: aq-data/forecast.json: URLError", "band_guard: 예보 발행 시각을 몰라 기다리지 않는다"]


def test_wait_slot_command_exits_zero_with_one_warning_when_the_next_hour_is_beyond_the_range_a_datetime_can_hold(tmp_path, capsys):
    # Arrange
    client, clock = SlotClient([GZ], {bg.FORECAST_PATH: forecast("9999-12-31T23:30:00Z")}), FakeClock()

    # Act
    code, stdout, output, summary = run_main(["wait-slot"], client, tmp_path, capsys, now=clock.now, sleep=clock.sleep)

    # Assert — 예외로 끝나지 않는다: 종료 코드 0, 경고 한 줄 + 결과 한 줄
    result = "예보 발행 시각으로 기다릴 슬롯을 정할 수 없어 기다리지 않는다"
    assert code == 0 and output is None
    assert stdout.splitlines() == [
        "::warning::band_guard: 예보 발행 시각 9999-12-31T23:30:00+00:00 의 다음 관측 슬롯을 정할 수 없다 — 기다리지 않고 돈다",
        f"band_guard: {result}",
    ]
    assert summary == f"### band 관측 슬롯 대기\n\n- 결과: {result}\n\n"


@pytest.mark.parametrize(
    "generated_at",
    ["2026-10-16T16:42:00Z\n::error::x", "%0A::set-env name=A::b", "2026-10-16T16:42:00Z\u2028::error::x", "2026-10-16T16:42:00Z\r::error::x"],
    ids=["newline", "escaped-newline", "line-separator", "carriage-return"],
)
def test_wait_slot_command_never_lets_a_remote_timestamp_start_a_log_line_of_its_own(tmp_path, capsys, generated_at):
    # Arrange — forecast.json 은 원격이 준 값이다. 줄바꿈 문자나 `::명령::` 모양이 섞여도 러너가 워크플로 명령으로 읽을 줄이 생기면 안 된다
    client, clock = SlotClient([GZ], {bg.FORECAST_PATH: forecast(generated_at)}), FakeClock()

    # Act
    code, stdout, _, _ = run_main(["wait-slot"], client, tmp_path, capsys, now=clock.now, sleep=clock.sleep)

    # Assert — 경고 한 줄 + 결과 한 줄뿐이고 모든 줄이 우리 접두사로 시작한다(`splitlines` 는 `\r`·U+2028 도 줄 끝으로 본다)
    lines = stdout.splitlines()
    assert code == 0 and len(lines) == 2
    assert all(line.startswith(("band_guard: ", "::warning::band_guard: ")) for line in lines)
    assert lines[0].startswith("::warning::") and not lines[1].startswith("::")


def test_wait_slot_command_prints_a_repeated_poll_failure_once_and_still_exits_zero(tmp_path, capsys):
    # Arrange — 마감까지 36번 조회하는 내내 같은 오류
    error = f"{SLOT_PATH}: TimeoutError"
    client = SlotClient([cc.SourceError(error)])
    clock = FakeClock()

    # Act
    code, stdout, _, summary = run_main(["wait-slot"], client, tmp_path, capsys, now=clock.now, sleep=clock.sleep)

    # Assert — 같은 경고는 한 줄이다(처음 나온 순서). 마감을 넘겨도 실패하지 않는다
    lines = stdout.splitlines()
    annotations = [line for line in lines if line.startswith("::warning::")]
    assert code == 0 and client.polls() == 36
    assert annotations[0] == f"::warning::band_guard: {error}"
    assert len(annotations) == 2 and "기다리지 않고 돈다" in annotations[1]
    assert lines[-1].startswith("band_guard: ") and "그대로 돈다" in lines[-1] and "그대로 돈다" in summary


def test_wait_slot_command_uses_the_real_clock_and_sleep_by_default(tmp_path, capsys, monkeypatch):
    # Arrange — 진짜 시계(UTC, tz 있음)와 `time.sleep` 이 기본이다. issue 를 지금부터 5시간 뒤로 둬서 마감이 시작 + 90분이 되게 하고
    # (시험을 도는 시각과 무관하다), 실제로 자지 않도록 `time.sleep` 만 바꿔치운다
    slept: list[float] = []
    monkeypatch.setattr(bg.time, "sleep", slept.append)
    issue = datetime.now(timezone.utc) + timedelta(hours=5)
    slot = f"openaq-shadow/openaq-{pt.stamp_of(issue.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))}.json.gz"
    client = SlotClient([None, GZ], {bg.FORECAST_PATH: forecast(issue.isoformat())}, slot_path=slot)

    # Act
    code, stdout, _, _ = run_main(["wait-slot"], client, tmp_path, capsys)

    # Assert
    assert code == 0 and slept == [120]
    assert "이 있다" in stdout and "::warning::" not in stdout
