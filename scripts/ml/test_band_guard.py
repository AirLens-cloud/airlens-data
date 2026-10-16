"""band_guard 시험 — 판정은 순수 함수, 조회는 가짜 응답이라 네트워크가 0 이다."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

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
    조회한 (경로, 본문을 읽었나) 를 기록한다. 재시도·상태 코드·헤더는 test_compare_candidate.py 가 `HttpSource` 로 시험한다."""

    def __init__(self, files: dict[str, object] | None = None) -> None:
        super().__init__(sleep=lambda _s: None)
        self.files = files or {}
        self.calls: list[tuple[str, bool]] = []

    def _lookup(self, rel_path: str, *, read_body: bool) -> bytes | None:
        self.calls.append((rel_path, read_body))
        value = self.files.get(rel_path)
        if isinstance(value, BaseException):
            raise value
        if value is None:
            return None
        return value if read_body else b""  # type: ignore[return-value]

    def read_bytes(self, rel_path: str) -> bytes | None:
        return self._lookup(rel_path, read_body=True)

    def exists(self, rel_path: str) -> bool:
        return self._lookup(rel_path, read_body=False) is not None

    @property
    def paths(self) -> list[str]:
        return [path for path, _ in self.calls]


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
    client = FakeClient({bg.FORECAST_PATH: forecast_file, pt.BAND_LATEST: band(), CANDIDATE_PATH: b"{}"})

    # Act
    decision, warnings = bg.check(mode, False, client)

    # Assert
    assert decision.skip is False
    assert len(warnings) == 1 and expected in warnings[0]
    assert client.paths == [bg.FORECAST_PATH]


def test_candidate_check_skips_when_the_stamped_file_exists():
    # Arrange
    client = FakeClient({bg.FORECAST_PATH: forecast(), CANDIDATE_PATH: b"{}"})

    # Act
    decision, warnings = bg.check("candidate", False, client)

    # Assert — 후보 모드는 운영 latest 를 조회하지 않고, 후보 파일은 열어 보기만 한다
    assert (decision.skip, warnings) == (True, [])
    assert client.calls == [(bg.FORECAST_PATH, True), (CANDIDATE_PATH, False)]


def test_candidate_check_runs_when_the_stamped_file_is_missing():
    # Arrange
    client = FakeClient({bg.FORECAST_PATH: forecast()})

    # Act
    decision, warnings = bg.check("candidate", False, client)

    # Assert
    assert (decision.skip, warnings) == (False, [])
    assert client.paths == [bg.FORECAST_PATH, CANDIDATE_PATH]


def test_candidate_check_runs_and_warns_when_the_existence_check_fails():
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
    client = FakeClient({bg.FORECAST_PATH: forecast(), pt.BAND_LATEST: band(), CANDIDATE_PATH: b"{}"})

    # Act
    decision, warnings = bg.check(mode, True, client)

    # Assert
    assert (decision.skip, warnings, client.calls) == (False, [], [])


def test_paths_come_from_publish_targets():
    # Assert — 경로 문자열이 두 곳에 있으면 한쪽만 고쳐졌을 때 판정이 조용히 어긋난다
    assert pt.BAND_LATEST == "aq-data/forecast-band/latest.json"
    assert CANDIDATE_PATH == pt.band_targets("candidate", STAMP)[0].dest
    assert PROD_COPY_PATH == pt.band_targets("candidate", STAMP, STAMP)[1].dest


# ── main: GITHUB_OUTPUT · 요약 · 종료 코드 ───────────────────────────────────
def run_main(argv, client, tmp_path, capsys, *, with_files=True):
    out_file, summary_file = tmp_path / "output.txt", tmp_path / "summary.md"
    env = {"GITHUB_OUTPUT": str(out_file), "GITHUB_STEP_SUMMARY": str(summary_file)} if with_files else {}
    code = bg.main(argv, client=client, env=env)
    stdout = capsys.readouterr().out
    read = lambda p: p.read_text(encoding="utf-8") if p.exists() else None  # noqa: E731
    return code, stdout, read(out_file), read(summary_file)


@pytest.mark.parametrize(
    ("files", "skip"),
    [({bg.FORECAST_PATH: forecast(), CANDIDATE_PATH: b"{}"}, "true"), ({bg.FORECAST_PATH: forecast()}, "false")],
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
    assert (code, output, client.calls) == (0, "skip=false\n", [])
    assert "force=true" in summary


def test_check_command_works_without_the_github_files(tmp_path, capsys):
    # Act — 로컬 실행: 환경변수가 없으면 출력 파일을 건드리지 않는다
    code, stdout, output, summary = run_main(["check", "--mode", "candidate", "--force", "false"], FakeClient({bg.FORECAST_PATH: forecast()}), tmp_path, capsys, with_files=False)

    # Assert
    assert code == 0 and (output, summary) == (None, None)
    assert "band_guard: 실행" in stdout


@pytest.mark.parametrize("argv", [["check", "--mode", "candidate", "--force", "maybe"], ["check", "--mode", "staging", "--force", "false"], ["check", "--mode", "candidate"], ["snapshot", "--mode", "candidate"], ["bogus"], []])
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
    assert client.calls == [(pt.BAND_LATEST, True), (PROD_COPY_PATH, False)]


def test_snapshot_does_not_write_when_the_copy_for_that_issue_already_exists(tmp_path, capsys):
    # Arrange
    client = FakeClient({pt.BAND_LATEST: band(), PROD_COPY_PATH: b"{}"})
    dest = tmp_path / "prod_latest.json"

    # Act
    code, stdout, _, _ = run_main(["snapshot", "--mode", "candidate", "--dest", str(dest)], client, tmp_path, capsys)

    # Assert
    assert code == 0 and not dest.exists()
    assert "이미 있다" in stdout


def test_snapshot_in_production_mode_does_nothing_and_never_touches_the_network(tmp_path, capsys):
    # Arrange
    client = FakeClient({pt.BAND_LATEST: band()})
    dest = tmp_path / "prod_latest.json"

    # Act
    code, stdout, _, _ = run_main(["snapshot", "--mode", "production", "--dest", str(dest)], client, tmp_path, capsys)

    # Assert
    assert (code, client.calls, dest.exists()) == (0, [], False)
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
    assert client.calls == [(pt.BAND_LATEST, True)]


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


def test_snapshot_still_writes_when_the_existence_check_fails_and_says_so(tmp_path, capsys):
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
