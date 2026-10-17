"""summarize_predict_log.py 시험 (AAA, 네트워크 0).

predict 는 실시간 기상 조회가 실패한 지점을 상수 기상으로 대체하고 그 규모를 로그 한 줄로만 남긴다.
이 스크립트는 그 한 줄을 run 요약과 경고로 올린다 — 문구가 바뀌어 0건으로 보이는 일이 없도록 실제 문구로 시험한다.
"""
from __future__ import annotations

import pytest

import summarize_predict_log as spl

# predict 가 실제로 찍는 형태 (이모지 + 건수 + 대시 설명)
REAL_LINE = "⚠️ met 실시간 조회 실패 7건 — 해당 지점 기상은 상수 fallback"
# wheel 0.3.1 부터는 건수 뒤에 이번 실행의 원인별 수가 괄호로 붙는다
REASONS_LINE = "⚠️ met 실시간 조회 실패 177건 (ReadTimeout 150, HTTP 503 27) — 해당 지점 기상은 상수 fallback"
# wheel 0.3.2 부터는 지점별 조회 예산을 다 써서 묻지 않은 지점도 건수에 들고, 원인에 `시간 예산 초과 N` 으로 붙는다
BUDGET_LINE = "⚠️ met 실시간 조회 실패 312건 (시간 예산 초과 192, ReadTimeout 120) — 해당 지점 기상은 상수 fallback"
BUDGET_NOTE = "⏳ met 지점별 조회 예산 900초를 다 써서 남은 지점은 조회하지 않았다"


def test_met_failures_reads_the_count_from_the_real_log_line():
    # Arrange
    log = "loading model\n" + REAL_LINE + "\ndone 516 stations\n"

    # Act / Assert
    assert spl.met_failures(log) == 7


def test_met_failures_reads_only_the_count_when_reasons_follow():
    # Arrange — 괄호 안 원인별 수(150·503·27)는 건수가 아니다
    log = "loading model\n" + REASONS_LINE + "\ndone 496 stations\n"

    # Act / Assert
    assert spl.met_failures(log) == 177


def test_met_failures_counts_budget_skipped_stations_once():
    # Arrange — 예산 초과분(192)은 건수(312)에 이미 들어 있다. 예산 안내 줄은 건수가 아니다
    log = "loading model\n" + BUDGET_NOTE + "\n" + BUDGET_LINE + "\ndone 496 stations\n"

    # Act / Assert
    assert spl.met_failures(log) == 312


def test_met_failures_is_zero_when_the_line_is_absent():
    assert spl.met_failures("loading model\ndone 516 stations\n") == 0


def test_met_failures_sums_repeated_lines():
    # Arrange — 실행당 한 번 찍히지만 여러 번 찍혀도 합친다
    log = REAL_LINE + "\n" + REAL_LINE.replace("7건", "3건") + "\n"

    # Act / Assert
    assert spl.met_failures(log) == 10


def test_met_failures_ignores_other_numbers_and_similar_phrases():
    # Arrange
    log = "met 조회 실패 9건\n실시간 조회 실패 5건\n516 rows\n"

    # Act / Assert
    assert spl.met_failures(log) == 0


def test_render_summary_says_none_for_zero_and_counts_otherwise():
    # Act / Assert
    assert "없음" in spl.render_summary(0)
    seven = spl.render_summary(7)
    assert "7건" in seven and "상수 기상" in seven
    assert seven.startswith("### predict 실행 요약")


def test_main_writes_summary_and_warning_when_failures_exist(tmp_path, monkeypatch, capsys):
    # Arrange
    log = tmp_path / "predict.log"
    log.write_text("start\n" + REAL_LINE + "\n", encoding="utf-8")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    # Act
    code = spl.main([str(log)])

    # Assert
    out = capsys.readouterr().out
    assert code == 0
    assert "::warning title=Open-Meteo fallback::" in out
    assert "7건" in out
    assert "7건" in summary.read_text(encoding="utf-8")


def test_main_with_zero_failures_writes_summary_without_warning(tmp_path, monkeypatch, capsys):
    # Arrange
    log = tmp_path / "predict.log"
    log.write_text("start\nfinished\n", encoding="utf-8")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    # Act
    code = spl.main([str(log)])

    # Assert
    assert code == 0
    assert "::warning" not in capsys.readouterr().out
    assert "없음" in summary.read_text(encoding="utf-8")


def test_main_appends_to_an_existing_summary_file(tmp_path, monkeypatch):
    # Arrange — 다른 스텝이 쓴 요약을 덮어쓰면 안 된다
    log = tmp_path / "predict.log"
    log.write_text("finished\n", encoding="utf-8")
    summary = tmp_path / "summary.md"
    summary.write_text("earlier step\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    # Act
    spl.main([str(log)])

    # Assert
    text = summary.read_text(encoding="utf-8")
    assert text.startswith("earlier step\n")
    assert "predict 실행 요약" in text


def test_main_without_summary_env_only_prints(tmp_path, monkeypatch, capsys):
    # Arrange
    log = tmp_path / "predict.log"
    log.write_text("finished\n", encoding="utf-8")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    # Act
    code = spl.main([str(log)])

    # Assert
    assert code == 0
    assert "predict 실행 요약" in capsys.readouterr().out


@pytest.mark.parametrize("content", ["", "   \n\n"])
def test_main_flags_an_empty_log_instead_of_reporting_zero_failures(content, tmp_path, monkeypatch, capsys):
    # Arrange — 로그가 비었으면 "0건" 이 아니라 "알 수 없다" 여야 한다
    log = tmp_path / "predict.log"
    log.write_text(content, encoding="utf-8")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    # Act
    code = spl.main([str(log)])

    # Assert
    assert code == 0
    assert "predict log empty" in capsys.readouterr().out
    assert not summary.exists()


def test_main_flags_a_missing_log_file_and_still_exits_zero(tmp_path, capsys):
    # Act
    code = spl.main([str(tmp_path / "nope.log")])

    # Assert — advisory: 판단은 사람이 한다. 파일 경로는 찍지 않고 예외 종류만.
    out = capsys.readouterr().out
    assert code == 0
    assert "predict log unreadable" in out
    assert "FileNotFoundError" in out
    assert "nope.log" not in out


def test_main_tolerates_undecodable_bytes(tmp_path, capsys):
    # Arrange
    log = tmp_path / "predict.log"
    log.write_bytes(b"\xff\xfe broken\n" + REAL_LINE.encode("utf-8") + b"\n")

    # Act
    code = spl.main([str(log)])

    # Assert
    assert code == 0
    assert "7건" in capsys.readouterr().out
