"""publish_targets.py 시험 (AAA, 네트워크 0).

이 모듈이 지키는 것은 하나다 — **후보 모드는 운영 경로를 건드리지 못한다**. 모드를 바꾸는 한 줄(워크플로의
`PUBLISH_MODE`)이 발행 위치를 통째로 정하므로, 매핑 표가 어긋나면 운영 발행물을 덮어쓰는 사고가 난다.
"""
from __future__ import annotations

import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

import publish_targets as pt

ETL_DIR = Path(__file__).resolve().parents[1] / "etl"
STAMP = "2026101512"

_spec = importlib.util.spec_from_file_location("_hf_publish_for_ml_tests", ETL_DIR / "hf_publish.py")
assert _spec is not None and _spec.loader is not None
hf_publish = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hf_publish)

PRODUCTION_TABLE = [
    ("grid", "grid_latest.json", "aq-data/predictions/grid_latest.json"),
    ("grid", "grid_latest.json", f"aq-data/predictions/history/grid_{STAMP}.json"),
    ("dqss", "data_quality.json", "aq-data/data_quality.json"),
    ("dqss", "data_quality.json", f"aq-data/quality-history/quality_{STAMP}.json"),
    ("shadow", "shadow_stations.json", f"aq-data/quality-history/shadow_{STAMP}.json"),
]


def _tuples(mode: str) -> list[tuple[str, str, str]]:
    return [(t.group, t.src, t.dest) for t in pt.predictions_targets(mode, STAMP)]


# ── 후보 모드 ────────────────────────────────────────────────────────────────
def test_candidate_destinations_all_live_under_the_candidate_root():
    # Act
    dests = [dest for _g, _s, dest in _tuples("candidate")]

    # Assert
    assert dests, "후보 대상이 비어 있다"
    assert all(d.startswith("aq-data/_candidate/") for d in dests)


def test_candidate_never_writes_a_production_path():
    # Arrange
    production = {dest for _g, _s, dest in _tuples("production")}

    # Act
    candidate = {dest for _g, _s, dest in _tuples("candidate")}

    # Assert — 운영 경로와 겹치지 않는다(latest 포함)
    assert candidate.isdisjoint(production)


def test_candidate_has_no_latest_file_only_timestamped_snapshots():
    # Act
    dests = [dest for _g, _s, dest in _tuples("candidate")]

    # Assert — 후보에는 "최신본" 포인터가 없다. 이름이 `<접두>_<10자리>.json` 이라 기존 prune 정규식에 걸린다.
    assert not any("latest" in d for d in dests)
    assert all(re.fullmatch(r".+_\d{10}\.json", d) for d in dests)


def test_candidate_names_are_recognised_by_the_existing_prune_regex():
    # Arrange — 컷오버 뒤 후보를 비울 때 새 코드가 필요 없어야 한다 (`prune --keep-days 0`)
    dests = [dest for _g, _s, dest in _tuples("candidate")]
    far_future = datetime(2100, 1, 1, tzinfo=timezone.utc)

    # Act
    stale, skipped_bad_ts, skipped_backfill = hf_publish._stale_files(dests, "aq-data/_candidate/", far_future)

    # Assert
    assert sorted(stale) == sorted(dests)
    assert (skipped_bad_ts, skipped_backfill) == (0, 0)


def test_candidate_prune_regex_would_not_touch_production_paths():
    # Arrange
    production = [dest for _g, _s, dest in _tuples("production")]

    # Act
    stale, _, _ = hf_publish._stale_files(production, "aq-data/_candidate/", datetime(2100, 1, 1, tzinfo=timezone.utc))

    # Assert
    assert stale == []


def test_candidate_shadow_is_advisory_and_the_others_are_required():
    # Act
    required = {t.group: t.required for t in pt.predictions_targets("candidate", STAMP)}

    # Assert
    assert required == {"grid": True, "dqss": True, "shadow": False}


# ── 운영 모드 ────────────────────────────────────────────────────────────────
def test_production_table_matches_the_documented_mapping():
    # Act / Assert — 기존 운영 워크플로의 발행 경로를 그대로 옮긴 표
    assert _tuples("production") == PRODUCTION_TABLE


def test_production_prune_targets_are_thirty_day_history_folders_only():
    # Act
    prefixes = pt.prune_prefixes("production")

    # Assert — latest 파일이 있는 폴더는 대상이 아니다
    assert prefixes == [("aq-data/predictions/history", 30), ("aq-data/quality-history", 30)]


def test_candidate_has_no_automatic_prune():
    # Act / Assert
    assert pt.prune_prefixes("candidate") == []


@pytest.mark.parametrize("fn", [pt.predictions_targets, lambda mode, stamp: pt.prune_prefixes(mode)])
def test_unknown_mode_is_rejected(fn):
    # Act / Assert — 오타(`prod`)가 조용히 어느 한쪽 모드로 해석되면 안 된다
    with pytest.raises(ValueError, match="mode"):
        fn("prod", STAMP)


# ── 스탬프 ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("value", ["2026101512", "2026123123", "2024022900"])
def test_parse_stamp_accepts_real_hours(value):
    assert pt.parse_stamp(value) == value


@pytest.mark.parametrize("value", ["", "20261015", "202610151200", "2026-10-15", "2026131512", "2026023012", "2026101524", "abcdefghij"])
def test_parse_stamp_rejects_wrong_shapes_and_impossible_times(value):
    with pytest.raises(ValueError):
        pt.parse_stamp(value)


@pytest.mark.parametrize(
    ("grid", "expected"),
    [
        ({"predictions": [{"timestamp": "2026-10-15T12:00:00Z"}]}, "2026101512"),
        ({"predictions": [{"timestamp": "2026-10-15T12:00:00+00:00"}]}, "2026101512"),
        # 오프셋이 있으면 UTC 로 바꾼다 — 파일명 시각과 내부 timestamp 가 어긋나지 않게
        ({"predictions": [{"timestamp": "2026-10-15T12:00:00+09:00"}]}, "2026101503"),
        ({"predictions": [], "generated_at": "2026-10-15T23:41:07Z"}, "2026101523"),
        ({"generated_at": "2026-10-15T23:41:07Z"}, "2026101523"),
    ],
)
def test_stamp_from_grid_uses_first_prediction_timestamp_in_utc(grid, expected):
    assert pt.stamp_from_grid(grid) == expected


def test_stamp_from_grid_prefers_prediction_timestamp_over_run_time():
    # Arrange — 실행은 13:41 에 끝났지만 예측 슬롯은 12시다. 같은 실행의 grid·DQSS·shadow 는 12시 STAMP 로 짝지어야 한다.
    grid = {"generated_at": "2026-10-15T13:41:07Z", "predictions": [{"timestamp": "2026-10-15T12:00:00Z"}]}

    # Act / Assert
    assert pt.stamp_from_grid(grid) == "2026101512"


@pytest.mark.parametrize("grid", [{}, {"predictions": []}, {"predictions": [{"timestamp": ""}]}, {"predictions": [{}]}])
def test_stamp_from_grid_without_any_time_fails(grid):
    with pytest.raises((ValueError, KeyError)):
        pt.stamp_from_grid(grid)


# ── CLI ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def artifact_dir(tmp_path: Path) -> Path:
    (tmp_path / "grid_latest.json").write_text(json.dumps({"predictions": [{"timestamp": "2026-10-15T12:00:00Z"}]}), encoding="utf-8")
    (tmp_path / "data_quality.json").write_text("{}", encoding="utf-8")
    (tmp_path / "shadow_stations.json").write_text("{}", encoding="utf-8")
    return tmp_path


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = pt.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.mark.parametrize(
    ("mode", "group", "dests"),
    [
        ("candidate", "grid", [f"aq-data/_candidate/predictions/history/grid_{STAMP}.json"]),
        ("candidate", "dqss", [f"aq-data/_candidate/quality-history/quality_{STAMP}.json"]),
        ("candidate", "shadow", [f"aq-data/_candidate/quality-history/shadow_{STAMP}.json"]),
        ("production", "grid", ["aq-data/predictions/grid_latest.json", f"aq-data/predictions/history/grid_{STAMP}.json"]),
        ("production", "dqss", ["aq-data/data_quality.json", f"aq-data/quality-history/quality_{STAMP}.json"]),
    ],
)
def test_cli_maps_prints_src_equals_dest_lines_for_the_group(mode, group, dests, artifact_dir, capsys):
    # Act
    code, out, _ = _run(capsys, "maps", "--mode", mode, "--dir", str(artifact_dir), "--group", group)

    # Assert — hf_publish batch 의 `--map SRC=DEST` 형식 그대로(한 줄에 하나)
    lines = out.splitlines()
    assert code == 0
    assert [line.split("=", 1)[1] for line in lines] == dests
    assert all(Path(line.split("=", 1)[0]).is_file() for line in lines)


def test_cli_maps_output_parses_with_the_publisher_map_parser(artifact_dir, capsys):
    # Arrange
    _, out, _ = _run(capsys, "maps", "--mode", "candidate", "--dir", str(artifact_dir), "--group", "grid")

    # Act — 발행기가 실제로 쓰는 파서로 되읽는다
    parsed = hf_publish._parse_map(out.splitlines())

    # Assert
    assert parsed == [(artifact_dir / "grid_latest.json", f"aq-data/_candidate/predictions/history/grid_{STAMP}.json")]


def test_cli_maps_with_missing_advisory_shadow_prints_nothing_and_succeeds(artifact_dir, capsys):
    # Arrange
    (artifact_dir / "shadow_stations.json").unlink()

    # Act
    code, out, _ = _run(capsys, "maps", "--mode", "candidate", "--dir", str(artifact_dir), "--group", "shadow")

    # Assert — 호출부가 빈 출력으로 "섀도우 없음" 을 판단한다
    assert (code, out) == (0, "")


def test_cli_maps_with_missing_required_dqss_file_fails_naming_it(artifact_dir, capsys):
    # Arrange
    (artifact_dir / "data_quality.json").unlink()

    # Act
    code, out, err = _run(capsys, "maps", "--mode", "candidate", "--dir", str(artifact_dir), "--group", "dqss")

    # Assert — 필수 산출물이 없으면 빈 출력으로 넘어가지 않고 실패한다(빈 --map 으로 발행기가 도는 것을 막는다)
    assert code == 1
    assert out == ""
    assert "data_quality.json" in err


def test_cli_maps_without_grid_cannot_derive_a_stamp_and_fails(artifact_dir, capsys):
    # Arrange
    (artifact_dir / "grid_latest.json").unlink()

    # Act
    code, out, err = _run(capsys, "maps", "--mode", "candidate", "--dir", str(artifact_dir), "--group", "dqss")

    # Assert
    assert code == 1
    assert out == ""
    assert "publish_targets:" in err


@pytest.mark.parametrize(
    ("grid_text", "error_class"),
    [
        ("{not json", "JSONDecodeError"),
        ("[]", "AttributeError"),  # grid 가 객체가 아니다
        ("null", "AttributeError"),
        ('{"predictions": [1]}', "TypeError"),  # 행이 객체가 아니다
        ('{"predictions": [null]}', "TypeError"),
        ('{"predictions": [{}]}', "KeyError"),
        ('{"predictions": [{"timestamp": "not-a-time"}]}', "ValueError"),
    ],
)
def test_cli_maps_with_a_broken_grid_fails_with_class_and_message_not_a_traceback(grid_text, error_class, artifact_dir, capsys):
    # Arrange
    (artifact_dir / "grid_latest.json").write_text(grid_text, encoding="utf-8")

    # Act
    code, out, err = _run(capsys, "maps", "--mode", "candidate", "--dir", str(artifact_dir), "--group", "grid")

    # Assert — 빈 출력으로 발행기가 도는 것을 막는다(종료 코드 1). 메시지는 한 줄이다.
    assert (code, out) == (1, "")
    assert err.startswith(f"publish_targets: {error_class}")
    assert "Traceback" not in err and len(err.splitlines()) == 1


def test_cli_prune_prints_prefix_and_days_lines_for_production_only(capsys):
    # Act
    code_p, out_p, _ = _run(capsys, "prune", "--mode", "production")
    code_c, out_c, _ = _run(capsys, "prune", "--mode", "candidate")

    # Assert
    assert code_p == 0
    assert out_p.splitlines() == ["aq-data/predictions/history 30", "aq-data/quality-history 30"]
    assert (code_c, out_c) == (0, "")


def test_cli_rejects_unknown_mode(capsys):
    # Act / Assert — argparse choices 가 막는다 (종료 코드 2)
    with pytest.raises(SystemExit) as exc:
        pt.main(["prune", "--mode", "prod"])
    assert exc.value.code == 2
