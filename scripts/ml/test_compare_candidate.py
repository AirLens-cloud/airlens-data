"""compare_candidate.py 시험 (AAA, 네트워크 0).

판정 기준은 이 스크립트의 존재 이유다 — 컷오버 판단을 사람의 눈대중이 아니라 같은 기준으로 하게 한다.
그래서 기준의 **경계값**(0.1·1.0·99%·95%)을 한 칸 안쪽/바깥쪽으로 나눠 고정한다. 입력은 모두 손으로 만든
합성 payload 이고, 판정 함수는 순수 함수라 네트워크 없이 돈다. HTTP 원천은 가짜 응답으로만 시험한다.
"""
from __future__ import annotations

import copy
import http.client
import io
import json
import re
import ssl
import urllib.error
from pathlib import Path

import pytest

import compare_candidate as cc
import publish_targets as pt

STAMP = "2026101512"
SHA = "a" * 64


def row(i: int, **over: object) -> dict:
    base = {
        "name": f"station-{i:04d}",
        "lat": 30.0 + i * 0.01,
        "lon": 120.0 + i * 0.01,
        "timestamp": "2026-10-15T12:00:00Z",
        "predicted_p10": 10.0 + i % 7,
        "predicted_p50": 20.0 + i % 7,
        "predicted_p90": 30.0 + i % 7,
        "observed_pm25": 15.0 + i % 5,
        "input_updated_at": "2026-10-15T11:00:00Z",
        "confidence_grade": "B",
        "aod_source": "modis",
        "feature_regime": "default",
    }
    base.update(over)
    return base


def grid(n: int = 200, **top: object) -> dict:
    payload = {
        "schema_version": "grid_latest.v1",
        "generated_at": "2026-10-15T12:41:00Z",
        "model_version": "v2.0",
        "model_sha": SHA,
        "model_available": True,
        "observation_slot": "2026-10-15T12:00:00Z",
        "count": n,
        "predictions": [row(i) for i in range(n)],
    }
    payload.update(top)
    return payload


def quality(n: int = 100, **top: object) -> dict:
    payload = {
        "schema_version": "data_quality.v1",
        "stations": [{"station_id": f"s{i:03d}", "final_score": 50.0 + i % 40, "badge": "good" if i % 2 else "fair"} for i in range(n)],
    }
    payload.update(top)
    return payload


def tweak(payload: dict, indices: range | list[int], **fields: object) -> dict:
    """predictions 의 일부 행 필드를 바꾼 사본."""
    out = copy.deepcopy(payload)
    for i in indices:
        out["predictions"][i].update(fields)
    return out


def shift(payload: dict, indices: range | list[int], field: str, delta: float) -> dict:
    out = copy.deepcopy(payload)
    for i in indices:
        out["predictions"][i][field] = out["predictions"][i][field] + delta
    return out


def ok(_payload: dict) -> None:
    return None


def classify(prod, cand, prod_q=None, cand_q=None, validate=ok) -> cc.PairResult:
    return cc.classify_pair(STAMP, prod, cand, prod_q, cand_q, validate)


def check(result: cc.PairResult, name_part: str) -> cc.Check:
    found = [c for c in result.checks if name_part in c.name]
    assert found, f"검사 {name_part!r} 가 없다: {[c.name for c in result.checks]}"
    return found[0]


# ── 엄격 짝: grid ────────────────────────────────────────────────────────────
def test_identical_payloads_pass_as_a_strict_pair():
    # Act
    result = classify(grid(), grid())

    # Assert
    assert (result.verdict, result.strict, result.reasons) == (cc.PASS, True, [])
    assert result.max_delta == {"predicted_p10": 0.0, "predicted_p50": 0.0, "predicted_p90": 0.0}
    assert all(c.ok is not False for c in result.checks)


def test_quantile_difference_of_exactly_one_tenth_everywhere_passes():
    # Arrange — 0.1 단위 반올림 값이라 한 칸 차이는 허용한다(부동소수 오차 포함: 20.0 → 20.1 의 차는 0.10000000000000142)
    cand = shift(grid(), range(200), "predicted_p50", 0.1)

    # Act
    result = classify(grid(), cand)

    # Assert
    assert result.verdict == cc.PASS


def test_two_of_two_hundred_rows_over_tolerance_still_passes_at_the_99_percent_boundary():
    # Arrange — 198/200 = 99.0% (경계 포함)
    cand = shift(grid(), [0, 1], "predicted_p50", 0.2)

    # Act
    result = classify(grid(), cand)

    # Assert
    assert result.verdict == cc.PASS
    assert result.max_delta["predicted_p50"] == 0.2


def test_three_of_two_hundred_rows_over_tolerance_fails_the_99_percent_rule():
    # Arrange — 197/200 = 98.5%
    cand = shift(grid(), [0, 1, 2], "predicted_p50", 0.2)

    # Act
    result = classify(grid(), cand)

    # Assert
    assert result.verdict == cc.FAIL
    assert check(result, "p10/p50/p90").ok is False
    assert any("p10/p50/p90" in r for r in result.reasons)
    assert len(result.worst_rows) == 3


def test_one_row_off_by_exactly_one_microgram_passes_but_more_than_one_fails():
    # Arrange — 최대 |Δ| ≤ 1.0 (한 행이 99% 규칙 안에 있어도 크게 어긋나면 막는다)
    at_limit = shift(grid(), [5], "predicted_p90", 1.0)
    beyond = shift(grid(), [5], "predicted_p90", 1.5)

    # Act
    passed, failed = classify(grid(), at_limit), classify(grid(), beyond)

    # Assert
    assert passed.verdict == cc.PASS
    assert failed.verdict == cc.FAIL
    assert check(failed, "최대").ok is False
    assert failed.max_delta["predicted_p90"] == 1.5


def test_non_numeric_quantile_fails_even_when_few_rows_are_affected():
    # Arrange
    cand = tweak(grid(), [3], predicted_p10="n/a")

    # Act
    result = classify(grid(), cand)

    # Assert
    assert result.verdict == cc.FAIL
    assert "숫자 아님" in check(result, "최대").detail


@pytest.mark.parametrize(("value", "expected"), [
    (3, 3.0), (0.1, 0.1), (-2.5, -2.5), (0, 0.0),
    (float("nan"), None), (float("inf"), None), (float("-inf"), None),
    (True, None), ("1", None), (None, None),
])
def test_num_accepts_finite_numbers_only(value, expected):
    # Act / Assert
    assert cc._num(value) == expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("side", ["prod", "cand"])
def test_non_finite_quantile_fails_even_when_a_single_row_is_affected(bad, side):
    # Arrange — NaN 과의 비교는 늘 False 라 `|Δ| > 허용` 에 안 걸려 조용히 통과하던 구멍. 한 행(0.5%)이어도 막는다.
    broken = tweak(grid(), [3], predicted_p50=bad)
    prod, cand = (broken, grid()) if side == "prod" else (grid(), broken)

    # Act
    result = classify(prod, cand)

    # Assert
    assert result.verdict == cc.FAIL
    assert check(result, "p10/p50/p90").detail.startswith("199/200")  # 99% 규칙만으로는 통과하는 비율
    assert check(result, "최대").ok is False
    assert "숫자 아님 1" in check(result, "최대").detail
    worst = result.worst_rows[0]
    assert (worst["name"], worst["field"], worst["abs_delta"]) == ("station-0003", "predicted_p50", None)
    assert worst[side] == str(bad)  # 표에는 문자열로 나간다


def test_missing_row_in_candidate_fails_on_key_set():
    # Arrange
    cand = grid()
    cand["predictions"].pop()
    cand["count"] -= 1

    # Act
    result = classify(grid(), cand)

    # Assert
    assert result.verdict == cc.FAIL
    assert check(result, "행 키 집합").ok is False
    assert "운영만 1" in check(result, "행 키 집합").detail


def test_count_mismatch_fails_even_if_rows_match():
    # Arrange
    cand = grid()
    cand["count"] = 199

    # Act
    result = classify(grid(), cand)

    # Assert
    assert result.verdict == cc.FAIL
    assert check(result, "count").ok is False


def test_row_keys_of_mixed_types_do_not_break_the_key_set_check():
    # Arrange — 후보에만 있는 행의 이름이 None 과 문자열이라 키를 정렬하면 TypeError 였다
    cand = grid()
    cand["predictions"] += [row(900, name=None), row(901)]
    cand["count"] += 2

    # Act
    result = classify(grid(), cand)

    # Assert
    assert result.verdict == cc.FAIL
    assert "후보만 2" in check(result, "행 키 집합").detail


def test_duplicate_row_keys_fail():
    # Arrange
    cand = grid()
    cand["predictions"].append(copy.deepcopy(cand["predictions"][0]))
    cand["count"] += 1

    # Act
    result = classify(grid(), cand)

    # Assert
    assert result.verdict == cc.FAIL
    assert check(result, "키 중복").ok is False


def test_rows_with_different_inputs_are_excluded_from_quantile_comparison():
    # Arrange — 입력(observed_pm25)이 다른 행은 출력이 달라도 정상이다. 한 행(0.5%)만 입력이 다르면 99% 규칙 안이다.
    cand = tweak(grid(), [0], observed_pm25=99.0, predicted_p50=999.0)

    # Act
    result = classify(grid(), cand)

    # Assert
    assert result.verdict == cc.PASS
    assert check(result, "입력 같은 행").detail.startswith("199/200")
    assert result.max_delta["predicted_p50"] == 0.0


def test_input_updated_at_difference_also_counts_as_different_input():
    # Arrange
    cand = tweak(grid(), [0, 1, 2], input_updated_at="2026-10-15T10:00:00Z")

    # Act
    result = classify(grid(), cand)

    # Assert — 197/200 = 98.5% < 99%
    assert result.verdict == cc.FAIL
    assert check(result, "입력 같은 행").ok is False


@pytest.mark.parametrize("field", ["confidence_grade", "aod_source", "feature_regime"])
def test_categorical_fields_must_match_on_99_percent_of_rows(field):
    # Arrange
    one_row = tweak(grid(), [0], **{field: "changed"})  # 199/200 = 99.5%
    three_rows = tweak(grid(), [0, 1, 2], **{field: "changed"})  # 197/200 = 98.5%

    # Act / Assert
    assert classify(grid(), one_row).verdict == cc.PASS
    failed = classify(grid(), three_rows)
    assert failed.verdict == cc.FAIL
    assert check(failed, field).ok is False


def test_candidate_contract_violation_fails_a_strict_pair():
    # Arrange
    def reject(_payload):
        return "predictions[0].lat: required"

    # Act
    result = classify(grid(), grid(), validate=reject)

    # Assert
    assert result.verdict == cc.FAIL
    assert check(result, "후보 계약").detail == "predictions[0].lat: required"


# ── 엄격 짝이 아닌 쌍 / 짝 없음 ─────────────────────────────────────────────
@pytest.mark.parametrize(
    ("field", "value"),
    [("model_sha", "b" * 64), ("model_version", "v2.1"), ("observation_slot", "2026-10-15T11:00:00Z")],
)
def test_different_identity_field_makes_the_pair_input_diff_not_strict(field, value):
    # Arrange
    cand = grid(**{field: value})

    # Act
    result = classify(grid(), cand)

    # Assert — 입력이 달라 엄격 비교는 하지 않는다. 값 차이가 커도 FAIL 이 아니다.
    assert (result.verdict, result.strict) == (cc.INPUT_DIFF, False)
    assert field in result.reasons[0]
    assert check(result, "후보 계약").ok is True
    assert check(result, "키 겹침").detail == "100.0%"


def test_input_diff_pair_ignores_value_differences_but_reports_overlap():
    # Arrange — 키가 절반만 겹친다. 보고만 하고 판정에는 쓰지 않는다.
    cand = grid(model_sha="b" * 64)
    cand["predictions"] = cand["predictions"][:100] + [row(1000 + i) for i in range(100)]

    # Act
    result = classify(grid(), cand)

    # Assert
    assert result.verdict == cc.INPUT_DIFF
    overlap = check(result, "키 겹침")
    assert overlap.ok is None
    assert "95.0% 미만" in overlap.detail


def test_input_diff_pair_with_contract_violation_is_a_fail():
    # Arrange
    def reject(_payload):
        return "count: minimum"

    # Act
    result = classify(grid(), grid(model_version="v9"), validate=reject)

    # Assert
    assert result.verdict == cc.FAIL
    assert result.strict is False
    assert "후보가 계약을 어겼다" in result.reasons[0]


@pytest.mark.parametrize("cand", [
    grid(model_available=False),
    grid(model_sha=None),
    grid(model_available=False, model_sha=None),
    {k: v for k, v in grid().items() if k != "model_available"},
], ids=["unavailable", "no-sha", "both", "key-missing"])
def test_candidate_that_ran_the_fallback_path_fails_before_any_strict_comparison(cand):
    # Arrange — 가중치를 못 써서 폴백으로 돈 후보는 값이 운영과 같든 달든 컷오버 근거가 아니다

    # Act
    result = classify(grid(), cand)

    # Assert
    assert (result.verdict, result.strict) == (cc.FAIL, False)
    assert result.reasons == ["후보가 폴백 경로로 돌았다"]
    assert check(result, "후보 모델 사용").ok is False
    assert SHA not in check(result, "후보 모델 사용").detail  # 있고 없음만 적는다


def test_two_null_model_shas_are_not_a_strict_pair():
    # Arrange — null 은 "알 수 없음" 이다. null == null 을 같은 모델로 읽으면 폴백끼리 PASS 가 난다.
    both_null = (grid(1, model_sha=None), grid(1, model_sha=None))

    # Act / Assert
    assert cc.strict_mismatches(*both_null) == ["model_sha"]
    assert cc.strict_mismatches(grid(1), grid(1)) == []
    assert cc.strict_mismatches(grid(1, model_sha=None), grid(1)) == ["model_sha"]
    assert cc.strict_mismatches(grid(1), grid(1, model_sha=None)) == ["model_sha"]


def test_production_without_a_model_sha_is_input_diff_when_the_candidate_is_healthy():
    # Arrange — 운영이 구 아티팩트(model_sha null)면 어느 가중치와 비교하는지 알 수 없어 엄격 비교하지 않는다.
    prod = grid(model_sha=None)

    # Act
    result = classify(prod, grid())

    # Assert
    assert (result.verdict, result.strict) == (cc.INPUT_DIFF, False)
    assert "model_sha" in result.reasons[0]


@pytest.mark.parametrize(("prod", "cand", "side"), [(None, grid(), "운영"), (grid(), None, "후보"), (None, None, "운영")])
def test_missing_grid_on_either_side_is_unpaired(prod, cand, side):
    # Act
    result = classify(prod, cand)

    # Assert
    assert (result.verdict, result.strict) == (cc.UNPAIRED, False)
    assert side in result.reasons[0]


# ── 엄격 짝: DQSS ────────────────────────────────────────────────────────────
def test_identical_dqss_payloads_pass():
    # Act
    result = classify(grid(), grid(), quality(), quality())

    # Assert
    assert result.verdict == cc.PASS
    assert check(result, "DQSS |Δfinal_score|").ok is True
    assert check(result, "DQSS badge").ok is True


def test_dqss_score_difference_of_one_passes_and_more_than_one_counts_against():
    # Arrange — 5% 지점이 허용(≤1) 밖이면 95% 경계 포함으로 통과, 6% 면 실패
    def with_scores(off: int) -> dict:
        q = quality()
        for i in range(off):
            q["stations"][i]["final_score"] += 1.5
        for i in range(off, 100):
            q["stations"][i]["final_score"] += 1.0
        return q

    # Act
    five, six = classify(grid(), grid(), quality(), with_scores(5)), classify(grid(), grid(), quality(), with_scores(6))

    # Assert
    assert five.verdict == cc.PASS
    assert six.verdict == cc.FAIL
    assert check(six, "DQSS |Δfinal_score|").detail.startswith("94/100")


def test_dqss_badge_mismatch_follows_the_95_percent_rule():
    # Arrange
    def with_badges(off: int) -> dict:
        q = quality()
        for i in range(off):
            q["stations"][i]["badge"] = "poor"
        return q

    # Act / Assert
    assert classify(grid(), grid(), quality(), with_badges(5)).verdict == cc.PASS
    failed = classify(grid(), grid(), quality(), with_badges(6))
    assert failed.verdict == cc.FAIL
    assert check(failed, "DQSS badge").ok is False


def test_dqss_station_set_difference_fails():
    # Arrange
    cand_q = quality()
    cand_q["stations"].pop()

    # Act
    result = classify(grid(), grid(), quality(), cand_q)

    # Assert
    assert result.verdict == cc.FAIL
    assert check(result, "station_id 집합").ok is False


def test_dqss_contract_name_difference_fails_for_one_side_withheld():
    # Arrange — 한쪽만 정보량 게이트에 걸려 withheld 로 나온 경우는 같은 입력에서 달라진 것이므로 잡는다
    withheld = {"schema_version": "data_quality_withheld.v1", "stations": []}

    # Act
    result = classify(grid(), grid(), quality(), withheld)

    # Assert
    assert result.verdict == cc.FAIL
    assert check(result, "DQSS 계약 이름").ok is False


def test_both_sides_withheld_compare_only_the_contract_name_and_station_set():
    # Arrange
    withheld = {"schema_version": "data_quality_withheld.v1", "stations": []}

    # Act
    result = classify(grid(), grid(), withheld, copy.deepcopy(withheld))

    # Assert
    assert result.verdict == cc.PASS
    assert not [c for c in result.checks if "DQSS |Δ" in c.name or "DQSS badge" in c.name]


def test_dqss_none_scores_match_only_when_both_are_none():
    # Arrange
    both_none, one_none = quality(), quality()
    prod_q = quality()
    prod_q["stations"][0]["final_score"] = None
    both_none["stations"][0]["final_score"] = None
    one_none["stations"][0]["final_score"] = 50.0

    # Act
    same = cc.compare_quality(prod_q, both_none)
    different = cc.compare_quality(prod_q, one_none)

    # Assert
    assert [c for c in same if "final_score" in c.name][0].detail.startswith("100/100")
    assert [c for c in different if "final_score" in c.name][0].detail.startswith("99/100")


@pytest.mark.parametrize(("prod_score", "cand_score"), [
    (float("nan"), 50.0), (50.0, float("nan")), (float("nan"), float("nan")),
    (float("inf"), float("inf")), (float("inf"), 50.0),
], ids=["prod-nan", "cand-nan", "both-nan", "both-inf", "prod-inf"])
def test_dqss_non_finite_final_score_is_never_a_hit(prod_score, cand_score):
    # Arrange — 한 측정소(1%)만 어긋나므로 99/100 이 정확한 값이다
    prod_q, cand_q = quality(), quality()
    prod_q["stations"][0]["final_score"] = prod_score
    cand_q["stations"][0]["final_score"] = cand_score

    # Act
    checks = cc.compare_quality(prod_q, cand_q)

    # Assert
    assert [c for c in checks if "final_score" in c.name][0].detail.startswith("99/100")


def test_candidate_without_dqss_file_fails_but_production_without_it_is_only_reported():
    # Act
    cand_missing = classify(grid(), grid(), quality(), None)
    prod_missing = classify(grid(), grid(), None, quality())
    both_missing = classify(grid(), grid(), None, None)

    # Assert
    assert cand_missing.verdict == cc.FAIL
    assert check(cand_missing, "DQSS 파일").ok is False
    assert prod_missing.verdict == cc.PASS
    assert check(prod_missing, "DQSS 파일").ok is None
    assert both_missing.verdict == cc.PASS
    assert "양쪽 없음" in check(both_missing, "DQSS 파일").detail


# ── 요약·종료 코드 ───────────────────────────────────────────────────────────
def _result(verdict: str, strict: bool = True) -> cc.PairResult:
    return cc.PairResult("2026101512", verdict, strict)


@pytest.mark.parametrize(
    ("results", "min_strict", "code"),
    [
        ([_result(cc.PASS)] * 3, 3, 0),
        ([_result(cc.PASS)] * 2, 3, 2),  # 엄격 PASS 가 모자라면 판단 보류
        ([_result(cc.PASS)] * 5 + [_result(cc.FAIL)], 3, 1),  # FAIL 은 다른 조건보다 앞선다
        ([_result(cc.FAIL)], 3, 1),
        ([_result(cc.INPUT_DIFF, False)] * 10, 3, 2),  # INPUT-DIFF 는 엄격 PASS 에 안 센다
        ([_result(cc.UNPAIRED, False)] * 10, 1, 2),
        ([_result(cc.PASS)] + [_result(cc.INPUT_DIFF, False)] * 2, 1, 0),
        ([], 3, 2),
    ],
    ids=["enough", "too-few", "fail-wins", "fail-only", "input-diff-only", "unpaired-only", "mixed-ok", "empty"],
)
def test_summarize_exit_code(results, min_strict, code):
    # Act
    summary = cc.summarize(results, prod_only=0, min_strict=min_strict)

    # Assert
    assert summary["exit_code"] == code


def test_summarize_counts_and_max_delta_use_strict_pairs_only():
    # Arrange
    strict = cc.PairResult(STAMP, cc.PASS, True, max_delta={"predicted_p10": 0.1, "predicted_p50": 0.3, "predicted_p90": 0.2})
    other = cc.PairResult(STAMP, cc.INPUT_DIFF, False, max_delta={"predicted_p10": 9.0})

    # Act
    summary = cc.summarize([strict, other], prod_only=4, min_strict=1)

    # Assert
    assert summary["counts"] == {cc.PASS: 1, cc.FAIL: 0, cc.INPUT_DIFF: 1, cc.UNPAIRED: 0}
    assert summary["max_delta"] == {"predicted_p10": 0.1, "predicted_p50": 0.3, "predicted_p90": 0.2}
    assert summary["prod_only_stamps"] == 4


@pytest.mark.parametrize(
    ("value", "expected"),
    [("2026101512", "2026101512"), ("20261015", "2026101500"), ("2026-10-15", "2026101500")],
)
def test_parse_since_accepts_three_forms(value, expected):
    assert cc.parse_since(value) == expected


@pytest.mark.parametrize("value", ["", "2026", "2026-10", "202610151", "abcdefgh", "2026-10-15T12"])
def test_parse_since_rejects_other_forms(value):
    with pytest.raises(ValueError, match="--since"):
        cc.parse_since(value)


def test_render_markdown_has_summary_table_and_details_for_strict_and_failed_pairs_only():
    # Arrange
    passed = classify(grid(), grid())
    failed = classify(grid(), shift(grid(), [0, 1, 2], "predicted_p50", 0.2))
    input_diff = classify(grid(), grid(model_sha="b" * 64))
    results = [passed, failed, input_diff]
    summary = cc.summarize(results, prod_only=2, min_strict=3)

    # Act
    text = cc.render_markdown(results, summary)

    # Assert
    assert "## 후보 ↔ 운영 비교 (predictions)" in text
    assert "PASS 1 · FAIL 1 · INPUT-DIFF 1 · UNPAIRED 0" in text
    assert "운영 STAMP 중 후보 없음 2" in text
    assert "| 검사 | 값 | 결과 |" in text
    assert "실패 행 상위 3" in text
    assert text.count("### ") == 2  # PASS 와 FAIL 의 상세만, INPUT-DIFF 는 표의 사유 한 줄


# ── 입력 원천: 로컬 디렉터리 ─────────────────────────────────────────────────
def write_json(root: Path, rel: str, payload: dict) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_local_source_lists_only_well_formed_stamps_sorted(tmp_path):
    # Arrange
    for name in ["grid_2026101512.json", "grid_2026101509.json", "grid_latest.json", "grid_2026101512.json.gz",
                 "grid_20261015.json", "quality_2026101512.json", "xgrid_2026101512.json"]:
        write_json(tmp_path, f"{cc.CAND_GRID_DIR}/{name}", {})

    # Act
    stamps = cc.LocalSource(tmp_path).list_stamps(cc.CAND_GRID_DIR, "grid")

    # Assert
    assert stamps == ["2026101509", "2026101512"]


def test_local_source_missing_directory_is_empty_and_missing_file_is_none(tmp_path):
    # Act
    source = cc.LocalSource(tmp_path)

    # Assert
    assert source.list_stamps(cc.CAND_GRID_DIR, "grid") == []
    assert source.read_json("aq-data/nope.json") is None


def test_local_source_unreadable_json_raises_source_error_with_relative_path_only(tmp_path):
    # Arrange
    path = tmp_path / "aq-data" / "bad.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    # Act / Assert
    with pytest.raises(cc.SourceError) as exc:
        cc.LocalSource(tmp_path).read_json("aq-data/bad.json")
    assert "aq-data/bad.json" in str(exc.value)
    assert str(tmp_path) not in str(exc.value)


@pytest.mark.parametrize("body", ["[]", '"text"', "1", "null"])
def test_local_source_non_object_json_raises_source_error_instead_of_reading_as_missing(tmp_path, body):
    # Arrange — `null` 을 None(없음)으로 돌려주면 UNPAIRED 로 가려지고, 배열은 판정 단계에서 AttributeError 가 난다
    path = tmp_path / "aq-data" / "odd.json"
    path.parent.mkdir(parents=True)
    path.write_text(body, encoding="utf-8")

    # Act / Assert
    with pytest.raises(cc.SourceError, match="JSON 객체 아님"):
        cc.LocalSource(tmp_path).read_json("aq-data/odd.json")


# ── run(): 짝짓기 ────────────────────────────────────────────────────────────
class MemorySource:
    """경로 → payload 사전. 읽은 경로를 기록한다."""

    def __init__(self, files: dict[str, dict]) -> None:
        self.files = files
        self.reads: list[str] = []

    def list_stamps(self, rel_dir: str, prefix: str) -> list[str]:
        out = []
        for path in self.files:
            head, _, name = path.rpartition("/")
            if head == rel_dir and name.startswith(prefix + "_"):
                out.append(name.removeprefix(prefix + "_").removesuffix(".json"))
        return sorted(out)

    def read_json(self, rel_path: str) -> dict | None:
        self.reads.append(rel_path)
        return self.files.get(rel_path)


def tree(stamps_prod: list[str], stamps_cand: list[str], *, cand_quality: bool = True) -> dict[str, dict]:
    files: dict[str, dict] = {}
    for s in stamps_prod:
        files[f"{cc.PROD_GRID_DIR}/grid_{s}.json"] = grid()
        files[f"{cc.PROD_QUALITY_DIR}/quality_{s}.json"] = quality()
    for s in stamps_cand:
        files[f"{cc.CAND_GRID_DIR}/grid_{s}.json"] = grid()
        if cand_quality:
            files[f"{cc.CAND_QUALITY_DIR}/quality_{s}.json"] = quality()
    return files


def test_run_pairs_candidate_stamps_with_the_same_stamp_in_production():
    # Arrange
    source = MemorySource(tree(["2026101509", "2026101512", "2026101515"], ["2026101512", "2026101515"]))

    # Act
    results, summary = cc.run(source, ok, since=None, min_strict=1)

    # Assert
    assert [r.stamp for r in results] == ["2026101512", "2026101515"]
    assert [r.verdict for r in results] == [cc.PASS, cc.PASS]
    assert summary["exit_code"] == 0


def test_run_reports_production_only_stamps_inside_the_candidate_window():
    # Arrange — 후보는 12·18시만 돌았고, 운영은 09~21시 3시간마다
    source = MemorySource(tree(["2026101509", "2026101512", "2026101515", "2026101518", "2026101521"], ["2026101512", "2026101518"]))

    # Act
    _, summary = cc.run(source, ok, since=None, min_strict=1)

    # Assert — 첫 후보 STAMP(12시) 이후의 운영-only: 15시, 21시 (09시는 후보 시작 전이라 세지 않는다)
    assert summary["prod_only_stamps"] == 2


def test_run_marks_candidate_without_production_counterpart_unpaired():
    # Arrange
    source = MemorySource(tree(["2026101512"], ["2026101512", "2026101515"]))

    # Act
    results, summary = cc.run(source, ok, since=None, min_strict=1)

    # Assert
    assert [(r.stamp, r.verdict) for r in results] == [("2026101512", cc.PASS), ("2026101515", cc.UNPAIRED)]
    assert summary["exit_code"] == 0


def test_run_filters_candidate_stamps_by_since():
    # Arrange
    source = MemorySource(tree(["2026101509", "2026101512", "2026101515"], ["2026101509", "2026101512", "2026101515"]))

    # Act
    results, _ = cc.run(source, ok, since="2026101512", min_strict=1)

    # Assert
    assert [r.stamp for r in results] == ["2026101512", "2026101515"]


def test_run_reads_quality_files_only_for_strict_pairs():
    # Arrange — 두 번째 STAMP 는 model_sha 가 달라 엄격 짝이 아니다
    files = tree(["2026101512", "2026101515"], ["2026101512", "2026101515"])
    files[f"{cc.CAND_GRID_DIR}/grid_2026101515.json"] = grid(model_sha="c" * 64)
    source = MemorySource(files)

    # Act
    results, _ = cc.run(source, ok, since=None, min_strict=1)

    # Assert
    quality_reads = [p for p in source.reads if "quality-history" in p]
    assert [r.verdict for r in results] == [cc.PASS, cc.INPUT_DIFF]
    assert quality_reads == [f"{cc.PROD_QUALITY_DIR}/quality_2026101512.json", f"{cc.CAND_QUALITY_DIR}/quality_2026101512.json"]


def test_run_without_any_candidate_is_insufficient_not_a_pass():
    # Arrange
    source = MemorySource(tree(["2026101512"], []))

    # Act
    results, summary = cc.run(source, ok, since=None, min_strict=3)

    # Assert
    assert results == []
    assert summary["exit_code"] == 2


# ── 입력 원천: 공개 저장소 (가짜 응답) ───────────────────────────────────────
class FakeResponse:
    def __init__(self, body: bytes, link: str = "", read_error: BaseException | None = None) -> None:
        self._body = body
        self._read_error = read_error  # 연결은 됐는데 본문을 읽다 끊기는 경우
        self.headers = {"Link": link} if link else {}

    def read(self) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def http_error(code: int, url: str = "https://example.invalid/secret/path") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "msg", {}, io.BytesIO(b"secret body"))


def scripted_source(monkeypatch: pytest.MonkeyPatch, outcomes: dict[str, list[object]]) -> tuple[cc.HttpSource, list[str], list[float]]:
    """URL 끝부분으로 고른 결과 목록을 차례로 돌려주는 HttpSource. (source, 요청 URL 기록, sleep 기록)."""
    calls: list[str] = []
    sleeps: list[float] = []

    def fake_open(self, url: str):
        calls.append(url)
        for suffix, queue in outcomes.items():
            if url.endswith(suffix):
                seen = sum(1 for c in calls if c.endswith(suffix))
                outcome = queue[min(seen, len(queue)) - 1]
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome
        raise AssertionError(f"예상 밖 URL: {url}")

    monkeypatch.setattr(cc.HttpSource, "_open", fake_open)
    return cc.HttpSource(sleep=sleeps.append), calls, sleeps


def test_http_read_json_returns_payload_and_none_for_404(monkeypatch):
    # Arrange
    source, calls, _ = scripted_source(monkeypatch, {"a.json": [FakeResponse(b'{"x": 1}')], "missing.json": [http_error(404)]})

    # Act / Assert
    assert source.read_json("dir/a.json") == {"x": 1}
    assert source.read_json("dir/missing.json") is None
    assert calls[0] == f"{cc.RESOLVE_BASE}/dir/a.json"


def test_http_read_json_with_invalid_json_raises_source_error(monkeypatch):
    # Arrange
    source, _, _ = scripted_source(monkeypatch, {"bad.json": [FakeResponse(b"<html>")]})

    # Act / Assert
    with pytest.raises(cc.SourceError, match="JSON 아님"):
        source.read_json("dir/bad.json")


@pytest.mark.parametrize("body", [b"[]", b'"text"', b"1", b"null"])
def test_http_read_json_with_non_object_json_raises_source_error(monkeypatch, body):
    # Arrange
    source, _, _ = scripted_source(monkeypatch, {"odd.json": [FakeResponse(body)]})

    # Act / Assert
    with pytest.raises(cc.SourceError, match="JSON 객체 아님"):
        source.read_json("dir/odd.json")


def test_http_read_bytes_returns_the_body_untouched_and_none_for_404(monkeypatch):
    # Arrange — JSON 이 아닌 바이트도 그대로 돌려준다(band_guard 의 snapshot 이 원문 그대로 저장한다)
    raw = b'{"a":  1,\n "b": "\xed\x95\x9c"}\n\xff'
    source, calls, _ = scripted_source(monkeypatch, {"a.json": [FakeResponse(raw)], "missing.json": [http_error(404)]})

    # Act / Assert
    assert source.read_bytes("dir/a.json") == raw
    assert source.read_bytes("dir/missing.json") is None
    assert calls[0] == f"{cc.RESOLVE_BASE}/dir/a.json"


class UnreadableResponse(FakeResponse):
    """열기만 해야 한다 — 본문을 읽으면 시험이 실패한다."""

    def read(self) -> bytes:
        raise AssertionError("exists() 는 본문을 읽으면 안 된다")


def test_http_exists_opens_the_file_without_reading_the_body_and_is_false_for_404(monkeypatch):
    # Arrange
    source, calls, _ = scripted_source(monkeypatch, {"here.json": [UnreadableResponse(b"")], "missing.json": [http_error(404)]})

    # Act / Assert
    assert source.exists("dir/here.json") is True
    assert source.exists("dir/missing.json") is False
    assert calls == [f"{cc.RESOLVE_BASE}/dir/here.json", f"{cc.RESOLVE_BASE}/dir/missing.json"]


def test_http_exists_retries_and_reports_a_persistent_failure_as_source_error(monkeypatch):
    # Arrange
    source, calls, sleeps = scripted_source(monkeypatch, {"a.json": [http_error(503)]})

    # Act / Assert — 확인을 못 한 것과 없는 것은 다르다
    with pytest.raises(cc.SourceError, match="status=503"):
        source.exists("dir/a.json")
    assert len(calls) == cc.FETCH_RETRIES and sleeps == [2.0, 4.0]


def test_http_retries_503_then_succeeds(monkeypatch):
    # Arrange
    source, calls, sleeps = scripted_source(monkeypatch, {"a.json": [http_error(503), http_error(503), FakeResponse(b"{}")]})

    # Act
    payload = source.read_json("dir/a.json")

    # Assert
    assert payload == {}
    assert len(calls) == 3
    assert sleeps == [2.0, 4.0]


def test_http_persistent_server_error_reports_status_without_url_or_body(monkeypatch):
    # Arrange
    source, calls, _ = scripted_source(monkeypatch, {"a.json": [http_error(500)]})

    # Act / Assert
    with pytest.raises(cc.SourceError) as exc:
        source.read_json("dir/a.json")
    assert "HTTPError status=500" in str(exc.value)
    assert "example.invalid" not in str(exc.value)
    assert "secret body" not in str(exc.value)
    assert len(calls) == cc.FETCH_RETRIES


def test_http_client_error_other_than_404_is_not_retried(monkeypatch):
    # Arrange
    source, calls, _ = scripted_source(monkeypatch, {"a.json": [http_error(403)]})

    # Act / Assert
    with pytest.raises(cc.SourceError, match="status=403"):
        source.read_json("dir/a.json")
    assert len(calls) == 1


def test_http_network_error_reports_class_name_only(monkeypatch):
    # Arrange
    source, _, _ = scripted_source(monkeypatch, {"a.json": [urllib.error.URLError("host example.invalid refused")]})

    # Act / Assert
    with pytest.raises(cc.SourceError) as exc:
        source.read_json("dir/a.json")
    assert "URLError" in str(exc.value)
    assert "example.invalid" not in str(exc.value)


def test_http_body_read_cut_off_midway_is_retried_then_succeeds(monkeypatch):
    # Arrange — 본문을 읽다 끊기면(IncompleteRead) URLError 가 아니라 HTTPException 이라 예전에는 재시도 없이 traceback 이었다
    cut = FakeResponse(b"", read_error=http.client.IncompleteRead(b"{", 10))
    source, calls, sleeps = scripted_source(monkeypatch, {"a.json": [cut, FakeResponse(b'{"x": 1}')]})

    # Act
    payload = source.read_json("dir/a.json")

    # Assert
    assert payload == {"x": 1}
    assert len(calls) == 2
    assert sleeps == [2.0]


@pytest.mark.parametrize("error", [
    http.client.IncompleteRead(b"{", 10),
    http.client.RemoteDisconnected("peer example.invalid closed"),
    ssl.SSLError("handshake with example.invalid failed"),
    ConnectionResetError("reset by example.invalid"),
    TimeoutError("read timed out example.invalid"),
], ids=lambda error: type(error).__name__)
def test_http_persistent_body_read_failure_ends_as_source_error_with_class_name_only(monkeypatch, error):
    # Arrange
    source, calls, _ = scripted_source(monkeypatch, {"a.json": [FakeResponse(b"", read_error=error)]})

    # Act / Assert — 종료 코드 2 로 이어지는 SourceError 이고, 메시지에는 클래스 이름만 남는다
    with pytest.raises(cc.SourceError) as exc:
        source.read_json("dir/a.json")
    assert str(exc.value) == f"dir/a.json: {type(error).__name__}"
    assert len(calls) == cc.FETCH_RETRIES


def tree_entry(rel: str, kind: str = "file") -> dict:
    return {"type": kind, "path": rel}


def test_http_list_stamps_follows_link_header_pagination_and_filters_names(monkeypatch):
    # Arrange
    first = [tree_entry(f"{cc.CAND_GRID_DIR}/grid_2026101515.json"), tree_entry(f"{cc.CAND_GRID_DIR}/grid_latest.json")]
    second = [tree_entry(f"{cc.CAND_GRID_DIR}/grid_2026101512.json"), tree_entry(f"{cc.CAND_GRID_DIR}/quality_2026101512.json")]
    next_url = f"{cc.TREE_BASE}/{cc.CAND_GRID_DIR}?cursor=abc"
    source, calls, _ = scripted_source(monkeypatch, {
        cc.CAND_GRID_DIR: [FakeResponse(json.dumps(first).encode(), link=f'<{next_url}>; rel="next"')],
        "cursor=abc": [FakeResponse(json.dumps(second).encode())],
    })

    # Act
    stamps = source.list_stamps(cc.CAND_GRID_DIR, "grid")

    # Assert
    assert stamps == ["2026101512", "2026101515"]
    assert calls[1] == next_url


def test_http_list_stamps_of_a_directory_that_does_not_exist_yet_is_empty(monkeypatch):
    # Arrange — 후보가 한 번도 발행되지 않은 상태
    source, _, _ = scripted_source(monkeypatch, {cc.CAND_GRID_DIR: [http_error(404)]})

    # Act / Assert
    assert source.list_stamps(cc.CAND_GRID_DIR, "grid") == []


def test_http_list_stamps_with_non_json_listing_raises_source_error(monkeypatch):
    # Arrange
    source, _, _ = scripted_source(monkeypatch, {cc.CAND_GRID_DIR: [FakeResponse(b"<html>")]})

    # Act / Assert
    with pytest.raises(cc.SourceError, match="JSON 아님"):
        source.list_stamps(cc.CAND_GRID_DIR, "grid")


@pytest.mark.parametrize("listing", [
    {"path": "x"}, "text", 3, None, [1, 2], [tree_entry("ok"), "not-a-dict"], [[]],
], ids=["object", "string", "number", "null", "numbers", "mixed", "nested-list"])
def test_http_list_stamps_with_unexpected_listing_shape_raises_source_error(monkeypatch, listing):
    # Arrange — JSON 은 맞지만 객체 배열이 아니면 AttributeError traceback 대신 SourceError
    source, _, _ = scripted_source(monkeypatch, {cc.CAND_GRID_DIR: [FakeResponse(json.dumps(listing).encode())]})

    # Act / Assert
    with pytest.raises(cc.SourceError, match="객체 배열이 아님"):
        source.list_stamps(cc.CAND_GRID_DIR, "grid")


def test_http_list_stamps_of_an_empty_listing_is_empty(monkeypatch):
    # Arrange
    source, _, _ = scripted_source(monkeypatch, {cc.CAND_GRID_DIR: [FakeResponse(b"[]")]})

    # Act / Assert
    assert source.list_stamps(cc.CAND_GRID_DIR, "grid") == []


def test_http_source_never_sends_credentials(monkeypatch):
    # Arrange — 공개 저장소만 읽는다. 요청 헤더는 User-Agent 하나뿐이어야 한다(Authorization·쿠키 없음).
    seen: list[dict[str, str]] = []

    def fake_urlopen(req, timeout):
        seen.append(dict(req.header_items()))
        return FakeResponse(b"{}")

    monkeypatch.setattr(cc.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("HF_TOKEN", "must-not-be-sent")

    # Act
    cc.HttpSource().read_json("dir/a.json")

    # Assert
    assert seen and all({name.lower() for name in headers} == {"user-agent"} for headers in seen)


# ── main(): --local 종단 ─────────────────────────────────────────────────────
def build_local_tree(root: Path, stamps: list[str], *, mutate_candidate=None) -> None:
    for s in stamps:
        write_json(root, f"{cc.PROD_GRID_DIR}/grid_{s}.json", grid())
        write_json(root, f"{cc.PROD_QUALITY_DIR}/quality_{s}.json", quality())
        cand = grid() if mutate_candidate is None else mutate_candidate(grid())
        write_json(root, f"{cc.CAND_GRID_DIR}/grid_{s}.json", cand)
        write_json(root, f"{cc.CAND_QUALITY_DIR}/quality_{s}.json", quality())


def test_main_local_identical_trees_pass_with_exit_zero_and_markdown(tmp_path, capsys, monkeypatch):
    # Arrange
    build_local_tree(tmp_path, ["2026101509", "2026101512", "2026101515"])
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    # Act
    code = cc.main(["--local", str(tmp_path)], validate=ok)

    # Assert
    out = capsys.readouterr().out
    assert code == 0
    assert "PASS 3 · FAIL 0" in out
    assert "| 2026101512 | PASS | 예 |" in out


def test_main_local_changed_candidate_fails_with_exit_one(tmp_path, capsys, monkeypatch):
    # Arrange — 모든 후보 행의 p50 을 5 만큼 올린다
    build_local_tree(tmp_path, ["2026101509", "2026101512", "2026101515"],
                     mutate_candidate=lambda g: shift(g, range(200), "predicted_p50", 5.0))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    # Act
    code = cc.main(["--local", str(tmp_path)], validate=ok)

    # Assert
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL 3" in out
    assert "최대 \\|Δ\\|≤1.0" in out  # 표 셀 안의 `|` 는 이스케이프된다


def _table_blocks(markdown: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("|"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _columns(line: str) -> int:
    return len(re.findall(r"(?<!\\)\|", line)) - 1


def test_markdown_tables_keep_a_constant_column_count_even_though_labels_contain_pipes(tmp_path, capsys, monkeypatch):
    # Arrange — 실패 행 표까지 나오도록 후보 p50 을 크게 올린다. 검사 이름에 `|Δ|` 가 들어 있다.
    build_local_tree(tmp_path, ["2026101512"], mutate_candidate=lambda g: shift(g, range(200), "predicted_p50", 5.0))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    # Act
    cc.main(["--local", str(tmp_path), "--min-strict-pairs", "1"], validate=ok)
    blocks = _table_blocks(capsys.readouterr().out)

    # Assert — 이스케이프 안 된 `|` 가 셀 안에 있으면 GitHub 가 열을 더 만들어 표가 깨진다
    assert len(blocks) >= 3  # 요약 표 · 검사 표 · 실패 행 표
    for block in blocks:
        assert {_columns(line) for line in block} == {_columns(block[0])}, block[0]


def test_cell_escapes_pipes_and_flattens_newlines():
    # Arrange / Act / Assert
    assert cc._cell("a|b\nc") == "a\\|b c"
    assert cc._cell(1.5) == "1.5"


def test_main_local_too_few_strict_pairs_exits_two_unless_minimum_is_lowered(tmp_path, capsys, monkeypatch):
    # Arrange
    build_local_tree(tmp_path, ["2026101512"])
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    # Act
    default_min = cc.main(["--local", str(tmp_path)], validate=ok)
    lowered = cc.main(["--local", str(tmp_path), "--min-strict-pairs", "1"], validate=ok)

    # Assert
    capsys.readouterr()
    assert (default_min, lowered) == (2, 0)


def test_main_json_output_is_parseable_and_carries_the_exit_code(tmp_path, capsys, monkeypatch):
    # Arrange
    build_local_tree(tmp_path, ["2026101512"])
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    # Act
    code = cc.main(["--local", str(tmp_path), "--json", "--min-strict-pairs", "1"], validate=ok)

    # Assert
    data = json.loads(capsys.readouterr().out)
    assert code == data["summary"]["exit_code"] == 0
    assert data["product"] == "predictions"
    assert data["pairs"][0]["stamp"] == "2026101512"
    assert data["pairs"][0]["verdict"] == cc.PASS


def test_main_json_output_stays_strict_json_when_the_candidate_has_nan(tmp_path, capsys, monkeypatch):
    # Arrange — `json.dumps` 기본값이면 NaN 토큰이 그대로 나가 jq 같은 엄격 파서가 거부한다
    build_local_tree(tmp_path, ["2026101512"], mutate_candidate=lambda g: tweak(g, [3], predicted_p50=float("nan")))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    # Act
    code = cc.main(["--local", str(tmp_path), "--json", "--min-strict-pairs", "1"], validate=ok)

    # Assert
    data = json.loads(capsys.readouterr().out, parse_constant=lambda token: pytest.fail(f"JSON 에 {token} 토큰이 나갔다"))
    worst = data["pairs"][0]["worst_rows"][0]
    assert code == 1
    assert (worst["field"], worst["cand"], worst["abs_delta"]) == ("predicted_p50", "nan", None)


def test_main_appends_markdown_to_step_summary_but_json_mode_does_not(tmp_path, capsys, monkeypatch):
    # Arrange
    build_local_tree(tmp_path / "tree", ["2026101512"])
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    # Act
    cc.main(["--local", str(tmp_path / "tree"), "--json", "--min-strict-pairs", "1"], validate=ok)
    json_mode_written = summary.exists()
    cc.main(["--local", str(tmp_path / "tree"), "--min-strict-pairs", "1"], validate=ok)

    # Assert
    capsys.readouterr()
    assert json_mode_written is False
    assert "후보 ↔ 운영 비교" in summary.read_text(encoding="utf-8")


def test_main_since_limits_the_compared_stamps(tmp_path, capsys, monkeypatch):
    # Arrange
    build_local_tree(tmp_path, ["2026101409", "2026101512", "2026101515"])
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    # Act
    code = cc.main(["--local", str(tmp_path), "--since", "2026-10-15", "--json", "--min-strict-pairs", "1"], validate=ok)

    # Assert
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert [p["stamp"] for p in data["pairs"]] == ["2026101512", "2026101515"]


def test_main_invalid_since_exits_two_with_a_message(tmp_path, capsys):
    # Act
    code = cc.main(["--local", str(tmp_path), "--since", "yesterday"], validate=ok)

    # Assert
    assert code == 2
    assert "--since" in capsys.readouterr().err


def test_main_unreadable_input_exits_two_not_one(tmp_path, capsys):
    # Arrange — 읽기 실패는 "후보가 틀렸다(1)" 가 아니라 "판단 불가(2)"
    write_json(tmp_path, f"{cc.CAND_GRID_DIR}/grid_2026101512.json", {})
    (tmp_path / cc.PROD_GRID_DIR).mkdir(parents=True)
    (tmp_path / cc.PROD_GRID_DIR / "grid_2026101512.json").write_text("{broken", encoding="utf-8")

    # Act
    code = cc.main(["--local", str(tmp_path)], validate=ok)

    # Assert
    assert code == 2
    assert "compare_candidate:" in capsys.readouterr().err


# ── 계약 검증기 연결 ─────────────────────────────────────────────────────────
TS = "2026-10-15T12:00:00Z"


def minimal_valid_grid() -> dict:
    return {
        "schema_version": "grid_latest.v1",
        "generated_at": TS,
        "model_version": "test",
        "model_sha": None,
        "model_available": True,
        "observation_source": None,
        "observation_slot": None,
        "count": 1,
        "predictions": [{
            "name": "test-1", "lat": 0.0, "lon": 0.0, "timestamp": TS,
            "predicted_p10": 1.0, "predicted_p50": 2.0, "predicted_p90": 3.0,
            "model_version": "test", "source": "test",
        }],
        "metadata": {
            "quantiles": [0.1, 0.5, 0.9], "unit": "ug/m3", "note": "test",
            "epistemic_uncertainty": False, "band_calibration_applied": False,
            "aod_provenance": {"counts": {}, "n_rows": 1, "n_source_known": 0, "n_source_unknown": 1, "signal_share": None},
            "train_aod_provenance": "unrecorded", "observation_staleness": None,
        },
    }


def test_contract_validator_accepts_a_valid_grid_and_describes_violations():
    # Arrange
    validate = cc.contract_validator()
    broken = minimal_valid_grid()
    del broken["predictions"][0]["lat"]

    # Act / Assert
    assert validate(minimal_valid_grid()) is None
    assert "lat" in validate(broken)
    assert isinstance(validate({"schema_version": "unknown.v9"}), str)
    assert isinstance(validate({}), str)


def test_main_uses_the_real_contract_validator_when_none_is_injected(tmp_path, capsys, monkeypatch):
    # Arrange — 후보 grid 가 계약을 어기면(행의 lat 없음) 값이 같아도 FAIL 이다
    good = {**minimal_valid_grid(), "model_sha": SHA}  # model_sha 가 null 이면 계약 검사 전에 폴백 FAIL 로 끝난다
    bad = copy.deepcopy(good)
    del bad["predictions"][0]["lat"]
    write_json(tmp_path, f"{cc.PROD_GRID_DIR}/grid_{STAMP}.json", good)
    write_json(tmp_path, f"{cc.CAND_GRID_DIR}/grid_{STAMP}.json", bad)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    # Act
    code = cc.main(["--local", str(tmp_path), "--min-strict-pairs", "1"])

    # Assert
    assert code == 1
    assert "후보 계약" in capsys.readouterr().out


# ── band: 판정 ───────────────────────────────────────────────────────────────
BAND_STAMP = "2026101506"
ISSUE = "2026-10-15T06:00:00+00:00"
HORIZONS = (1, 6, 24)
N_HOLDOUT = {"1": 1144, "6": 1146, "24": 469}


def band(n_cities: int = 20, **top: object) -> dict:
    """20 도시 × 3 지평 = 60 셀. 95% 경계(57/60)를 셀 하나 단위로 나눌 수 있는 크기다."""
    payload = {
        "schema_version": "w3-band-v1",
        "generated_at": "2026-10-15T06:40:00+00:00",
        "model": "timesfm-test + CQR",
        "issue_time": ISSUE,
        "cities": [
            {"name": f"city-{i:02d}", "horizons": [
                {"lead_hours": h, "valid_time": f"2026-10-15T{6 + h % 18:02d}:00:00+00:00",
                 "p10": 5.0 + i % 5 + h / 10, "p50": 10.0 + i % 5 + h / 10, "p90": 20.0 + i % 5 + h / 10}
                for h in HORIZONS]}
            for i in range(n_cities)
        ],
        "uncertainty": {
            "method": "CQR conformal",
            "picp80_claim_by_horizon": {h: {"picp80_holdout": 0.81, "n_holdout": n, "status": "ok"} for h, n in N_HOLDOUT.items()},
            "provisional_horizons": [],
        },
        "dqss": {"grade": None, "status": "unscored"},
    }
    payload.update(top)
    return payload


def band_shift(payload: dict, cells: list[tuple[int, int]], field: str, delta: float) -> dict:
    """(도시 번호, 지평 번호) 셀들의 field 를 delta 만큼 옮긴 사본."""
    out = copy.deepcopy(payload)
    for city, horizon in cells:
        out["cities"][city]["horizons"][horizon][field] += delta
    return out


def band_set(payload: dict, city: int, horizon: int, **fields: object) -> dict:
    out = copy.deepcopy(payload)
    out["cities"][city]["horizons"][horizon].update(fields)
    return out


def classify_band(prod, cand, validate=ok) -> cc.PairResult:
    return cc.classify_band_pair(BAND_STAMP, prod, cand, validate)


def test_identical_band_payloads_pass_as_a_strict_pair():
    # Act
    result = classify_band(band(), band())

    # Assert
    assert (result.verdict, result.strict, result.reasons) == (cc.PASS, True, [])
    assert result.max_delta == {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    assert all(c.ok is not False for c in result.checks)


def test_the_helper_payload_satisfies_the_real_band_contract():
    # Arrange / Act / Assert — 아래 시험의 입력이 계약에서 벗어나 있으면 시험이 아무것도 보증하지 못한다
    assert cc.contract_validator()(band()) is None


def test_band_difference_of_exactly_the_tolerance_everywhere_passes():
    # Arrange — 0.01 은 경계값(부동소수 오차 포함)이다
    cand = band_shift(band(), [(c, h) for c in range(20) for h in range(3)], "p50", 0.01)

    # Act
    result = classify_band(band(), cand)

    # Assert
    assert result.verdict == cc.PASS


def test_three_of_sixty_cells_over_tolerance_still_passes_at_the_95_percent_boundary():
    # Arrange
    cand = band_shift(band(), [(0, 0), (1, 1), (2, 2)], "p50", 0.02)

    # Act
    result = classify_band(band(), cand)

    # Assert
    assert result.verdict == cc.PASS
    assert check(result, "p50 |Δ|").detail == "57/60 (95.0%)"


def test_four_of_sixty_cells_over_tolerance_fails_the_95_percent_rule():
    # Arrange
    cand = band_shift(band(), [(0, 0), (1, 1), (2, 2), (3, 0)], "p50", 0.02)

    # Act
    result = classify_band(band(), cand)

    # Assert
    assert result.verdict == cc.FAIL
    assert check(result, "p50 |Δ|").ok is False
    assert len(result.worst_rows) == 4


@pytest.mark.parametrize("quantile", ["p10", "p50", "p90"])
def test_every_quantile_counts_toward_the_cell_rule(quantile):
    # Arrange — 보정 표본이 같은 지평(h1)에서는 세 분위수 모두 판정한다. 한 분위수만 어긋나도 어긋난 셀이다
    cand = band_shift(band(), [(c, 0) for c in range(4)], quantile, 0.05)

    # Act
    result = classify_band(band(), cand)

    # Assert
    assert result.verdict == cc.FAIL
    assert result.worst_rows[0]["field"] == f"h1 {quantile}"


def test_a_single_cell_off_by_exactly_half_passes_but_more_than_half_fails():
    # Arrange — 한 셀은 95% 규칙이 허용하는 범위지만, 지평별 최대 |Δ| 는 0.5 를 넘으면 안 된다
    at_limit = band_shift(band(), [(0, 0)], "p50", 0.5)
    over = band_shift(band(), [(0, 0)], "p50", 0.51)

    # Act
    ok_result, bad_result = classify_band(band(), at_limit), classify_band(band(), over)

    # Assert
    assert ok_result.verdict == cc.PASS and ok_result.max_delta["p50"] == 0.5
    assert bad_result.verdict == cc.FAIL
    assert check(bad_result, "최대").ok is False


@pytest.mark.parametrize("bad", [None, "12", float("nan"), float("inf"), float("-inf"), True])
@pytest.mark.parametrize("side", ["prod", "cand"])
def test_non_numeric_or_non_finite_band_value_fails_even_for_a_single_cell(bad, side):
    # Arrange — NaN 은 모든 비교가 False 라 허용오차 판정을 조용히 통과한다
    broken = band_set(band(), 3, 1, p90=bad)
    prod, cand = (broken, band()) if side == "prod" else (band(), broken)

    # Act
    result = classify_band(prod, cand)

    # Assert
    assert result.verdict == cc.FAIL
    assert "숫자 아님 1" in check(result, "최대").detail


def test_missing_city_in_candidate_fails_on_the_city_set():
    # Arrange
    cand = band()
    del cand["cities"][5]

    # Act
    result = classify_band(band(), cand)

    # Assert
    assert result.verdict == cc.FAIL
    assert check(result, "도시 집합").detail == "운영 20 / 후보 19 (운영만 1, 후보만 0)"


def test_extra_city_in_candidate_fails_on_the_city_set():
    # Act
    result = classify_band(band(), band(21))

    # Assert
    assert check(result, "도시 집합").detail == "운영 20 / 후보 21 (운영만 0, 후보만 1)"
    assert result.verdict == cc.FAIL


def test_a_missing_horizon_in_one_city_fails_on_the_horizon_set():
    # Arrange
    cand = band()
    del cand["cities"][7]["horizons"][2]

    # Act
    result = classify_band(band(), cand)

    # Assert
    assert result.verdict == cc.FAIL
    assert check(result, "지평 집합").detail == "도시별 지평이 다른 도시 1곳"
    assert check(result, "도시 집합").ok is True


def test_duplicate_city_horizon_cells_fail():
    # Arrange
    cand = band()
    cand["cities"][4]["horizons"].append(copy.deepcopy(cand["cities"][4]["horizons"][0]))

    # Act
    result = classify_band(band(), cand)

    # Assert
    assert result.verdict == cc.FAIL
    assert check(result, "키 중복").detail == "운영 0 / 후보 1"


def test_payloads_without_a_common_cell_fail_instead_of_passing_vacuously():
    # Arrange — 도시 이름이 전부 다르면 비교할 셀이 없다
    cand = band()
    for i, city in enumerate(cand["cities"]):
        city["name"] = f"other-{i:02d}"

    # Act
    result = classify_band(band(), cand)

    # Assert
    assert result.verdict == cc.FAIL
    assert check(result, "비교할 셀").ok is False


def test_malformed_city_entries_are_skipped_not_crashed_on():
    # Arrange — 모양이 틀린 항목은 셀로 세지 않는다(계약 검사가 따로 잡는다)
    cand = band()
    cand["cities"] += ["text", None, {"name": "no-horizons"}, {"name": "bad", "horizons": [1, None]}]

    # Act
    result = classify_band(band(), cand)

    # Assert — 도시 이름은 있지만 셀이 없는 항목은 도시 집합에 들지 않는다
    assert check(result, "도시 집합").ok is True
    assert result.verdict == cc.PASS


# ── band: 엄격 짝 조건 ───────────────────────────────────────────────────────
@pytest.mark.parametrize("issue_time", ["2026-10-15T06:00:00Z", "2026-10-15T15:00:00+09:00", "2026-10-15T06:00:00"])
def test_same_instant_written_differently_is_still_a_strict_pair(issue_time):
    # Act — 시각은 파싱해서 비교한다(`Z` · 오프셋 · 오프셋 없음(UTC))
    result = classify_band(band(), band(issue_time=issue_time))

    # Assert
    assert (result.verdict, result.strict) == (cc.PASS, True)


@pytest.mark.parametrize("issue_time", ["2026-10-15T06:00:01+00:00", "2026-10-15T07:00:00+00:00", "", None, 12, "not-a-time"])
def test_different_or_unreadable_issue_time_makes_the_pair_input_diff(issue_time):
    # Act
    result = classify_band(band(), band(issue_time=issue_time))

    # Assert
    assert (result.verdict, result.strict) == (cc.INPUT_DIFF, False)
    assert "issue_time" in result.reasons[0]


def test_different_model_makes_the_pair_input_diff():
    # Act
    result = classify_band(band(), band(model="other model"))

    # Assert
    assert (result.verdict, result.strict) == (cc.INPUT_DIFF, False)
    assert "model" in result.reasons[0]


def with_n_holdout(payload: dict, by_horizon: dict[str, object]) -> dict:
    """지평(문자열 키)별 n_holdout 을 바꾼 사본."""
    out = copy.deepcopy(payload)
    for horizon, n in by_horizon.items():
        out["uncertainty"]["picp80_claim_by_horizon"][horizon]["n_holdout"] = n
    return out


ALL_HORIZONS = {h: n + 1 for h, n in N_HOLDOUT.items()}  # 세 지평 모두 보정 표본이 다르다


def test_n_holdout_difference_keeps_the_pair_strict_and_leaves_only_that_horizon_out_of_the_p10_p90_judgement():
    # Arrange — 지평 24 의 보정(CQR) 표본만 다르다. 그 지평의 p10/p90 은 크게 달라도 정당하다.
    cand = band_shift(band_shift(band(), [(c, 2) for c in range(20)], "p10", 3.0), [(c, 2) for c in range(20)], "p90", -3.0)
    cand = with_n_holdout(cand, {"24": N_HOLDOUT["24"] + 1})

    # Act
    result = classify_band(band(), cand)

    # Assert — 짝은 엄격 짝으로 남고(p50 은 판정한다), 뺀 지평의 큰 차이는 최대 |Δ| 에도 들어가지 않는다
    assert (result.verdict, result.strict, result.reasons) == (cc.PASS, True, [])
    assert (result.tail_judged, result.tail_skipped) == (["1", "6"], ["24"])
    assert check(result, "p10/p90 |Δ|").detail == "40/40 (100.0%) — 지평 1, 6"
    assert check(result, "판정 제외").detail == "h24(n_holdout 운영 469 / 후보 470)"
    assert check(result, "판정 제외").ok is None
    assert result.max_delta == {"p10": 0.0, "p50": 0.0, "p90": 0.0}


def test_p50_is_judged_on_every_horizon_even_when_n_holdout_differs_everywhere():
    # Arrange — 보정 표본이 모든 지평에서 달라도 p50 은 보정과 무관하다
    cand = with_n_holdout(band_shift(band(), [(c, 2) for c in range(20)], "p50", 0.2), ALL_HORIZONS)

    # Act
    result = classify_band(band(), cand)

    # Assert
    assert (result.verdict, result.strict) == (cc.FAIL, True)
    assert check(result, "p50 |Δ|").detail == "40/60 (66.7%)"
    assert check(result, "p50 |Δ|").ok is False
    assert {w["field"] for w in result.worst_rows} == {"h24 p50"}


def test_p10_p90_are_judged_on_the_horizons_whose_n_holdout_matches():
    # Arrange — 지평 24 는 표본이 달라 빼고, 표본이 같은 지평 1 의 p10 이 네 도시에서 어긋난다
    cand = with_n_holdout(band_shift(band(), [(c, 0) for c in range(4)], "p10", 0.05), {"24": N_HOLDOUT["24"] + 1})

    # Act
    result = classify_band(band(), cand)

    # Assert — 엄격 지평의 셀 40 개 중 36 개(90%)만 맞아 95% 규칙에 걸린다
    assert (result.verdict, result.strict) == (cc.FAIL, True)
    assert check(result, "p10/p90 |Δ|").detail == "36/40 (90.0%) — 지평 1, 6"
    assert check(result, "p10/p90 |Δ|").ok is False
    assert check(result, "p50 |Δ|").ok is True
    assert [w["field"] for w in result.worst_rows] == ["h1 p10"] * 4


def test_every_horizon_with_a_different_n_holdout_leaves_p10_p90_unjudged_but_the_pair_stays_strict():
    # Arrange — 세 지평 모두 표본이 다르다. p10/p90 이 어디서 얼마나 달라도 판정하지 않는다.
    cand = band_shift(band_shift(band(), [(c, h) for c in range(20) for h in range(3)], "p10", 4.0),
                      [(c, h) for c in range(20) for h in range(3)], "p90", 4.0)
    cand = with_n_holdout(cand, ALL_HORIZONS)

    # Act
    result = classify_band(band(), cand)

    # Assert — p50 만 검증된 엄격 PASS 다. 무엇이 검증되지 않았는지 검사 표에 그대로 적힌다.
    assert (result.verdict, result.strict, result.reasons) == (cc.PASS, True, [])
    assert (result.tail_judged, result.tail_skipped) == ([], ["1", "6", "24"])
    assert check(result, "p10/p90 |Δ|").ok is None
    assert "엄격 비교할 지평 없음" in check(result, "p10/p90 |Δ|").detail
    assert check(result, "최대").detail == "p10 (판정 없음), p50 0.0, p90 (판정 없음)"
    assert check(result, "판정 제외").detail == "h1(n_holdout 운영 1144 / 후보 1145), h6(n_holdout 운영 1146 / 후보 1147), h24(n_holdout 운영 469 / 후보 470)"


def test_identical_payloads_judge_p10_p90_on_every_horizon_and_list_no_excluded_horizon():
    # Act
    result = classify_band(band(), band())

    # Assert
    assert (result.tail_judged, result.tail_skipped) == (["1", "6", "24"], [])
    assert check(result, "p10/p90 |Δ|").detail == "60/60 (100.0%) — 지평 1, 6, 24"
    assert not [c for c in result.checks if "판정 제외" in c.name]


def test_a_horizon_that_exists_on_one_side_only_is_neither_judged_nor_listed_as_left_out():
    # Arrange — 후보에 지평 24 가 통째로 없다(지평 집합 검사가 FAIL 로 잡는다). 비교하는 지평은 양쪽에 있는 것뿐이다.
    cand = band()
    for city in cand["cities"]:
        del city["horizons"][2]

    # Act
    result = classify_band(band(), cand)

    # Assert
    assert result.verdict == cc.FAIL and check(result, "지평 집합").ok is False
    assert (result.tail_judged, result.tail_skipped) == (["1", "6"], [])


def test_a_horizon_left_out_of_the_p10_p90_judgement_is_still_covered_by_the_contract_check():
    # Arrange — 비교가 읽지 않는 지평의 p90 이 숫자가 아니어도 후보 계약 검사가 잡는다(실제 계약 검증기)
    cand = with_n_holdout(band_set(band(), 3, 2, p90="x"), {"24": N_HOLDOUT["24"] + 1})

    # Act
    result = classify_band(band(), cand, cc.contract_validator())

    # Assert
    assert result.verdict == cc.FAIL
    assert check(result, "후보 계약").ok is False
    assert check(result, "최대").ok is True  # 비교는 그 셀의 p90 을 읽지 않았다


def _drop_claims(payload: dict) -> None:
    del payload["uncertainty"]["picp80_claim_by_horizon"]


def _empty_claims(payload: dict) -> None:
    payload["uncertainty"]["picp80_claim_by_horizon"] = {}


def _null_n_holdout(payload: dict) -> None:
    payload["uncertainty"]["picp80_claim_by_horizon"]["1"]["n_holdout"] = None


def _bool_n_holdout(payload: dict) -> None:
    payload["uncertainty"]["picp80_claim_by_horizon"]["1"]["n_holdout"] = True


def _float_n_holdout(payload: dict) -> None:
    payload["uncertainty"]["picp80_claim_by_horizon"]["1"]["n_holdout"] = 1144.0


def _claim_not_object(payload: dict) -> None:
    payload["uncertainty"]["picp80_claim_by_horizon"]["1"] = 1144


def _uncertainty_not_object(payload: dict) -> None:
    payload["uncertainty"] = []


EVERY_HORIZON = ["1", "6", "24"]


@pytest.mark.parametrize(
    ("mutate", "unreadable"),
    [(_drop_claims, EVERY_HORIZON), (_empty_claims, EVERY_HORIZON), (_uncertainty_not_object, EVERY_HORIZON),
     (_null_n_holdout, ["1"]), (_bool_n_holdout, ["1"]), (_float_n_holdout, ["1"]), (_claim_not_object, ["1"])],
)
@pytest.mark.parametrize("side", ["prod", "cand"])
def test_n_holdout_that_cannot_be_read_is_never_treated_as_equal(mutate, unreadable, side):
    # Arrange — 양쪽이 똑같이 망가져도 "같다" 고 말할 수 없다(없음 == 없음 이 엄격 지평이 되면 안 된다)
    broken = band()
    mutate(broken)
    prod, cand = (broken, band()) if side == "prod" else (band(), broken)

    # Act
    both_broken = classify_band(broken, copy.deepcopy(broken))
    one_broken = classify_band(prod, cand)

    # Assert — 읽을 수 없는 지평만 p10/p90 판정에서 빠진다. 짝은 그대로 엄격 짝이다(p50 은 판정한다).
    for result in (both_broken, one_broken):
        assert (result.verdict, result.strict) == (cc.PASS, True)
        assert result.tail_skipped == unreadable
        assert result.tail_judged == [h for h in EVERY_HORIZON if h not in unreadable]
    for h in unreadable:  # 읽을 수 없는 쪽은 `?` 로 적힌다
        prod_n, cand_n = ("?", N_HOLDOUT[h]) if side == "prod" else (N_HOLDOUT[h], "?")
        assert f"h{h}(n_holdout 운영 {prod_n} / 후보 {cand_n})" in check(one_broken, "판정 제외").detail
        assert f"h{h}(n_holdout 운영 ? / 후보 ?)" in check(both_broken, "판정 제외").detail


def test_holdout_counts_keep_only_the_horizons_read_as_integers_keyed_by_string():
    # Arrange
    payload = band()
    payload["uncertainty"]["picp80_claim_by_horizon"]["6"]["n_holdout"] = "1146"

    # Act / Assert
    assert cc._holdout_counts(payload) == {"1": 1144, "24": 469}
    assert cc._holdout_counts({}) == {}


def test_input_diff_pair_ignores_value_differences_but_reports_city_overlap():
    # Arrange — 값이 크게 달라도 입력이 다르면 PASS/FAIL 을 내지 않는다
    cand = band_shift(band(model="other model"), [(c, h) for c in range(20) for h in range(3)], "p50", 9.0)
    del cand["cities"][0]

    # Act
    result = classify_band(band(), cand)

    # Assert
    assert result.verdict == cc.INPUT_DIFF
    assert [(c.name, c.ok) for c in result.checks] == [("후보 계약", True), ("도시 겹침(보고)", None)]
    assert check(result, "도시 겹침").detail == "95.0%"


def test_input_diff_pair_below_the_overlap_standard_says_so():
    # Arrange
    cand = band(model="other model")
    cand["cities"] = cand["cities"][:10]

    # Act
    result = classify_band(band(), cand)

    # Assert
    assert check(result, "도시 겹침").detail == "50.0% — 95.0% 미만"


def test_input_diff_pair_with_contract_violation_is_a_fail():
    # Arrange
    def broken(payload: dict) -> str | None:
        return "p10: 숫자 아님" if payload.get("model") == "other model" else None

    # Act
    result = classify_band(band(), band(model="other model"), broken)

    # Assert
    assert result.verdict == cc.FAIL
    assert result.strict is False
    assert "후보가 계약을 어겼다" in result.reasons[0]


def test_strict_band_pair_with_contract_violation_fails_even_when_values_match():
    # Act
    result = classify_band(band(), band(), lambda payload: "cities[0].name: 누락")

    # Assert
    assert result.verdict == cc.FAIL
    assert check(result, "후보 계약").ok is False
    assert result.strict is True


@pytest.mark.parametrize(("prod", "cand", "side"), [(None, band(), "운영 사본"), (band(), None, "후보"), (None, None, "운영 사본")])
def test_missing_band_file_on_either_side_is_unpaired(prod, cand, side):
    # Act
    result = classify_band(prod, cand)

    # Assert
    assert (result.verdict, result.strict) == (cc.UNPAIRED, False)
    assert side in result.reasons[0]


@pytest.mark.parametrize(
    "value",
    ["2026-10-15T06:00:00Z", "2026-10-15T06:00:00+00:00", "2026-10-15T15:00:00+09:00", "2026-10-15T06:00:00", " 2026-10-15T06:00:00Z ",
     "", "  ", "not-a-time", "2026-13-01T00:00:00Z", None, 1, ["2026-10-15T06:00:00Z"], "0001-01-01T00:00:00+14:00"],
)
def test_instant_agrees_with_the_shared_parser_in_publish_targets(value):
    # Arrange — 이 스크립트는 단독 실행이라 파서를 한 벌 더 둔다. 같은 입력에서 같은 시각이거나, 함께 실패해야 한다(None ↔ ValueError)
    try:
        shared = pt.parse_utc(value)
    except ValueError:
        shared = None

    # Act / Assert
    assert cc._instant(value) == shared


# ── band: 요약·짝짓기·종단 ───────────────────────────────────────────────────
def test_summarize_with_band_quantiles_keys_the_max_delta_by_band_names():
    # Arrange
    strict = cc.PairResult(BAND_STAMP, cc.PASS, True, max_delta={"p10": 0.1, "p50": 0.3, "p90": 0.2})

    # Act
    summary = cc.summarize([strict], prod_only=0, min_strict=1, quantiles=cc.BAND_QUANTILES)

    # Assert
    assert summary["max_delta"] == {"p10": 0.1, "p50": 0.3, "p90": 0.2}


def test_the_default_minimum_of_strict_pairs_is_two_for_band_and_three_for_predictions():
    # Act / Assert — 컷오버 조건(엄격 짝 ≥ 2 / ≥ 3)이 기본값이다
    assert cc.MIN_STRICT_PAIRS_BY_PRODUCT == {"predictions": 3, "band": 2}


def test_render_markdown_names_the_product_and_shows_band_labels():
    # Arrange
    failed = classify_band(band(), band_shift(band(), [(0, 0), (1, 1), (2, 2), (3, 2)], "p50", 0.2))
    summary = cc.summarize([failed], prod_only=0, min_strict=2, quantiles=cc.BAND_QUANTILES)

    # Act
    text = cc.render_markdown([failed], summary, "band")

    # Assert
    assert "## 후보 ↔ 운영 비교 (band)" in text
    assert "| city-00 | h1 p50 |" in text
    assert "최대 |Δ| p10 0.0, p50 0.2, p90 0.0" in text
    assert "p10/p90 엄격 비교 지평" not in text  # 이 줄은 run_band 가 채우는 요약에만 있다


def band_tree(stamps_prod: list[str], stamps_cand: list[str]) -> dict[str, dict]:
    files: dict[str, dict] = {}
    for s in stamps_prod:
        files[f"{cc.PROD_BAND_DIR}/band_{s}.json"] = band()
    for s in stamps_cand:
        files[f"{cc.CAND_BAND_DIR}/band_{s}.json"] = band()
    return files


def test_run_band_pairs_candidate_stamps_with_the_same_stamp_in_the_production_snapshots():
    # Arrange
    source = MemorySource(band_tree(["2026101500", "2026101506", "2026101512"], ["2026101506", "2026101512"]))

    # Act
    results, summary = cc.run_band(source, ok, since=None, min_strict=2)

    # Assert
    assert [(r.stamp, r.verdict) for r in results] == [("2026101506", cc.PASS), ("2026101512", cc.PASS)]
    assert summary["exit_code"] == 0
    assert source.reads == [f"{cc.PROD_BAND_DIR}/band_2026101506.json", f"{cc.CAND_BAND_DIR}/band_2026101506.json",
                            f"{cc.PROD_BAND_DIR}/band_2026101512.json", f"{cc.CAND_BAND_DIR}/band_2026101512.json"]


def test_run_band_summary_counts_the_p10_p90_horizons_judged_and_left_out_over_strict_pairs_only():
    # Arrange — 06시 짝은 지평 24 의 보정 표본만 다르고, 12시 짝은 issue_time 이 달라 엄격 짝이 아니다(지평을 세지 않는다)
    files = band_tree(["2026101506", "2026101512"], ["2026101506", "2026101512"])
    files[f"{cc.CAND_BAND_DIR}/band_2026101506.json"] = with_n_holdout(band(), {"24": N_HOLDOUT["24"] + 1})
    files[f"{cc.CAND_BAND_DIR}/band_2026101512.json"] = band(issue_time="2026-10-15T07:00:00+00:00")

    # Act
    results, summary = cc.run_band(MemorySource(files), ok, since=None, min_strict=1)
    text = cc.render_markdown(results, summary, "band")

    # Assert
    assert [(r.stamp, r.verdict, r.strict) for r in results] == [("2026101506", cc.PASS, True), ("2026101512", cc.INPUT_DIFF, False)]
    assert results[1].tail_judged == results[1].tail_skipped == []
    assert (summary["tail_horizons_judged"], summary["tail_horizons_skipped"]) == (2, 1)
    assert "p10/p90 엄격 비교 지평 2곳 · n_holdout 이 달라 판정에서 뺀 지평 1곳 (p50 은 모든 지평을 판정한다)" in text


def test_the_predictions_summary_has_no_p10_p90_horizon_line():
    # Arrange
    results = [_result(cc.PASS)]

    # Act
    summary = cc.summarize(results, prod_only=0, min_strict=1)

    # Assert — 지평별 보정 표본은 band 의 사정이다
    assert "tail_horizons_judged" not in summary
    assert "p10/p90" not in cc.render_markdown(results, summary)


def test_run_band_marks_a_candidate_without_a_snapshot_unpaired_and_counts_snapshot_only_stamps():
    # Arrange — 사본은 00·12·18시, 후보는 06·12시. 첫 후보(06시) 이후의 사본-only 는 18시 하나(00시는 후보 시작 전)
    source = MemorySource(band_tree(["2026101500", "2026101512", "2026101518"], ["2026101506", "2026101512"]))

    # Act
    results, summary = cc.run_band(source, ok, since=None, min_strict=1)

    # Assert
    assert [(r.stamp, r.verdict) for r in results] == [("2026101506", cc.UNPAIRED), ("2026101512", cc.PASS)]
    assert summary["prod_only_stamps"] == 1
    assert summary["exit_code"] == 0


def test_run_band_filters_candidate_stamps_by_since_and_without_any_candidate_is_insufficient():
    # Arrange
    source = MemorySource(band_tree(["2026101500", "2026101506", "2026101512"], ["2026101500", "2026101506", "2026101512"]))

    # Act
    results, _ = cc.run_band(source, ok, since="2026101506", min_strict=1)
    empty, summary = cc.run_band(MemorySource(band_tree(["2026101506"], [])), ok, since=None, min_strict=2)

    # Assert
    assert [r.stamp for r in results] == ["2026101506", "2026101512"]
    assert empty == [] and summary["exit_code"] == 2


def build_band_tree(root: Path, stamps: list[str], *, mutate_candidate=None) -> None:
    for s in stamps:
        write_json(root, f"{cc.PROD_BAND_DIR}/band_{s}.json", band())
        write_json(root, f"{cc.CAND_BAND_DIR}/band_{s}.json", band() if mutate_candidate is None else mutate_candidate(band()))


def test_main_band_local_trees_pass_with_the_band_default_of_two_strict_pairs(tmp_path, capsys, monkeypatch):
    # Arrange
    build_band_tree(tmp_path / "two", ["2026101500", "2026101506"])
    build_band_tree(tmp_path / "one", ["2026101500"])
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    # Act
    two = cc.main(["--product", "band", "--local", str(tmp_path / "two")], validate=ok)
    out = capsys.readouterr().out
    one = cc.main(["--product", "band", "--local", str(tmp_path / "one")], validate=ok)
    capsys.readouterr()
    one_lowered = cc.main(["--product", "band", "--local", str(tmp_path / "one"), "--min-strict-pairs", "1"], validate=ok)
    capsys.readouterr()

    # Assert — band 의 기본 최소는 2다(predictions 는 3)
    assert (two, one, one_lowered) == (0, 2, 0)
    assert "## 후보 ↔ 운영 비교 (band)" in out
    assert "PASS 2 · FAIL 0" in out
    assert "| 2026101506 | PASS | 예 |" in out


def test_main_band_local_changed_candidate_fails_with_exit_one(tmp_path, capsys, monkeypatch):
    # Arrange — 모든 후보 셀의 p50 을 5 만큼 올린다
    build_band_tree(tmp_path, ["2026101500", "2026101506"],
                    mutate_candidate=lambda p: band_shift(p, [(c, h) for c in range(20) for h in range(3)], "p50", 5.0))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    # Act
    code = cc.main(["--product", "band", "--local", str(tmp_path)], validate=ok)

    # Assert
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL 2" in out
    assert "최대 \\|Δ\\|≤0.5" in out


def test_main_band_json_output_names_the_product_and_stays_strict_json_with_nan(tmp_path, capsys, monkeypatch):
    # Arrange — NaN 토큰이 그대로 나가면 jq 같은 엄격 파서가 거부한다
    build_band_tree(tmp_path, ["2026101506"], mutate_candidate=lambda p: band_set(p, 2, 1, p50=float("nan")))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    # Act
    code = cc.main(["--product", "band", "--local", str(tmp_path), "--json", "--min-strict-pairs", "1"], validate=ok)

    # Assert
    data = json.loads(capsys.readouterr().out, parse_constant=lambda token: pytest.fail(f"JSON 에 {token} 토큰이 나갔다"))
    worst = data["pairs"][0]["worst_rows"][0]
    assert code == 1
    assert data["product"] == "band"
    assert (worst["name"], worst["field"], worst["cand"], worst["abs_delta"]) == ("city-02", "h6 p50", "nan", None)
    assert data["pairs"][0]["tail_judged"] == ["1", "6", "24"] and data["pairs"][0]["tail_skipped"] == []
    assert (data["summary"]["tail_horizons_judged"], data["summary"]["tail_horizons_skipped"]) == (3, 0)


def test_main_band_uses_the_real_contract_validator_when_none_is_injected(tmp_path, capsys, monkeypatch):
    # Arrange — 후보가 계약을 어기면(도시 이름 없음) 값이 같아도 FAIL 이다
    bad = band()
    del bad["cities"][0]["name"]
    write_json(tmp_path, f"{cc.PROD_BAND_DIR}/band_{BAND_STAMP}.json", band())
    write_json(tmp_path, f"{cc.CAND_BAND_DIR}/band_{BAND_STAMP}.json", bad)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    # Act
    code = cc.main(["--product", "band", "--local", str(tmp_path), "--min-strict-pairs", "1"])

    # Assert
    assert code == 1
    assert "후보 계약" in capsys.readouterr().out


def test_main_predictions_remains_the_default_product(tmp_path, capsys, monkeypatch):
    # Arrange — band 파일만 있는 트리에서 기본 제품(predictions)은 후보가 없어 판단 보류(2)다
    build_band_tree(tmp_path, ["2026101506"])
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    # Act
    code = cc.main(["--local", str(tmp_path), "--json"], validate=ok)

    # Assert
    data = json.loads(capsys.readouterr().out)
    assert (code, data["product"], data["pairs"]) == (2, "predictions", [])
