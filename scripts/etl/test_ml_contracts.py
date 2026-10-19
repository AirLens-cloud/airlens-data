"""ML 산출물 계약 5종 시험 (네트워크 0).

`contracts/validate.py` 는 `hf_publish._validator()` 처럼 파일 경로로 적재한다.
최소 페이로드는 각 스키마를 읽고 손으로 만들었다 — 실데이터를 복사하지 않는다.

여기서 검사하는 것:
1. 5개 스키마가 `load_schema` 로 로드된다 (미지원 키워드가 있으면 로드에서 죽는다).
2. `$id` 와 `properties.schema_version.const` 가 파일 이름과 같다.
3. `resolve_schema` 가 페이로드의 `schema_version` 으로 그 파일을 고른다 (`--schema auto`).
4. required 만 채운 페이로드는 통과하고, `schema_version` 을 바꾸거나 required 를 하나
   지우면 `ContractError` 로 실패한다 — 계약이 비어 있지 않다. 지운 필드를 짚는 오류
   문구까지 확인한다.
5. 값 제약(minimum·maximum·minItems·enum·const·date-time)을 한 곳씩 깨면 그 자리를 짚는
   `ContractError` 로 실패한다.
6. `grid_latest.v1` 의 `metadata.observation_staleness` 는 최소 페이로드에서 null 이다.
   객체 형태로도 통과하고, 그 안의 required 를 하나 지우면 실패한다.
7. 발행기가 쓰는 경로 `hf_publish.check_contract(<파일>, "auto")` 가 계약마다 최소
   페이로드는 통과시키고, 값 제약을 깬 페이로드는 `SystemExit(1)` 로 막는다.
8. `grid_latest.v1` 의 행별 `met_source` 와 `metadata.met_provenance` 는 선택 필드다(옛
   발행물에는 없다). 실으면 통과하고, 집계 객체의 required 를 하나 지우거나 값 제약을 깨면
   실패한다.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest

import hf_publish as hp

CONTRACTS_DIR = Path(__file__).resolve().parents[2] / "contracts"

_spec = importlib.util.spec_from_file_location(
    "_airlens_ml_contracts_validate", CONTRACTS_DIR / "validate.py"
)
assert _spec is not None and _spec.loader is not None
validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validator)

TS = "2026-01-01T00:00:00Z"

# 계약마다 required 만 채운 최소 페이로드. 배열은 항목 스키마까지 시험되도록 1건씩 둔다.
MINIMAL: dict[str, dict] = {
    "grid_latest.v1": {
        "schema_version": "grid_latest.v1",
        "generated_at": TS,
        "model_version": "test",
        "model_sha": None,
        "model_available": True,
        "observation_source": None,
        "observation_slot": None,
        "count": 1,
        "predictions": [
            {
                "name": "test-1",
                "lat": 0.0,
                "lon": 0.0,
                "timestamp": TS,
                "predicted_p10": 1.0,
                "predicted_p50": 2.0,
                "predicted_p90": 3.0,
                "model_version": "test",
                "source": "test",
            }
        ],
        "metadata": {
            "quantiles": [0.1, 0.5, 0.9],
            "unit": "ug/m3",
            "note": "test",
            "epistemic_uncertainty": False,
            "band_calibration_applied": False,
            "aod_provenance": {
                "counts": {},
                "n_rows": 1,
                "n_source_known": 0,
                "n_source_unknown": 1,
                "signal_share": None,
            },
            "train_aod_provenance": "unrecorded",
            "observation_staleness": None,
        },
    },
    "data_quality.v1": {
        "schema_version": "data_quality.v1",
        "generated_at": TS,
        "meta": {
            "source": "measured",
            "generated_at": TS,
            "engine": "test",
            "inputs": {
                "freshness": "measured",
                "completeness": "measured",
                "consistency": "measured",
                "stability": "measured",
                "residual": "measured",
            },
            "measured_weight_range": [100, 100],
            "graded_stations": 1,
            "total_stations": 1,
            "cutoff_basis": "test",
            "note": "test",
        },
        "stations": [
            {
                "station_id": "test-1",
                "measured_weight": 100,
                "final_score": 90.0,
                "badge": "High",
                "reasons": {},
                "computed_at": TS,
            }
        ],
    },
    "data_quality_withheld.v1": {
        "schema_version": "data_quality_withheld.v1",
        "generated_at": TS,
        "meta": {
            "source": "withheld",
            "reason": "test",
            "failed_rules": ["test_rule"],
            "generated_at": TS,
        },
        "stations": [],
    },
    "shadow_stations.v1": {
        "schema_version": "shadow_stations.v1",
        "generated_at": TS,
        "meta": {
            "source": "openaq-shadow",
            "window_hours": 48,
            "readings_window_hours": 24,
            "min_std_samples": 3,
            "station_count": 1,
            "note": "test",
        },
        "stations": [
            {
                "station_id": "openaq-1",
                "last_update": TS,
                "readings_24h": 24,
                "rolling_std_48h": None,
            }
        ],
    },
    "w3-band-v1": {
        "schema_version": "w3-band-v1",
        "generated_at": TS,
        "model": "test",
        "issue_time": TS,
        "cities": [
            {
                "name": "Test City",
                "horizons": [
                    {"lead_hours": 1, "valid_time": TS, "p10": 1.0, "p50": 2.0, "p90": 3.0}
                ],
            }
        ],
        "uncertainty": {
            "method": "test",
            "picp80_claim_by_horizon": {},
            "provisional_horizons": [],
        },
        "dqss": {"grade": None, "status": "test"},
    },
}
NAMES = list(MINIMAL)


def _schema_path(name: str) -> Path:
    return CONTRACTS_DIR / f"{name}.schema.json"


def _field_paths(value, path: tuple = ()):
    """최소 페이로드 안의 모든 객체 필드 경로.

    최소 페이로드는 손으로 만든 required 전용 문서다. 통과하면 required 는 전부 실려 있다는
    뜻이고, 이 경로 중 어느 하나를 지워도 실패하면 실린 필드는 전부 required 라는 뜻이다.
    스키마에서 경로를 뽑으면 required 가 빠져도 시험 건수만 줄고 통과하므로 페이로드 쪽에서 뽑는다.
    """
    if isinstance(value, dict):
        for key, sub in value.items():
            yield path + (key,)
            yield from _field_paths(sub, path + (key,))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _field_paths(item, path + (i,))


def _without(payload: dict, path: tuple) -> dict:
    out = copy.deepcopy(payload)
    node = out
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    return out


def _with(payload: dict, path: tuple, value) -> dict:
    """페이로드 사본에서 `path` 한 곳만 `value` 로 바꾼다. 없는 키(선택 필드)는 만들어 넣는다."""
    out = copy.deepcopy(payload)
    node = out
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = copy.deepcopy(value)
    return out


REQUIRED_CASES = [
    pytest.param(name, path, id=f"{name}:{'.'.join(map(str, path))}")
    for name in NAMES
    for path in _field_paths(MINIMAL[name])
]

# 값 제약을 한 곳씩 깬다: (계약, 바꿀 위치, 넣을 값, 오류 문구에 실려야 할 것).
# 위치와 한계값은 각 스키마에서 읽었다. 문구에 위치(`$.…`)를 넣어 두면 의도한 제약이 아니라
# 다른 이유로 죽은 경우를 통과로 세지 않는다.
VALUE_VIOLATIONS = [
    # grid_latest.v1
    ("grid_latest.v1", ("predictions", 0, "lat"), 91, "$.predictions[0].lat: 91 > maximum 90"),
    (
        "grid_latest.v1",
        ("predictions", 0, "predicted_p50"),
        -1,
        "$.predictions[0].predicted_p50: -1 < minimum 0",
    ),
    # confidence_grade 는 선택 필드라 최소 페이로드에 없다 — `_with` 가 만들어 넣는다.
    (
        "grid_latest.v1",
        ("predictions", 0, "confidence_grade"),
        "Z",
        "$.predictions[0].confidence_grade: enum",
    ),
    # met_source 도 선택 필드다.
    (
        "grid_latest.v1",
        ("predictions", 0, "met_source"),
        "maybe",
        "$.predictions[0].met_source: enum",
    ),
    ("grid_latest.v1", ("metadata", "quantiles"), [], "$.metadata.quantiles: 항목 0개 < minItems 1"),
    ("grid_latest.v1", ("generated_at",), "not-a-date", "$.generated_at: date-time 파싱 실패"),
    # data_quality.v1
    ("data_quality.v1", ("meta", "source"), "withheld", "$.meta.source: enum"),
    (
        "data_quality.v1",
        ("stations", 0, "final_score"),
        101,
        "$.stations[0].final_score: 101 > maximum 100",
    ),
    # data_quality_withheld.v1
    (
        "data_quality_withheld.v1",
        ("stations",),
        [{"station_id": "x"}],
        "$.stations: const [] 기대",
    ),
    (
        "data_quality_withheld.v1",
        ("meta", "source"),
        "measured",
        "$.meta.source: const 'withheld' 기대",
    ),
    # shadow_stations.v1
    (
        "shadow_stations.v1",
        ("meta", "min_std_samples"),
        1,
        "$.meta.min_std_samples: 1 < minimum 2",
    ),
    (
        "shadow_stations.v1",
        ("meta", "source"),
        "x",
        "$.meta.source: const 'openaq-shadow' 기대",
    ),
    # w3-band-v1
    (
        "w3-band-v1",
        ("cities", 0, "horizons", 0, "lead_hours"),
        25,
        "$.cities[0].horizons[0].lead_hours: 25 > maximum 24",
    ),
    ("w3-band-v1", ("cities",), [], "$.cities: 항목 0개 < minItems 1"),
    ("w3-band-v1", ("issue_time",), "not-a-date", "$.issue_time: date-time 파싱 실패"),
]
VALUE_CASES = [
    pytest.param(name, path, bad, message, id=f"{name}:{'.'.join(map(str, path))}")
    for name, path, bad, message in VALUE_VIOLATIONS
]

# `metadata.observation_staleness` 는 최소 페이로드에서 null 이라 객체 쪽 스키마가 시험되지
# 않는다. 객체 변형은 required 만 채우고 dropped 에 1건을 둔다. lat/lon 은 선택 필드라 넣지
# 않는다 — 실린 필드가 전부 required 여야 아래처럼 지워 보는 시험이 성립한다.
STALENESS_PATH = ("metadata", "observation_staleness")
STALENESS = {
    "max_age_hours": 6.0,
    "n_published": 1,
    "n_dropped_stale": 1,
    "dropped": [{"station_id": "test-2", "last_seen": TS, "age_hours": 12.0}],
}
GRID_WITH_STALENESS = _with(MINIMAL["grid_latest.v1"], STALENESS_PATH, STALENESS)
STALENESS_CASES = [
    pytest.param(path, id=".".join(map(str, path)))
    for path in _field_paths(STALENESS, STALENESS_PATH)
]

# `metadata.met_provenance` 는 선택 필드라 최소 페이로드에 없다. 객체 변형은 required 만 채우고
# counts 는 비워 둔다 — STALENESS 와 같은 이유로 실린 필드가 전부 required 여야 한다.
MET_PATH = ("metadata", "met_provenance")
MET_PROVENANCE = {
    "counts": {},
    "n_rows": 1,
    "n_source_known": 0,
    "n_source_unknown": 1,
    "live_share": None,
}
GRID_WITH_MET = _with(MINIMAL["grid_latest.v1"], MET_PATH, MET_PROVENANCE)
MET_CASES = [
    pytest.param(path, id=".".join(map(str, path)))
    for path in _field_paths(MET_PROVENANCE, MET_PATH)
]


@pytest.mark.parametrize("name", NAMES)
def test_schema_loads_without_unsupported_keywords(name):
    schema = validator.load_schema(_schema_path(name))  # 미지원 키워드가 있으면 ContractError
    assert schema["type"] == "object"
    # 이름만 줘도 contracts/ 안의 같은 파일을 찾는다 (CLI 의 `--schema <이름>` 경로)
    assert validator.load_schema(name) == schema


@pytest.mark.parametrize("name", NAMES)
def test_id_and_schema_version_const_match_the_file_name(name):
    schema = validator.load_schema(_schema_path(name))
    assert schema["$id"] == name
    assert schema["properties"]["schema_version"]["const"] == name


@pytest.mark.parametrize("name", NAMES)
def test_resolve_schema_picks_the_file_named_by_schema_version(name):
    resolved = validator.resolve_schema(MINIMAL[name])
    assert resolved == validator.load_schema(_schema_path(name))
    assert resolved["$id"] == name


@pytest.mark.parametrize("name", NAMES)
def test_minimal_payload_passes_through_auto_resolution(name):
    payload = MINIMAL[name]
    validator.validate_payload(payload, validator.resolve_schema(payload))


@pytest.mark.parametrize("name", NAMES)
def test_other_schema_version_fails(name):
    payload = {**MINIMAL[name], "schema_version": "other.v1"}
    with pytest.raises(validator.ContractError):  # const 불일치
        validator.validate_payload(payload, validator.load_schema(_schema_path(name)))
    with pytest.raises(validator.ContractError):  # 그 이름의 계약이 없다
        validator.resolve_schema(payload)


@pytest.mark.parametrize(("name", "path"), REQUIRED_CASES)
def test_removing_a_required_field_fails(name, path):
    schema = validator.load_schema(_schema_path(name))
    missing = re.escape(f"필수 필드 '{path[-1]}' 누락")  # 지운 필드를 짚어야 한다
    with pytest.raises(validator.ContractError, match=missing):
        validator.validate_payload(_without(MINIMAL[name], path), schema)


@pytest.mark.parametrize(("name", "path", "bad", "message"), VALUE_CASES)
def test_breaking_a_value_constraint_fails_at_that_spot(name, path, bad, message):
    schema = validator.load_schema(_schema_path(name))
    payload = _with(MINIMAL[name], path, bad)
    with pytest.raises(validator.ContractError, match=re.escape(message)):
        validator.validate_payload(payload, schema)


def test_observation_staleness_object_passes_through_auto_resolution():
    payload = GRID_WITH_STALENESS
    assert payload["metadata"]["observation_staleness"] is not None  # null 과 다른 분기를 탄다
    validator.validate_payload(payload, validator.resolve_schema(payload))


@pytest.mark.parametrize("path", STALENESS_CASES)
def test_removing_a_required_observation_staleness_field_fails(path):
    schema = validator.load_schema(_schema_path("grid_latest.v1"))
    missing = re.escape(f"필수 필드 '{path[-1]}' 누락")
    with pytest.raises(validator.ContractError, match=missing):
        validator.validate_payload(_without(GRID_WITH_STALENESS, path), schema)


@pytest.mark.parametrize("met_source", ["live", "fallback", None])
def test_met_source_and_provenance_pass_through_auto_resolution(met_source):
    # 계약이 두 필드를 실제로 선언하는지부터 본다. additionalProperties 가 열려 있어
    # 선언이 없어도 아래 검증은 통과하기 때문이다.
    schema = validator.load_schema(_schema_path("grid_latest.v1"))
    assert schema["properties"]["predictions"]["items"]["properties"]["met_source"]["enum"] == [
        "live",
        "fallback",
        None,
    ]
    assert "met_provenance" in schema["properties"]["metadata"]["properties"]
    # 행 1건짜리 발행물: 행에 met_source, metadata 에 그 집계.
    known = met_source is not None
    payload = _with(GRID_WITH_MET, ("predictions", 0, "met_source"), met_source)
    payload = _with(
        payload,
        MET_PATH,
        {
            "counts": {met_source: 1} if known else {},
            "n_rows": 1,
            "n_source_known": 1 if known else 0,
            "n_source_unknown": 0 if known else 1,
            "live_share": (1.0 if met_source == "live" else 0.0) if known else None,
        },
    )
    validator.validate_payload(payload, validator.resolve_schema(payload))


@pytest.mark.parametrize("path", MET_CASES)
def test_removing_a_required_met_provenance_field_fails(path):
    schema = validator.load_schema(_schema_path("grid_latest.v1"))
    missing = re.escape(f"필수 필드 '{path[-1]}' 누락")
    with pytest.raises(validator.ContractError, match=missing):
        validator.validate_payload(_without(GRID_WITH_MET, path), schema)


def test_met_provenance_live_share_above_one_fails():
    schema = validator.load_schema(_schema_path("grid_latest.v1"))
    payload = _with(GRID_WITH_MET, MET_PATH + ("live_share",), 1.5)
    message = "$.metadata.met_provenance.live_share: 1.5 > maximum 1"
    with pytest.raises(validator.ContractError, match=re.escape(message)):
        validator.validate_payload(payload, schema)


# ── 발행기 경로: hf_publish.check_contract(<파일>, "auto") ───────────────────


@pytest.mark.parametrize("name", NAMES)
def test_publisher_auto_check_passes_the_minimal_payload(name, tmp_path, capsys):
    src = tmp_path / f"{name}.json"
    src.write_text(json.dumps(MINIMAL[name]), encoding="utf-8")
    hp.check_contract(str(src), "auto")
    assert "contract ok" in capsys.readouterr().out


@pytest.mark.parametrize(("name", "path", "bad", "message"), VALUE_CASES)
def test_publisher_auto_check_blocks_a_broken_value_constraint(
    name, path, bad, message, tmp_path, capsys
):
    src = tmp_path / f"{name}.json"
    src.write_text(json.dumps(_with(MINIMAL[name], path, bad)), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        hp.check_contract(str(src), "auto")
    assert e.value.code == 1
    assert message in capsys.readouterr().err  # 읽기 실패 등 다른 이유로 죽은 게 아니다
