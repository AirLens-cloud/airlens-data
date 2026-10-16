#!/usr/bin/env python3
"""후보(`aq-data/_candidate/`)와 운영 발행물을 같은 STAMP 끼리 짝지어 비교한다 (stdlib 전용).

predictions(grid + DQSS, 기본)와 band(예측 밴드, `--product band`)를 다룬다. 입력은 공개 저장소(토큰 불요) 또는
`--local DIR`(저장소 트리를 그대로 미러한 디렉터리 — 오프라인·시험용)이다. 비교 판정은 네트워크 없는 순수 함수라
단위 시험으로 고정한다.

**엄격 짝** = 같은 STAMP + `observation_slot` + `model_sha` + `model_version` 이 모두 같은 쌍. 입력이 같으니
출력도 같아야 한다 — 추론 경로에 난수가 없다(난수는 학습 부트스트랩에만 있다). 다만 run 사이에 바뀌는 것이
있다: 현재 시각, 실시간 Open-Meteo, DQSS 신선도(경과 시간), shadow 통계. 그래서 허용오차가 있다.
행 키는 (name, lat, lon). "행 입력 동일" = `observed_pm25` 와 `input_updated_at` 이 같음.
`model_sha` 는 양쪽 모두 값이 있어야 같다고 본다 — null 은 폴백 경로·구 아티팩트라 null == null 이 같은 모델이라는 뜻이
아니다. 후보가 폴백으로 돌았으면(`model_available` 이 true 가 아니거나 `model_sha` 가 null) 비교하지 않고 FAIL 이다.

  grid (엄격 짝, 입력 같은 행 기준)
    · 행 키 집합·count 동일, 키 중복 없음, 입력 같은 행 ≥ 99%
    · predicted_p10/p50/p90 |Δ| ≤ 0.1 인 행 ≥ 99%, 최대 |Δ| ≤ 1.0   (값은 0.1 단위로 반올림돼 나온다)
    · confidence_grade · aod_source · feature_regime 일치 ≥ 99% (필드별)
  DQSS (엄격 짝)
    · station_id 집합 동일, |Δfinal_score| ≤ 1 인 지점 ≥ 95%, badge 일치 ≥ 95%
  엄격 짝이 아닌 쌍(INPUT-DIFF)은 계약 통과와 키 겹침(≥ 95%)만 **보고**한다. 계약 위반은 FAIL 이다.

band — 후보 `_candidate/forecast-band/history/band_<STAMP>` ↔ 같은 STAMP 의 운영 사본 `_candidate/forecast-band/prod/`
(운영 band 는 latest 하나만 덮어써서 이력이 없다. 후보 잡이 그 시점 운영 latest 를 옮겨 둔다). STAMP = 발행물 `issue_time` 의
UTC 시. 값은 도시·지평(lead_hours) 셀 단위로 비교한다.
  엄격 짝 = `issue_time`(시각으로 비교) · `model` 이 모두 같은 쌍. 입력이 같으니 p50 은 같아야 한다.
  p10/p90 은 보정(CQR) 표본 수 `n_holdout`(uncertainty.picp80_claim_by_horizon)에 달려 있다 — 운영과 후보가 다른 시각에 돌면
  표본이 달라 p10/p90 이 정당하게 달라진다. 그래서 `n_holdout` 이 양쪽 정수로 읽히고 같은 지평에서만 p10/p90 을 엄격 비교하고,
  나머지 지평의 p10/p90 은 INPUT-DIFF(판정에서 뺀다, 보고만)다. p50 은 모든 지평을 판정한다.
    · 도시 집합 동일, 도시별 지평 집합 동일, 키 중복 없음 (필수)
    · p50 |Δ| ≤ 0.01 인 셀 ≥ 95% (공통 셀 전부), p10/p90 |Δ| ≤ 0.01 인 셀 ≥ 95% (엄격 지평의 셀),
      분위수별 최대 |Δ| ≤ 0.5   (p50 은 반올림 없이 float 로 나온다)
  이 값은 첫 엄격 짝을 본 뒤 보정한다. 계약 위반은 INPUT-DIFF 여도 FAIL 이다.
  요약에 p10/p90 을 엄격 비교한 쌍·지평 수와 뺀 쌍·지평 수를 적는다(엄격 짝마다 지평을 세어 더한 값이다). 엄격 비교한 쌍·지평이
  0 이면 p50 만 검증된 것이고, 요약의 p10/p90 최대 |Δ| 는 0 이 아니라 "(판정 없음)"(JSON 은 null)이다. p10/p90 판정이 0건이어도
  종료 코드는 0 일 수 있다 — 컷오버를 승인할 때 요약 줄을 본다.

판정: PASS / FAIL / INPUT-DIFF(입력이 달라 엄격 비교 불가) / UNPAIRED(한쪽에 파일 없음).
종료 코드: 0 = FAIL 없음 + 엄격 PASS ≥ 최소(predictions 3, band 2) / 1 = FAIL 있음 / 2 = 엄격 PASS 부족 또는 읽기 실패.

사용:
    python3 scripts/ml/compare_candidate.py                    # 공개 저장소의 후보 전부
    python3 scripts/ml/compare_candidate.py --since 2026-10-16 --json
    python3 scripts/ml/compare_candidate.py --local DIR --min-strict-pairs 1
    python3 scripts/ml/compare_candidate.py --product band
"""
from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

DATASET = "Robeedau/airlens-live"  # 공개 dataset
RESOLVE_BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
TREE_BASE = f"https://huggingface.co/api/datasets/{DATASET}/tree/main"

PROD_GRID_DIR = "aq-data/predictions/history"
CAND_GRID_DIR = "aq-data/_candidate/predictions/history"
PROD_QUALITY_DIR = "aq-data/quality-history"
CAND_QUALITY_DIR = "aq-data/_candidate/quality-history"
CAND_BAND_DIR = "aq-data/_candidate/forecast-band/history"
PROD_BAND_DIR = "aq-data/_candidate/forecast-band/prod"  # 운영 band 는 latest 만 있어서 후보 잡이 옮겨 둔 사본

# ── 판정 기준 ────────────────────────────────────────────────────────────────
STRICT_ID_FIELDS = ("observation_slot", "model_sha", "model_version")
QUANTILES = ("predicted_p10", "predicted_p50", "predicted_p90")
CATEGORICAL = ("confidence_grade", "aod_source", "feature_regime")
Q_TOL = 0.1  # 0.1 단위 반올림 값이라 한 칸 차이까지 허용
Q_TOL_FRACTION = 0.99
Q_MAX = 1.0
CATEGORICAL_FRACTION = 0.99
SAME_INPUT_FRACTION = 0.99
DQSS_TOL = 1.0
DQSS_SCORE_FRACTION = 0.95
DQSS_BADGE_FRACTION = 0.95
OVERLAP_FRACTION = 0.95  # 엄격 짝이 아닐 때 보고하는 키 겹침 기준
MIN_STRICT_PAIRS = 3
EPS = 1e-9  # 0.1 의 부동소수 오차

# band — 첫 엄격 짝을 본 뒤 보정한다
BAND_QUANTILES = ("p10", "p50", "p90")
BAND_TOL = 1e-2  # p50 은 CQR 전 값이 float 그대로 나온다(반올림 없음). 연산 장치가 다르면 미세하게 달라질 수 있다
BAND_TOL_FRACTION = 0.95
BAND_MAX = 0.5
MIN_STRICT_PAIRS_BY_PRODUCT = {"predictions": MIN_STRICT_PAIRS, "band": 2}

WORST_ROWS = 10
STAMP_RE = re.compile(r"\d{10}")
FETCH_TIMEOUT_S = 60
FETCH_RETRIES = 3
MAX_RESPONSE_BYTES = 32 * 1024 * 1024  # 응답 한 건의 상한. 읽는 파일은 모두 1 MB 아래라 넉넉하고, 끝없이 보내는 응답에도 메모리가 차지 않는다

PASS, FAIL, INPUT_DIFF, UNPAIRED = "PASS", "FAIL", "INPUT-DIFF", "UNPAIRED"


@dataclass
class Check:
    name: str
    detail: str
    ok: bool | None  # None = 보고만(판정에 안 쓴다)


@dataclass
class PairResult:
    stamp: str
    verdict: str
    strict: bool
    reasons: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    max_delta: dict[str, float] = field(default_factory=dict)
    worst_rows: list[dict] = field(default_factory=list)
    tail_judged: list[str] = field(default_factory=list)  # band: p10/p90 을 엄격 비교한 지평(lead_hours)
    tail_skipped: list[str] = field(default_factory=list)  # band: n_holdout 이 달라(또는 읽을 수 없어) p10/p90 판정에서 뺀 지평


class SourceError(Exception):
    """파일을 읽지 못함(없는 것과 다르다 — 없으면 None)."""


class Source(Protocol):
    def list_stamps(self, rel_dir: str, prefix: str) -> list[str]: ...
    def read_json(self, rel_path: str) -> dict | None: ...


def _json_object(data: object, rel_path: str) -> dict:
    """발행물은 JSON 객체다. 배열·문자열·null 은 '없음'(None)이 아니라 읽기 실패다."""
    if not isinstance(data, dict):
        raise SourceError(f"{rel_path}: JSON 객체 아님")
    return data


def parse_json_object(body: bytes, rel_path: str) -> dict:
    """본문 → JSON 객체. JSON 이 아니거나 객체가 아니면 SourceError."""
    try:
        data = json.loads(body)
    except ValueError:  # JSONDecodeError · UnicodeDecodeError
        raise SourceError(f"{rel_path}: JSON 아님") from None
    return _json_object(data, rel_path)


# ── 순수 판정 함수 ───────────────────────────────────────────────────────────
def _num(value: object) -> float | None:
    """유한한 숫자만 돌려준다. bool·문자열·NaN·inf 는 None — NaN 과의 비교는 늘 False 라 허용오차 판정을 조용히 통과한다."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _shown(value: object) -> object:
    """실패 행 표에 내보낼 값. NaN·inf 는 JSON 이 못 쓰는 토큰이라 문자열로 바꾼다."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _index(rows: list[dict], key: Callable[[dict], object]) -> tuple[dict, int]:
    """키 → 행. 두 번째 값은 중복으로 덮인 행 수."""
    out: dict = {}
    for row in rows:
        out[key(row)] = row
    return out, len(rows) - len(out)


def _row_key(row: dict) -> tuple:
    return (row.get("name"), row.get("lat"), row.get("lon"))


def _fraction(hit: int, total: int) -> float:
    return hit / total if total else 0.0


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def strict_mismatches(prod: dict, cand: dict) -> list[str]:
    """엄격 짝 조건 중 서로 다른 필드 이름들. 비면 엄격 짝. `model_sha` 는 null 이면(양쪽이 null 이어도) 다른 것으로 센다."""
    return [f for f in STRICT_ID_FIELDS
            if prod.get(f) != cand.get(f) or (f == "model_sha" and prod.get(f) is None)]


def compare_grid(prod: dict, cand: dict) -> tuple[list[Check], dict[str, float], list[dict]]:
    """엄격 짝의 grid 비교. (checks, 필드별 최대 |Δ|, 실패 행 상위)."""
    checks: list[Check] = []
    p_rows, p_dups = _index(prod.get("predictions") or [], _row_key)
    c_rows, c_dups = _index(cand.get("predictions") or [], _row_key)

    only_prod, only_cand = set(p_rows) - set(c_rows), set(c_rows) - set(p_rows)  # 개수만 쓴다(키에 None·str 이 섞여도 정렬하지 않는다)
    checks.append(Check("행 키 집합", f"운영 {len(p_rows)} / 후보 {len(c_rows)} (운영만 {len(only_prod)}, 후보만 {len(only_cand)})",
                        not only_prod and not only_cand))
    checks.append(Check("count", f"운영 {prod.get('count')} / 후보 {cand.get('count')}", prod.get("count") == cand.get("count")))
    checks.append(Check("키 중복", f"운영 {p_dups} / 후보 {c_dups}", p_dups == 0 and c_dups == 0))

    common = sorted(set(p_rows) & set(c_rows), key=repr)
    if not common:
        checks.append(Check("비교할 행", "공통 행 없음", False))
        return checks, {}, []

    same_input = [k for k in common
                  if p_rows[k].get("observed_pm25") == c_rows[k].get("observed_pm25")
                  and p_rows[k].get("input_updated_at") == c_rows[k].get("input_updated_at")]
    ratio = _fraction(len(same_input), len(common))
    checks.append(Check("입력 같은 행", f"{len(same_input)}/{len(common)} ({_pct(ratio)})", ratio >= SAME_INPUT_FRACTION))
    if not same_input:
        return checks, {}, []

    deltas: dict[str, list[float | None]] = {q: [] for q in QUANTILES}
    worst: list[dict] = []
    within = 0
    for k in same_input:
        row_ok = True
        row_worst: tuple[float, str, object, object] | None = None
        for q in QUANTILES:
            a, b = _num(p_rows[k].get(q)), _num(c_rows[k].get(q))
            if a is None or b is None:
                deltas[q].append(None)
                row_ok = False
                row_worst = (float("inf"), q, _shown(p_rows[k].get(q)), _shown(c_rows[k].get(q)))
                continue
            d = abs(a - b)
            deltas[q].append(d)
            if d > Q_TOL + EPS:
                row_ok = False
            if row_worst is None or d > row_worst[0]:
                row_worst = (d, q, a, b)
        within += row_ok
        if not row_ok and row_worst is not None:
            worst.append({"name": k[0], "field": row_worst[1], "prod": row_worst[2], "cand": row_worst[3],
                          "abs_delta": None if row_worst[0] == float("inf") else round(row_worst[0], 4)})
    frac = _fraction(within, len(same_input))
    checks.append(Check(f"p10/p50/p90 |Δ|≤{Q_TOL}", f"{within}/{len(same_input)} ({_pct(frac)})", frac >= Q_TOL_FRACTION))

    max_delta: dict[str, float] = {}
    for q in QUANTILES:
        finite = [d for d in deltas[q] if d is not None]
        max_delta[q] = round(max(finite), 4) if finite else 0.0
    unreadable = sum(d is None for q in QUANTILES for d in deltas[q])
    over = max(max_delta.values()) > Q_MAX + EPS or unreadable > 0
    checks.append(Check(f"최대 |Δ|≤{Q_MAX}", ", ".join(f"{q.removeprefix('predicted_')} {max_delta[q]}" for q in QUANTILES)
                        + (f" (숫자 아님 {unreadable})" if unreadable else ""), not over))

    for f in CATEGORICAL:
        match = sum(p_rows[k].get(f) == c_rows[k].get(f) for k in same_input)
        frac = _fraction(match, len(same_input))
        checks.append(Check(f"{f} 일치", f"{match}/{len(same_input)} ({_pct(frac)})", frac >= CATEGORICAL_FRACTION))

    worst.sort(key=lambda r: (r["abs_delta"] is not None, -(r["abs_delta"] or 0)))
    return checks, max_delta, worst[:WORST_ROWS]


def compare_quality(prod: dict, cand: dict) -> list[Check]:
    """엄격 짝의 DQSS 비교."""
    p_rows, p_dups = _index(prod.get("stations") or [], lambda r: r.get("station_id"))
    c_rows, c_dups = _index(cand.get("stations") or [], lambda r: r.get("station_id"))
    only_prod, only_cand = set(p_rows) - set(c_rows), set(c_rows) - set(p_rows)
    checks = [
        Check("DQSS 계약 이름", f"운영 {prod.get('schema_version')} / 후보 {cand.get('schema_version')}",
              prod.get("schema_version") == cand.get("schema_version")),
        Check("DQSS station_id 집합", f"운영 {len(p_rows)} / 후보 {len(c_rows)} (운영만 {len(only_prod)}, 후보만 {len(only_cand)})",
              not only_prod and not only_cand and p_dups == 0 and c_dups == 0),
    ]
    common = sorted(set(p_rows) & set(c_rows), key=repr)
    if not common:
        return checks  # 둘 다 withheld(빈 stations) 면 집합 일치만 본다
    score_hit = badge_hit = 0
    for k in common:
        a, b = p_rows[k].get("final_score"), c_rows[k].get("final_score")
        if a is None or b is None:
            score_hit += a is None and b is None
        else:
            na, nb = _num(a), _num(b)
            score_hit += na is not None and nb is not None and abs(na - nb) <= DQSS_TOL + EPS
        badge_hit += p_rows[k].get("badge") == c_rows[k].get("badge")
    frac = _fraction(score_hit, len(common))
    checks.append(Check(f"DQSS |Δfinal_score|≤{DQSS_TOL:g}", f"{score_hit}/{len(common)} ({_pct(frac)})", frac >= DQSS_SCORE_FRACTION))
    frac = _fraction(badge_hit, len(common))
    checks.append(Check("DQSS badge 일치", f"{badge_hit}/{len(common)} ({_pct(frac)})", frac >= DQSS_BADGE_FRACTION))
    return checks


def classify_pair(
    stamp: str,
    prod: dict | None,
    cand: dict | None,
    prod_q: dict | None,
    cand_q: dict | None,
    validate: Callable[[dict], str | None],
) -> PairResult:
    """한 STAMP 의 운영·후보 payload 로 판정한다. validate(payload) 는 계약 위반 설명 또는 None."""
    if prod is None or cand is None:
        side = "운영" if prod is None else "후보"
        return PairResult(stamp, UNPAIRED, False, [f"{side} 쪽 grid 파일이 없다"])

    if cand.get("model_available") is not True or cand.get("model_sha") is None:
        # 가중치를 못 쓰고 폴백 경로로 돌았다면 운영과 값이 같든 말든 컷오버 근거가 아니다(값 대신 있고 없음만 적는다).
        detail = (f"model_available={cand.get('model_available')!r}, "
                  f"model_sha {'null' if cand.get('model_sha') is None else '있음'}")
        return PairResult(stamp, FAIL, False, ["후보가 폴백 경로로 돌았다"], [Check("후보 모델 사용", detail, False)])

    diff = strict_mismatches(prod, cand)
    if diff:
        result = PairResult(stamp, INPUT_DIFF, False,
                            [f"입력이 다르다({', '.join(diff)}) — 엄격 비교 불가"])
        p_keys = {_row_key(r) for r in prod.get("predictions") or []}
        c_keys = {_row_key(r) for r in cand.get("predictions") or []}
        overlap = _fraction(len(p_keys & c_keys), max(len(p_keys), len(c_keys)))
        error = validate(cand)
        low = f" — {_pct(OVERLAP_FRACTION)} 미만" if overlap < OVERLAP_FRACTION else ""
        result.checks = [
            Check("후보 계약", error or "통과", error is None),
            Check("키 겹침(보고)", _pct(overlap) + low, None),
        ]
        if error:
            result.verdict, result.reasons = FAIL, [f"후보가 계약을 어겼다: {error}"]
        return result

    checks, max_delta, worst = compare_grid(prod, cand)
    if prod_q is not None or cand_q is not None:
        if cand_q is None:
            checks.append(Check("DQSS 파일", "후보 없음(운영은 있음)", False))
        elif prod_q is None:
            checks.append(Check("DQSS 파일", "운영 없음 — 미비교", None))
        else:
            checks.extend(compare_quality(prod_q, cand_q))
    else:
        checks.append(Check("DQSS 파일", "양쪽 없음 — 미비교", None))
    error = validate(cand)
    checks.append(Check("후보 계약", error or "통과", error is None))
    reasons = [f"{c.name}: {c.detail}" for c in checks if c.ok is False]
    return PairResult(stamp, FAIL if reasons else PASS, True, reasons, checks, max_delta, worst)


# ── band 판정 ────────────────────────────────────────────────────────────────
def _instant(value: object) -> datetime | None:
    """ISO 시각 문자열 → UTC 시각. 모양이 틀리면 None. 오프셋이 없으면 UTC 로 본다(`Z` 와 `+00:00` 은 같은 시각이다).

    `publish_targets.parse_utc` 와 같은 규칙이다(실패는 None ↔ ValueError). 이 스크립트는 `python3 -I` 로 단독 실행해서 다른
    모듈을 import 하지 않으므로 한 벌 더 두고, test_compare_candidate.py 가 둘을 대조한다.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError):
        return None


def _holdout_counts(payload: dict) -> dict[str, int]:
    """지평(문자열 키) → n_holdout. 정수로 읽히는 지평만 담는다 — 읽을 수 없는 지평은 같다고 말할 수 없다."""
    uncertainty = payload.get("uncertainty")
    claims = uncertainty.get("picp80_claim_by_horizon") if isinstance(uncertainty, dict) else None
    counts: dict[str, int] = {}
    for horizon, claim in claims.items() if isinstance(claims, dict) else ():
        n = claim.get("n_holdout") if isinstance(claim, dict) else None
        if isinstance(n, int) and not isinstance(n, bool):
            counts[str(horizon)] = n
    return counts


def band_identity_mismatches(prod: dict, cand: dict) -> list[str]:
    """엄격 짝 조건 중 서로 다른 항목 이름. 비면 엄격 짝. 값이 없거나 읽을 수 없으면 다른 것으로 센다."""
    out = []
    p_issue, c_issue = _instant(prod.get("issue_time")), _instant(cand.get("issue_time"))
    if p_issue is None or c_issue is None or p_issue != c_issue:
        out.append("issue_time")
    if prod.get("model") != cand.get("model"):
        out.append("model")
    return out


def _band_cells(payload: dict) -> tuple[dict[tuple, dict], int]:
    """(도시, lead_hours) → 지평 행. 두 번째 값은 중복으로 덮인 셀 수. 모양이 다른 항목은 건너뛴다(계약 검사가 따로 잡는다)."""
    cells: dict[tuple, dict] = {}
    total = 0
    cities = payload.get("cities")
    for city in cities if isinstance(cities, list) else []:
        horizons = city.get("horizons") if isinstance(city, dict) else None
        for horizon in horizons if isinstance(horizons, list) else []:
            if isinstance(horizon, dict):
                total += 1
                cells[(city.get("name"), horizon.get("lead_hours"))] = horizon
    return cells, total - len(cells)


def _leads_by_city(cells: dict[tuple, dict]) -> dict[object, set]:
    out: dict[object, set] = {}
    for city, lead in cells:
        out.setdefault(city, set()).add(lead)
    return out


def _lead_order(horizon: str) -> tuple[int, str]:
    return len(horizon), horizon  # "1" < "6" < "24"


def band_tail_split(prod: dict, cand: dict) -> tuple[list[str], list[str]]:
    """공통 지평(lead_hours 문자열)을 (p10/p90 을 엄격 비교하는 지평, 판정에서 뺀 지평)으로 나눈다.
    양쪽 n_holdout 이 정수로 읽히고 같은 지평만 엄격 비교한다(없음 == 없음 은 같은 것이 아니다)."""
    p_counts, c_counts = _holdout_counts(prod), _holdout_counts(cand)
    common = {str(lead) for _name, lead in set(_band_cells(prod)[0]) & set(_band_cells(cand)[0])}
    strict = {h for h in common if h in p_counts and p_counts[h] == c_counts.get(h)}
    return sorted(strict, key=_lead_order), sorted(common - strict, key=_lead_order)


def compare_band(prod: dict, cand: dict) -> tuple[list[Check], dict[str, float], list[dict]]:
    """엄격 짝의 band 비교. p50 은 공통 셀 전부, p10/p90 은 n_holdout 이 같은 지평의 셀만 판정한다.
    (checks, 분위수별 최대 |Δ|(판정한 셀 기준), 실패 셀 상위)."""
    p_cells, p_dups = _band_cells(prod)
    c_cells, c_dups = _band_cells(cand)
    p_leads, c_leads = _leads_by_city(p_cells), _leads_by_city(c_cells)

    only_prod, only_cand = set(p_leads) - set(c_leads), set(c_leads) - set(p_leads)  # 개수만 쓴다(키에 None·str 이 섞여도 정렬하지 않는다)
    checks = [Check("도시 집합", f"운영 {len(p_leads)} / 후보 {len(c_leads)} (운영만 {len(only_prod)}, 후보만 {len(only_cand)})",
                    not only_prod and not only_cand)]
    differing = [city for city in set(p_leads) & set(c_leads) if p_leads[city] != c_leads[city]]
    checks.append(Check("지평 집합", f"도시별 지평이 다른 도시 {len(differing)}곳", not differing))
    checks.append(Check("키 중복", f"운영 {p_dups} / 후보 {c_dups}", p_dups == 0 and c_dups == 0))

    common = sorted(set(p_cells) & set(c_cells), key=repr)
    if not common:
        checks.append(Check("비교할 셀", "공통 셀 없음", False))
        return checks, {}, []

    tail_judged, tail_skipped = band_tail_split(prod, cand)
    judged_leads = set(tail_judged)
    deltas: dict[str, list[float | None]] = {q: [] for q in BAND_QUANTILES}  # 판정한 셀만 담는다
    worst: list[dict] = []
    within_p50 = within_tail = tail_cells = 0
    for key in common:
        in_tail = str(key[1]) in judged_leads
        p50_ok = tail_ok = True
        cell_worst: tuple[float, str, object, object] | None = None
        for q in BAND_QUANTILES:
            if q != "p50" and not in_tail:
                continue  # n_holdout 이 다른 지평의 p10/p90 — 판정에서 뺀다
            a, b = _num(p_cells[key].get(q)), _num(c_cells[key].get(q))
            if a is None or b is None:
                deltas[q].append(None)
                good = False
                cell_worst = (float("inf"), q, _shown(p_cells[key].get(q)), _shown(c_cells[key].get(q)))
            else:
                d = abs(a - b)
                deltas[q].append(d)
                good = d <= BAND_TOL + EPS
                if cell_worst is None or d > cell_worst[0]:
                    cell_worst = (d, q, a, b)
            if not good:
                if q == "p50":
                    p50_ok = False
                else:
                    tail_ok = False
        within_p50 += p50_ok
        tail_cells += in_tail
        within_tail += in_tail and tail_ok
        if not (p50_ok and tail_ok) and cell_worst is not None:
            worst.append({"name": key[0], "field": f"h{key[1]} {cell_worst[1]}", "prod": cell_worst[2], "cand": cell_worst[3],
                          "abs_delta": None if cell_worst[0] == float("inf") else round(cell_worst[0], 6)})
    frac = _fraction(within_p50, len(common))
    checks.append(Check(f"p50 |Δ|≤{BAND_TOL:g}", f"{within_p50}/{len(common)} ({_pct(frac)})", frac >= BAND_TOL_FRACTION))
    if tail_cells:
        tail_frac = _fraction(within_tail, tail_cells)
        checks.append(Check(f"p10/p90 |Δ|≤{BAND_TOL:g}", f"{within_tail}/{tail_cells} ({_pct(tail_frac)}) — 지평 {', '.join(tail_judged)}",
                            tail_frac >= BAND_TOL_FRACTION))
    else:
        checks.append(Check(f"p10/p90 |Δ|≤{BAND_TOL:g}", "엄격 비교할 지평 없음 — n_holdout 이 모든 지평에서 다르거나 읽을 수 없다", None))
    if tail_skipped:
        p_counts, c_counts = _holdout_counts(prod), _holdout_counts(cand)
        checks.append(Check("p10/p90 판정 제외(INPUT-DIFF)", ", ".join(
            f"h{h}(n_holdout 운영 {p_counts.get(h, '?')} / 후보 {c_counts.get(h, '?')})" for h in tail_skipped), None))

    max_delta: dict[str, float] = {}
    for q in BAND_QUANTILES:
        finite = [d for d in deltas[q] if d is not None]
        max_delta[q] = round(max(finite), 6) if finite else 0.0
    unreadable = sum(d is None for q in BAND_QUANTILES for d in deltas[q])
    over = max(max_delta.values()) > BAND_MAX + EPS or unreadable > 0
    checks.append(Check(f"최대 |Δ|≤{BAND_MAX}", ", ".join(f"{q} {max_delta[q]}" if deltas[q] else f"{q} (판정 없음)" for q in BAND_QUANTILES)
                        + (f" (숫자 아님 {unreadable})" if unreadable else ""), not over))

    worst.sort(key=lambda r: (r["abs_delta"] is not None, -(r["abs_delta"] or 0)))
    return checks, max_delta, worst[:WORST_ROWS]


def classify_band_pair(
    stamp: str,
    prod: dict | None,
    cand: dict | None,
    validate: Callable[[dict], str | None],
) -> PairResult:
    """한 STAMP 의 운영 사본·후보 band payload 로 판정한다. validate(payload) 는 계약 위반 설명 또는 None."""
    if prod is None or cand is None:
        side = "운영 사본" if prod is None else "후보"
        return PairResult(stamp, UNPAIRED, False, [f"{side} 쪽 band 파일이 없다"])

    error = validate(cand)
    diff = band_identity_mismatches(prod, cand)
    if diff:
        result = PairResult(stamp, INPUT_DIFF, False, [f"입력이 다르다({', '.join(diff)}) — 엄격 비교 불가"])
        p_cities, c_cities = {k[0] for k in _band_cells(prod)[0]}, {k[0] for k in _band_cells(cand)[0]}
        overlap = _fraction(len(p_cities & c_cities), max(len(p_cities), len(c_cities)))
        low = f" — {_pct(OVERLAP_FRACTION)} 미만" if overlap < OVERLAP_FRACTION else ""
        result.checks = [
            Check("후보 계약", error or "통과", error is None),
            Check("도시 겹침(보고)", _pct(overlap) + low, None),
        ]
        if error:
            result.verdict, result.reasons = FAIL, [f"후보가 계약을 어겼다: {error}"]
        return result

    checks, max_delta, worst = compare_band(prod, cand)
    checks.append(Check("후보 계약", error or "통과", error is None))
    reasons = [f"{c.name}: {c.detail}" for c in checks if c.ok is False]
    tail_judged, tail_skipped = band_tail_split(prod, cand)
    return PairResult(stamp, FAIL if reasons else PASS, True, reasons, checks, max_delta, worst, tail_judged, tail_skipped)


def parse_since(value: str) -> str:
    """`YYYYMMDDHH` · `YYYYMMDD` · `YYYY-MM-DD` → 10자리 스탬프(이 시각 이후만 비교)."""
    digits = value.replace("-", "")
    if not digits.isdigit() or len(digits) not in (8, 10):
        raise ValueError(f"--since 는 YYYYMMDDHH, YYYYMMDD, YYYY-MM-DD 중 하나여야 한다: {value!r}")
    return digits if len(digits) == 10 else digits + "00"


def summarize(results: list[PairResult], prod_only: int, min_strict: int, quantiles: tuple[str, ...] = QUANTILES) -> dict:
    counts = {v: sum(r.verdict == v for r in results) for v in (PASS, FAIL, INPUT_DIFF, UNPAIRED)}
    strict_pass = sum(r.verdict == PASS and r.strict for r in results)
    max_delta = {q: max((r.max_delta.get(q, 0.0) for r in results if r.strict), default=0.0) for q in quantiles}
    code = 1 if counts[FAIL] else (2 if strict_pass < min_strict else 0)
    return {"pairs": len(results), "counts": counts, "strict_pass": strict_pass, "min_strict_pairs": min_strict,
            "prod_only_stamps": prod_only, "max_delta": max_delta, "exit_code": code}


def _cell(value: object) -> str:
    """Markdown 표 셀 값. `|` 는 열 구분자라 이스케이프하고 줄바꿈은 공백으로 바꾼다(표가 깨지지 않게)."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(results: list[PairResult], summary: dict, product: str = "predictions") -> str:
    c = summary["counts"]
    lines = [
        f"## 후보 ↔ 운영 비교 ({product})",
        "",
        f"짝 {summary['pairs']}개 — PASS {c[PASS]} · FAIL {c[FAIL]} · INPUT-DIFF {c[INPUT_DIFF]} · UNPAIRED {c[UNPAIRED]}"
        f" (후보 시간대의 운영 STAMP 중 후보 없음 {summary['prod_only_stamps']})",
        f"엄격 PASS {summary['strict_pass']} (필요 ≥ {summary['min_strict_pairs']}) · "
        "최대 |Δ| " + ", ".join(f"{q.removeprefix('predicted_')} {'(판정 없음)' if v is None else v}" for q, v in summary["max_delta"].items()),
        *([f"p10/p90 엄격 비교 쌍·지평 {summary['tail_horizons_judged']}개 · n_holdout 이 달라 판정에서 뺀 쌍·지평 {summary['tail_horizons_skipped']}개"
           " (p50 은 모든 지평을 판정한다)"] if "tail_horizons_judged" in summary else []),
        "",
        "| STAMP | 판정 | 엄격 | 사유 |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r.stamp} | {r.verdict} | {'예' if r.strict else '아니오'} | {_cell('; '.join(r.reasons) or '—')} |")
    for r in results:
        if not r.checks or not (r.strict or r.verdict == FAIL):
            continue  # INPUT-DIFF 는 위 표의 사유 한 줄이면 충분하다
        lines += ["", f"### {r.stamp} — {r.verdict}", "", "| 검사 | 값 | 결과 |", "|---|---|---|"]
        for ch in r.checks:
            mark = "—" if ch.ok is None else ("OK" if ch.ok else "FAIL")
            lines.append(f"| {_cell(ch.name)} | {_cell(ch.detail)} | {mark} |")
        if r.worst_rows:
            lines += ["", f"실패 행 상위 {len(r.worst_rows)}:", "", "| name | 필드 | 운영 | 후보 | \\|Δ\\| |", "|---|---|---|---|---|"]
            lines += [f"| {_cell(w['name'])} | {_cell(w['field'])} | {w['prod']} | {w['cand']} | {w['abs_delta']} |" for w in r.worst_rows]
    return "\n".join(lines) + "\n"


# ── 입력 원천 ────────────────────────────────────────────────────────────────
class LocalSource:
    """저장소 트리를 미러한 디렉터리 (`DIR/aq-data/...`)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def list_stamps(self, rel_dir: str, prefix: str) -> list[str]:
        directory = self.root / rel_dir
        if not directory.is_dir():
            return []
        stamps = []
        for path in directory.glob(f"{prefix}_*.json"):
            m = re.fullmatch(rf"{re.escape(prefix)}_(\d{{10}})\.json", path.name)
            if m:
                stamps.append(m.group(1))
        return sorted(stamps)

    def read_json(self, rel_path: str) -> dict | None:
        path = self.root / rel_path
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SourceError(f"{rel_path}: {type(exc).__name__}") from None
        return _json_object(data, rel_path)


class HttpSource:
    """공개 저장소. 토큰을 보내지 않는다. `band_guard.py` 도 이 클래스로 읽는다."""

    def __init__(self, sleep: Callable[[float], None] = time.sleep) -> None:
        self.sleep = sleep

    def _open(self, url: str):
        req = urllib.request.Request(url, headers={"User-Agent": "airlens-data-compare-candidate"})
        return urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S)  # noqa: S310 — 고정 https 호스트

    def _get(self, url: str, what: str):
        """(본문 bytes, Link 헤더). 404 는 (None, ''). 본문이 MAX_RESPONSE_BYTES 를 넘으면 SourceError(재시도하지 않는다)."""
        for attempt in range(1, FETCH_RETRIES + 1):
            try:
                with self._open(url) as resp:
                    body = resp.read(MAX_RESPONSE_BYTES + 1)  # 상한보다 한 바이트 더 읽어 넘었는지만 본다
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise SourceError(f"{what}: 응답 크기 상한 초과")
                    # 크기를 주고 읽으면 Content-Length 보다 짧게 끊겨도 예외가 없다(그냥 읽을 때는 IncompleteRead). http.client 가 Content-Length 로
                    # 센 남은 바이트 `length` 가 0 이 아니면(길이 모름·chunked 는 None) 같은 재시도로 보낸다. 문서화된 속성이 아니라서
                    # test_compare_candidate.py 가 진짜 HTTPResponse 로 동작을 고정한다
                    if resp.length:
                        raise http.client.IncompleteRead(body, resp.length)
                    return body, resp.headers.get("Link") or ""
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None, ""
                if exc.code not in (429, 500, 502, 503, 504) or attempt == FETCH_RETRIES:
                    raise SourceError(f"{what}: HTTPError status={exc.code}") from None
            except (OSError, http.client.HTTPException) as exc:  # URLError·타임아웃·SSLError·IncompleteRead(resp.read 도중) 포함
                if attempt == FETCH_RETRIES:
                    raise SourceError(f"{what}: {type(exc).__name__}") from None
            self.sleep(2.0 * attempt)
        raise AssertionError("unreachable")

    def list_stamps(self, rel_dir: str, prefix: str) -> list[str]:
        stamps: list[str] = []
        url: str | None = f"{TREE_BASE}/{rel_dir}"
        while url:
            body, link = self._get(url, f"{rel_dir} 목록")
            if body is None:
                return stamps
            try:
                entries = json.loads(body)
            except ValueError:
                raise SourceError(f"{rel_dir} 목록: JSON 아님") from None
            if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
                raise SourceError(f"{rel_dir} 목록: 객체 배열이 아님")
            for entry in entries:
                m = re.fullmatch(rf".*/{re.escape(prefix)}_(\d{{10}})\.json", str(entry.get("path", "")))
                if m:
                    stamps.append(m.group(1))
            nxt = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = nxt.group(1) if nxt else None
        return sorted(stamps)

    def read_bytes(self, rel_path: str) -> bytes | None:
        """본문 그대로. 없으면(404) None."""
        return self._get(f"{RESOLVE_BASE}/{rel_path}", rel_path)[0]

    def read_json(self, rel_path: str) -> dict | None:
        body = self.read_bytes(rel_path)
        return None if body is None else parse_json_object(body, rel_path)


def contract_validator() -> Callable[[dict], str | None]:
    """contracts/validate.py 를 파일 경로로 적재한다(hf_publish 와 같은 방식 — 전역 import 경로를 건드리지 않는다)."""
    path = Path(__file__).resolve().parents[2] / "contracts" / "validate.py"
    spec = importlib.util.spec_from_file_location("_airlens_contract_validate", path)
    if spec is None or spec.loader is None:
        raise SourceError("contracts/validate.py 를 적재할 수 없다")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def validate(payload: dict) -> str | None:
        try:
            module.validate_payload(payload, module.resolve_schema(payload))
        except module.ContractError as exc:
            return str(exc)
        return None

    return validate


def run(source: Source, validate: Callable[[dict], str | None], since: str | None, min_strict: int) -> tuple[list[PairResult], dict]:
    cand_stamps = [s for s in source.list_stamps(CAND_GRID_DIR, "grid") if since is None or s >= since]
    results: list[PairResult] = []
    for stamp in cand_stamps:
        prod = source.read_json(f"{PROD_GRID_DIR}/grid_{stamp}.json")
        cand = source.read_json(f"{CAND_GRID_DIR}/grid_{stamp}.json")
        prod_q = cand_q = None
        if prod is not None and cand is not None and not strict_mismatches(prod, cand):
            prod_q = source.read_json(f"{PROD_QUALITY_DIR}/quality_{stamp}.json")
            cand_q = source.read_json(f"{CAND_QUALITY_DIR}/quality_{stamp}.json")
        results.append(classify_pair(stamp, prod, cand, prod_q, cand_q, validate))

    return results, summarize(results, _prod_only_count(source, PROD_GRID_DIR, "grid", cand_stamps), min_strict)


def _prod_only_count(source: Source, rel_dir: str, prefix: str, cand_stamps: list[str]) -> int:
    """후보가 돈 시간대(첫 후보 STAMP 이후)에 운영만 있는 STAMP 수 — 후보가 건너뛴 슬롯."""
    if not cand_stamps:
        return 0
    first, seen = cand_stamps[0], set(cand_stamps)
    return len([s for s in source.list_stamps(rel_dir, prefix) if s >= first and s not in seen])


def run_band(source: Source, validate: Callable[[dict], str | None], since: str | None, min_strict: int) -> tuple[list[PairResult], dict]:
    cand_stamps = [s for s in source.list_stamps(CAND_BAND_DIR, "band") if since is None or s >= since]
    results = [classify_band_pair(stamp, source.read_json(f"{PROD_BAND_DIR}/band_{stamp}.json"),
                                  source.read_json(f"{CAND_BAND_DIR}/band_{stamp}.json"), validate)
               for stamp in cand_stamps]
    summary = summarize(results, _prod_only_count(source, PROD_BAND_DIR, "band", cand_stamps), min_strict, BAND_QUANTILES)
    summary["tail_horizons_judged"] = sum(len(r.tail_judged) for r in results)  # 지평을 채우는 것은 엄격 짝(PASS·FAIL)뿐이다
    summary["tail_horizons_skipped"] = sum(len(r.tail_skipped) for r in results)
    if summary["tail_horizons_judged"] == 0:  # 판정한 p10/p90 셀이 없으면 최대 |Δ| 0 은 "같다" 가 아니라 "보지 않았다" 이다
        summary["max_delta"].update({q: None for q in BAND_QUANTILES if q != "p50"})
    return results, summary


def main(argv: list[str] | None = None, validate: Callable[[dict], str | None] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--product", default="predictions", choices=["predictions", "band"])
    parser.add_argument("--local", type=Path, help="저장소 트리를 미러한 디렉터리 (없으면 공개 저장소를 읽는다)")
    parser.add_argument("--since", help="이 시각 이후 STAMP 만 (YYYYMMDDHH · YYYYMMDD · YYYY-MM-DD)")
    parser.add_argument("--min-strict-pairs", type=int, default=None,
                        help="엄격 PASS 최소 수 (기본: " + ", ".join(f"{k} {v}" for k, v in MIN_STRICT_PAIRS_BY_PRODUCT.items()) + ")")
    parser.add_argument("--json", action="store_true", help="Markdown 대신 JSON 을 stdout 으로")
    args = parser.parse_args(argv)

    try:
        since = parse_since(args.since) if args.since else None
        source: Source = LocalSource(args.local) if args.local else HttpSource()
        min_strict = args.min_strict_pairs if args.min_strict_pairs is not None else MIN_STRICT_PAIRS_BY_PRODUCT[args.product]
        runner = run_band if args.product == "band" else run
        results, summary = runner(source, validate or contract_validator(), since, min_strict)
    except (ValueError, SourceError) as exc:
        print(f"compare_candidate: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"product": args.product, "summary": summary, "pairs": [asdict(r) for r in results]},
                         ensure_ascii=False, indent=2))
    else:
        report = render_markdown(results, summary, args.product)
        print(report)
        path = os.environ.get("GITHUB_STEP_SUMMARY")
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(report)
    return summary["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
