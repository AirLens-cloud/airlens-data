#!/usr/bin/env python3
"""발행된 예측물(`grid_latest.json` + `predictions/history/` + 예측 밴드)의 신선도를 프로브한다.

**성공한 run 이 아니라 발행물이 판정 대상이다.** 발행 워크플로가 스케줄에서 밀리거나 비활성이 되면 run 은
실패하는 것이 아니라 *생성되지 않는다* — Actions 탭에는 빨간 줄이 하나도 남지 않고 마지막 성공 run 은 계속
초록이다. 결손을 알아채는 유일한 길이 "누가 로그를 열어보는 것" 이면 그건 감시가 아니다.

그래서 감시를 **발행 워크플로 밖**으로 뺀다. 별도 워크플로가 HF 발행물을 직접 GET 하고, 발행 워크플로가
켜져 있는지는 GitHub API 로 읽는다.

판정:
- **stale_latest** — `grid_latest.json.generated_at` 이 허용 나이를 넘음
- **history_gap** — 최신 history 슬롯이 발행 주기의 2배(= 예상 슬롯 2개 이상 연속 결손)와 허용 나이 중
  긴 쪽을 넘게 오래됨. `grid_latest` 만 보면 놓치는 경로다: latest 는 갱신됐는데 history 적재만 죽은
  상태가 있을 수 있다. 한 run 이 둘을 함께 쓰므로 스케줄 지연은 둘을 같이 늦춘다 — 슬롯 판정을 latest
  판정보다 먼저 열지 않는다.
- **history_empty** — history 에 슬롯 파일이 하나도 없음
- **malformed** — 발행물에 읽을 수 있는 `generated_at` 이 없음
- **stale_band** — 예측 밴드(`forecast-band/latest.json`)가 12시간을 넘게 오래됨. 웹이 같은 값(12시간,
  `ageMs > STALE`)으로 "오래됨" 배지를 달기 때문에 이 값을 따로 정하지 않고 맞춘다.
- **band_missing** — 예측 밴드 파일이 없음(404). 프로브 고장이 아니라 발행물의 결손이다.
- **workflow_inactive** — 감시 대상 워크플로의 상태가 `active` 가 아님(비활성·삭제 등). 이 레포에 아직
  없는 워크플로(404)는 "미배포" 로 지표에만 적고 문제로 세지 않는다.
- **probe_error** — 발행물이나 워크플로 상태를 읽지 못함 (HF 장애·경로 변경·권한). 조용히 통과시키면
  프로브 자체의 고장이 "정상" 으로 위장된다.

`consecutive_missed` / `coverage_24h` 는 판정이 아니라 **맥락**이다 — 이슈 본문에
넣어서 "지금 막 죽었나, 원래 이 정도로 새고 있었나" 를 한 줄로 구분시킨다.

환경변수(전부 선택):
  PREDICTIONS_MAX_AGE_HOURS   grid_latest 허용 나이 (기본 8 — `DEFAULT_MAX_AGE_HOURS`)
  PREDICTIONS_CADENCE_HOURS   발행 주기 (기본 3)
  BAND_MAX_AGE_HOURS          예측 밴드 허용 나이 (기본 12 — 웹의 오래됨 기준과 같다)
  WATCHED_WORKFLOWS           상태를 볼 워크플로 파일명, 쉼표 구분 (기본 `ml-predict.yml,band-forecast.yml`).
                              비우면 상태 점검을 건너뛴다. 발행 워크플로를 일부러 끄는 작업(롤백 등)을 할 때는
                              여기서 그 파일을 뺀다.
  GITHUB_REPOSITORY, GH_TOKEN 워크플로 상태 조회용 (Actions 가 채운다. 토큰이 없으면 익명 조회)
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

DATASET = "Robeedau/airlens-live"
_RESOLVE = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
LATEST_URL = f"{_RESOLVE}/aq-data/predictions/grid_latest.json"
BAND_URL = f"{_RESOLVE}/aq-data/forecast-band/latest.json"
HISTORY_TREE_URL = (
    f"https://huggingface.co/api/datasets/{DATASET}/tree/main"
    "/aq-data/predictions/history"
)
GITHUB_API = "https://api.github.com"

BAND_MAX_AGE = timedelta(hours=12)  # airlens-web 의 오래됨 기준(12h)과 같다
DEFAULT_WATCHED = "ml-predict.yml,band-forecast.yml"
# grid_latest 허용 나이(시간). 발행 주기는 3h 지만 GitHub 스케줄 지연으로 실제 간격이 중앙값 5~6.5h 라 4h 면 지연만으로
# 경보가 열린다. 정시 호출이 ml 예측까지 오면 4h 로 되돌린다. 워크플로의 dispatch 기본값·스케줄 대체값과 같아야 한다.
DEFAULT_MAX_AGE_HOURS = "8"

# `grid_YYYYMMDDHH.json` — 슬롯 시각은 관측 슬롯에서 오므로 고정 격자(0/3/6/9…)에
# 정렬되지 않는다. 실측 슬롯: 08,14,18,23 / 05,11,16,21,23 / 05,11. 따라서 "격자상의
# 빈 칸" 을 세는 판정은 쓸 수 없고, **슬롯 간 간격**으로 결손을 센다.
SLOT_RE = re.compile(r"grid_(\d{10})\.json$")
WORKFLOW_FILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.ya?ml")
REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


class BandFetch(NamedTuple):
    """예측 밴드 조회 결과. doc 이 None 이고 missing 도 False 면 조회 실패(오류는 fetch_errors 로 따로 온다)."""

    doc: dict | None
    missing: bool


def parse_iso8601(value: str | None) -> datetime | None:
    """`...Z` 형태를 aware datetime 으로. 파싱 불가면 None."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def parse_slot(path: str) -> datetime | None:
    """`.../grid_2026090311.json` → 2026-09-03T11:00Z. 규격 밖 이름이면 None."""
    match = SLOT_RE.search(path)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_workflow_list(raw: str) -> tuple[list[str], list[str]]:
    """`a.yml, b.yml` → (올바른 파일명들, 올바르지 않은 항목들). 중복은 한 번만."""
    names: list[str] = []
    bad: list[str] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if not WORKFLOW_FILE_RE.fullmatch(item):
            bad.append(item)
        elif item not in names:
            names.append(item)
    return names, bad


def classify(
    latest: dict | None,
    slots: list[datetime],
    now: datetime,
    max_age: timedelta,
    cadence: timedelta,
    fetch_errors: list[str] | None = None,
    *,
    band: BandFetch | None = None,
    band_max_age: timedelta = BAND_MAX_AGE,
    workflow_states: dict[str, str | None] | None = None,
) -> dict:
    """발행물 상태 판정. `problems` 가 비면 정상.

    `band` 가 None 이면 예측 밴드를 점검하지 않는다. `workflow_states` 는 파일명 → GitHub 의 워크플로 state
    (`active` 등), 이 레포에 없으면 None(미배포). 비어 있거나 None 이면 상태 점검을 건너뛴다.
    `metrics` 는 판정에 쓰이지 않는 맥락 수치다 — 이슈 본문 전용.
    """
    problems: list[str] = list(fetch_errors or [])
    metrics: dict[str, object] = {
        "latest_age_h": None,
        "newest_slot_age_h": None,
        "consecutive_missed": None,
        "coverage_24h": len([s for s in slots if now - s <= timedelta(hours=24)]),
        "expected_24h": int(timedelta(hours=24) / cadence),
        "generated_at": None,
        "newest_slot": None,
        "band_age_h": None,
        "band_generated_at": None,
        "workflows": dict(workflow_states or {}),
    }

    if latest is not None:
        generated_at = parse_iso8601(latest.get("generated_at"))
        if generated_at is None:
            problems.append("malformed — `grid_latest.json` 에 읽을 수 있는 `generated_at` 이 없다")
        else:
            age_h = (now - generated_at).total_seconds() / 3600
            metrics["latest_age_h"] = round(age_h, 1)
            metrics["generated_at"] = generated_at.strftime("%Y-%m-%dT%H:%M:%SZ")
            if age_h > max_age.total_seconds() / 3600:
                problems.append(
                    f"stale_latest — `grid_latest.json` 이 {age_h:.1f}h 전 발행분 "
                    f"(허용 {max_age.total_seconds() / 3600:.0f}h)"
                )

    if slots:
        newest = max(slots)
        gap_h = (now - newest).total_seconds() / 3600
        cadence_h = cadence.total_seconds() / 3600
        metrics["newest_slot_age_h"] = round(gap_h, 1)
        metrics["newest_slot"] = newest.strftime("%Y-%m-%dT%HZ")
        # 간격 / 주기 - 1 = 건너뛴 슬롯 수. 정시 발행이면 0.
        metrics["consecutive_missed"] = max(0, int(gap_h // cadence_h) - 1)
        gap_limit_h = max(2 * cadence_h, max_age.total_seconds() / 3600)  # latest 판정보다 먼저 열지 않는다
        if gap_h > gap_limit_h:
            problems.append(
                f"history_gap — 최신 슬롯이 {gap_h:.1f}h 전 "
                f"(허용 {gap_limit_h:.0f}h, 발행 주기 {cadence_h:.0f}h 기준 {metrics['consecutive_missed']}슬롯 연속 결손)"
            )
    elif not fetch_errors:
        problems.append("history_empty — `predictions/history/` 에 슬롯 파일이 하나도 없다")

    if band is not None:
        if band.missing:
            problems.append("band_missing — `forecast-band/latest.json` 이 없다 (404)")
        elif band.doc is not None:
            band_at = parse_iso8601(band.doc.get("generated_at"))
            if band_at is None:
                problems.append("malformed — `forecast-band/latest.json` 에 읽을 수 있는 `generated_at` 이 없다")
            else:
                band_age_h = (now - band_at).total_seconds() / 3600
                metrics["band_age_h"] = round(band_age_h, 1)
                metrics["band_generated_at"] = band_at.strftime("%Y-%m-%dT%H:%M:%SZ")
                if band_age_h > band_max_age.total_seconds() / 3600:  # 웹과 같은 `age > STALE` (경계값은 정상)
                    problems.append(
                        f"stale_band — `forecast-band/latest.json` 이 {band_age_h:.1f}h 전 발행분 "
                        f"(허용 {band_max_age.total_seconds() / 3600:.0f}h)"
                    )

    for name, state in sorted((workflow_states or {}).items()):
        if state is None:
            continue  # 아직 이 레포에 없는 워크플로 — 문제가 아니다(지표에 "미배포" 로만 적는다)
        if state != "active":
            problems.append(f"workflow_inactive — `{name}` 의 상태가 `{state}` (active 아님)")

    return {"problems": problems, "metrics": metrics}


def render_report(verdict: dict, now: datetime) -> str:
    """이슈 본문 겸 job summary. 매 프로브마다 이 본문으로 **덮어쓴다** (댓글 누적 금지)."""
    metrics = verdict["metrics"]
    problems = verdict["problems"]

    def show(value: object, unit: str = "") -> str:
        return "—" if value is None else f"{value}{unit}"

    lines = ["## 예측 발행 신선도 프로브", ""]
    if problems:
        lines.append(f"🔴 **{len(problems)}건 이상** — 예측 발행(`ml-predict`·`band-forecast`)이 멈췄거나 새고 있다")
        lines.append("")
        for problem in problems:
            lines.append(f"- {problem}")
    else:
        lines.append("✅ 발행 정상")

    lines += [
        "",
        "| 지표 | 값 |",
        "|---|---|",
        f"| `grid_latest.generated_at` | {show(metrics['generated_at'])} |",
        f"| latest 나이 | {show(metrics['latest_age_h'], ' h')} |",
        f"| 최신 history 슬롯 | {show(metrics['newest_slot'])} |",
        f"| 슬롯 나이 | {show(metrics['newest_slot_age_h'], ' h')} |",
        f"| 연속 결손 슬롯 | {show(metrics['consecutive_missed'])} |",
        f"| 24h 슬롯 커버리지 | {metrics['coverage_24h']} / {metrics['expected_24h']} |",
        f"| 예측 밴드 `generated_at` | {show(metrics['band_generated_at'])} |",
        f"| 밴드 나이 | {show(metrics['band_age_h'], ' h')} |",
    ]
    for name, state in sorted(metrics["workflows"].items()):
        lines.append(f"| 워크플로 `{name}` | {'미배포' if state is None else state} |")
    lines += [
        "",
        f"프로브 시각 `{now:%Y-%m-%dT%H:%M:%SZ}` · 소스 `{DATASET}` `aq-data/predictions/` · `aq-data/forecast-band/`",
    ]

    if problems:
        lines += [
            "",
            "**점검 순서** — 발행 워크플로가 스케줄에서 밀리거나 꺼져 있으면 run 이 실패하는 것이 아니라",
            "*생성되지 않아* Actions 탭이 조용하다.",
            "",
            "1. Actions 탭에서 `ml-predict`·`band-forecast` 의 최근 run 이 있는지, 스케줄이 밀리지 않았는지 확인",
            "2. 워크플로가 비활성(disabled)이면 다시 켠다 — 오래 활동이 없어 꺼진 경우는 keepalive 가 되살린다",
            "3. run 이 있는데도 결손이면 그 run 의 로그를 본다 (관측 슬롯 신선도 게이트 4h 는 fail-loud)",
            "",
            "복구되면 이 이슈는 다음 프로브에서 자동으로 닫힌다.",
        ]

    return "\n".join(lines) + "\n"


def _get_json(url: str, timeout: int) -> dict:
    """JSON 객체 하나를 읽는다. 객체가 아니면 ValueError."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 — 고정 https URL
        doc = json.loads(resp.read())
    if not isinstance(doc, dict):
        raise ValueError("JSON 객체가 아니다")
    return doc


def fetch_latest(url: str, timeout: int) -> tuple[dict | None, str | None]:
    """(문서, 에러). 에러는 판정에 쓰이므로 예외로 새어나가게 두지 않는다."""
    try:
        return _get_json(url, timeout), None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        return None, f"probe_error — `grid_latest.json` 을 읽지 못했다 ({type(exc).__name__}: {exc})"


def fetch_band(url: str, timeout: int) -> tuple[BandFetch, str | None]:
    """(BandFetch, 에러). 404 는 에러가 아니라 `missing` 이다 — 발행물의 결손이지 프로브의 고장이 아니다."""
    try:
        return BandFetch(_get_json(url, timeout), False), None
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return BandFetch(None, True), None
        return BandFetch(None, False), f"probe_error — `forecast-band/latest.json` 을 읽지 못했다 (HTTPError: {exc.code})"
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        return BandFetch(None, False), f"probe_error — `forecast-band/latest.json` 을 읽지 못했다 ({type(exc).__name__}: {exc})"


def fetch_slots(url: str, timeout: int) -> tuple[list[datetime], str | None]:
    """history 디렉터리의 슬롯 시각 목록. 페이지네이션(`Link: rel=next`) 추적."""
    slots: list[datetime] = []
    next_url: str | None = url
    try:
        while next_url:
            req = urllib.request.Request(next_url, headers={"User-Agent": "airlens-predictions-probe"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — 고정 https URL
                for entry in json.load(resp):
                    slot = parse_slot(entry.get("path", ""))
                    if slot is not None:
                        slots.append(slot)
                link = resp.headers.get("Link") or ""
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            next_url = match.group(1) if match else None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        return slots, f"probe_error — `predictions/history/` 를 나열하지 못했다 ({type(exc).__name__}: {exc})"
    return slots, None


def fetch_workflow_state(name: str, repo: str, token: str | None, timeout: int) -> tuple[str | None, str | None]:
    """(state, 에러). 404(이 레포에 없는 워크플로)는 (None, None) = 미배포.

    토큰은 첫 요청 헤더에만 달고(리다이렉트에는 따라가지 않는다), 오류에는 클래스와 HTTP 상태만 남긴다.
    """
    req = urllib.request.Request(
        f"{GITHUB_API}/repos/{repo}/actions/workflows/{name}",
        headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "airlens-predictions-probe"},
    )
    if token:
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — 고정 https URL
            doc = json.loads(resp.read())
        state = doc.get("state") if isinstance(doc, dict) else None
        if not isinstance(state, str):
            return None, f"probe_error — 워크플로 `{name}` 의 상태를 읽지 못했다 (state 필드 없음)"
        return state, None
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None, None
        return None, f"probe_error — 워크플로 `{name}` 의 상태를 읽지 못했다 (HTTPError: {exc.code})"
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        return None, f"probe_error — 워크플로 `{name}` 의 상태를 읽지 못했다 ({type(exc).__name__})"


def fetch_workflow_states(
    names: list[str], repo: str, token: str | None, timeout: int
) -> tuple[dict[str, str | None], list[str]]:
    """감시 대상 워크플로의 (파일명 → state, 에러들). 조회에 실패한 파일은 사전에 넣지 않는다."""
    states: dict[str, str | None] = {}
    errors: list[str] = []
    for name in names:
        state, error = fetch_workflow_state(name, repo, token, timeout)
        if error:
            errors.append(error)
        else:
            states[name] = state
    return states, errors


def main() -> int:
    max_age = timedelta(hours=float(os.environ.get("PREDICTIONS_MAX_AGE_HOURS", DEFAULT_MAX_AGE_HOURS)))
    cadence = timedelta(hours=float(os.environ.get("PREDICTIONS_CADENCE_HOURS", "3")))
    band_max_age = timedelta(hours=float(os.environ.get("BAND_MAX_AGE_HOURS", "12")))
    timeout = int(os.environ.get("PREDICTIONS_PROBE_TIMEOUT", "30"))
    now = datetime.now(timezone.utc)

    latest, latest_err = fetch_latest(os.environ.get("PREDICTIONS_LATEST_URL", LATEST_URL), timeout)
    slots, slots_err = fetch_slots(os.environ.get("PREDICTIONS_HISTORY_URL", HISTORY_TREE_URL), timeout)
    band, band_err = fetch_band(os.environ.get("BAND_LATEST_URL", BAND_URL), timeout)

    errors = [e for e in (latest_err, slots_err, band_err) if e]
    names, bad_names = parse_workflow_list(os.environ.get("WATCHED_WORKFLOWS", DEFAULT_WATCHED))
    if bad_names:
        errors.append(f"probe_error — WATCHED_WORKFLOWS 에 올바르지 않은 파일명이 있다 ({len(bad_names)}개 무시)")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    workflow_states: dict[str, str | None] | None = None
    if names and REPO_RE.fullmatch(repo):
        workflow_states, state_errors = fetch_workflow_states(names, repo, os.environ.get("GH_TOKEN"), timeout)
        errors += state_errors

    verdict = classify(
        latest, slots, now, max_age, cadence,
        fetch_errors=errors,
        band=band,
        band_max_age=band_max_age,
        workflow_states=workflow_states,
    )
    report = render_report(verdict, now)
    print(report)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(report)

    body_path = os.environ.get("PREDICTIONS_REPORT_PATH")
    if body_path:
        with open(body_path, "w", encoding="utf-8") as f:
            f.write(report)

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"healthy={'false' if verdict['problems'] else 'true'}\n")

    if not verdict["problems"]:
        return 0

    print(f"발행 신선도 이상 {len(verdict['problems'])}건", file=sys.stderr)

    # 경보 채널은 **이슈**다 — 프로브 스텝을 매번 red 로 만들면 (결손이 상시인
    # 때는 거의 매번이다) Actions 탭이 빨간 벽이 되고 아무도 안 본다. 그래서
    # 신선도 이상 자체로는 통과시키고, 프로브가 *읽지 못한* 경우(probe_error)만
    # red 로 올린다 — 그건 경보 채널 자체의 고장이라 이슈로도 안 잡힌다.
    if any(p.startswith("probe_error") for p in verdict["problems"]):
        return 1
    if os.environ.get("FAIL_ON_STALE", "false").lower() == "true":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
