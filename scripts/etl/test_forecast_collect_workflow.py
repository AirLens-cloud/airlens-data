"""forecast-collect.yml 의 첫 잡 `guard` 와 수집 잡 `collect-forecast` 의 연결을 고정한다 (AAA, YAML 파싱 — 네트워크 0).

**왜 이 파일이 있나.** 이 워크플로는 E2 VM 이 6시간마다 `workflow_dispatch` 로 정시에 시작시키고, GitHub `schedule` 은 백업으로 남아
있다. 같은 칸을 두 번 수집하지 않도록 첫 잡 `guard` 가 공개 `forecast.json` 만 읽어 `skip` 을 내고, 수집 잡은 그 출력을 본다.
이 연결은 조용히 깨진다 — 어긋나도 워크플로는 멀쩡히 돌고, 그저 (1) 가드가 토큰을 들고 있게 되거나, (2) 건너뛰기가 안 먹어 같은 예보를
두 번 수집하거나, (3) 가드가 죽었을 때 수집까지 같이 죽어 예보가 멈춘다. 이 워크플로는 PR 에서 돌지 않으니(cron·수동) 머지 뒤 몇 시간
지나야 보인다. 이 시험이 그 경계를 `pytest scripts` 로 못박는다.

규칙
  1. `guard` 는 시크릿·environment 가 없고 `contents: read` 뿐이며, 체크아웃은 해시로 고정하고 자격 증명을 남기지 않는다.
  2. `guard` 는 아무것도 설치하지 않는다 — 그래서 가드 스크립트는 표준 라이브러리만 import 해야 한다.
  3. `collect-forecast` 는 `needs: guard` 이고, 조건은 {저장소 확인, `skip != 'true'`, `!cancelled()`} 셋이다. 끝의 `!cancelled()` 가 기본
     조건 `success()` 를 대신해서 guard 잡이 실패해도 수집은 돈다(건너뛰기는 최적화일 뿐이다).
  4. `force` 는 boolean 입력이고 기본값이 false 다. 가드 잡의 환경 변수는 `true`/`false` 두 글자만 만든다(schedule 에는 입력이 없다).
  5. 백업 `schedule`·`concurrency`(같은 그룹으로 직렬화 — 늦은 run 의 가드가 먼저 온 run 의 수집이 끝난 뒤에 돈다)·워크플로 이름
     (band-forecast.yml 이 `workflow_run` 으로 이름을 보고 따라온다)은 그대로다.

한계: 조건식은 GitHub 가 평가하므로 실제 평가는 못 본다. 선언된 조건 집합과 철자만 본다.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/forecast-collect.yml"
BAND_WORKFLOW = ROOT / ".github/workflows/band-forecast.yml"
GUARD_SCRIPT = ROOT / "scripts/etl/forecast_slot_guard.py"

REPO_CLAUSE = "github.repository == 'AirLens-cloud/airlens-data'"
GUARD_RUN = ["set -euo pipefail", 'python3 scripts/etl/forecast_slot_guard.py --force "$FORCE"']
CHECKOUT_PIN = re.compile(r"actions/checkout@([0-9a-f]{40})")


def _load(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), f"{path.name}: 최상위가 매핑이 아니다"
    return doc


def _triggers(doc: dict) -> dict:
    # PyYAML 은 YAML 1.1 이라 bare `on` 을 True 로 읽는다(GitHub 의 파서는 문자열 "on").
    triggers = doc.get("on", doc.get(True))
    assert isinstance(triggers, dict), "트리거(on)를 못 찾았다 — 파싱이 깨졌다(vacuous pass 방지)"
    return triggers


def _job(name: str) -> dict:
    jobs = _load(WORKFLOW)["jobs"]
    assert name in jobs, f"잡 '{name}' 이 없다 — 잡 목록: {list(jobs)}"
    return jobs[name]


def _guard_step() -> dict:
    steps = [s for s in _job("guard")["steps"] if s.get("id") == "guard"]
    assert len(steps) == 1, "id 가 guard 인 스텝이 정확히 하나여야 한다"
    return steps[0]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert not any(isinstance(n, ast.ImportFrom) and n.level for n in ast.walk(tree)), "상대 import 가 있다"
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


# ── 잡 구성 ──────────────────────────────────────────────────────────────────
def test_the_workflow_has_exactly_the_guard_job_and_the_collect_job():
    # Arrange / Act
    jobs = _load(WORKFLOW)["jobs"]

    # Assert — 새 잡이 슬쩍 늘어 시크릿 근처에 서는 일을 막는다
    assert set(jobs) == {"guard", "collect-forecast"}


# ── guard 잡 ─────────────────────────────────────────────────────────────────
def test_the_guard_job_holds_no_secret_token_or_environment():
    # Arrange
    guard = _job("guard")

    # Act
    serialized = json.dumps(guard)

    # Assert — 공개 URL 만 읽는 잡이다. 토큰이 들어올 자리를 아예 두지 않는다
    assert "environment" not in guard and "env" not in guard
    assert not re.search(r"secrets|token", serialized, re.I), "guard 잡이 시크릿·토큰을 언급한다"


def test_the_guard_job_can_only_read_the_repository_contents():
    # Act
    guard = _job("guard")

    # Assert
    assert guard["permissions"] == {"contents": "read"}


def test_the_guard_job_is_bounded_and_limited_to_this_repository():
    # Act
    guard = _job("guard")

    # Assert — 6시간마다 도는 잡이라 멈춰도 5분에 끝난다
    assert guard["runs-on"] == "ubuntu-latest"
    assert guard["timeout-minutes"] == 5
    assert guard["if"] == REPO_CLAUSE


def test_the_guard_checkout_is_pinned_to_a_commit_and_keeps_no_credentials():
    # Arrange
    guard_steps = _job("guard")["steps"]
    collect_steps = _job("collect-forecast")["steps"]

    # Act
    guard_checkout = guard_steps[0]
    guard_pin = CHECKOUT_PIN.fullmatch(guard_checkout.get("uses", ""))
    collect_pin = CHECKOUT_PIN.fullmatch(collect_steps[0].get("uses", ""))

    # Assert — 태그가 아니라 40자 커밋 해시로 고정하고, 같은 워크플로의 다른 체크아웃과 같은 해시다
    assert guard_pin and collect_pin, "체크아웃이 40자 커밋 해시로 고정돼 있지 않다"
    assert guard_pin.group(1) == collect_pin.group(1)
    assert guard_checkout["with"] == {"persist-credentials": False}


def test_the_guard_job_runs_exactly_the_guard_script_with_the_force_input_as_an_argument():
    # Arrange
    guard = _job("guard")
    step = _guard_step()

    # Act
    run_lines = [line.strip() for line in step["run"].splitlines() if line.strip()]

    # Assert
    assert len(guard["steps"]) == 2 and guard["steps"][0]["uses"].startswith("actions/checkout@") and guard["steps"][1] == step
    assert run_lines == GUARD_RUN
    assert step["env"] == {"FORCE": "${{ inputs.force || false }}"}  # schedule 에는 입력이 없다 — `|| false` 가 true/false 두 글자로 만든다
    assert (ROOT / "scripts/etl/forecast_slot_guard.py").is_file()
    assert "${{" not in step["run"], "run 안에 식을 직접 쓰면 입력이 셸 코드가 된다 — env 로만 넘긴다"


def test_a_failing_guard_stays_visible_instead_of_being_swallowed():
    # Arrange
    guard = _job("guard")

    # Act
    swallowed = [s.get("name") for s in guard["steps"] if "continue-on-error" in s] + (["job"] if "continue-on-error" in guard else [])

    # Assert — 가드가 죽어도 수집은 돈다(`!cancelled()`). 그렇다고 가드의 실패가 안 보이면 건너뛰기가 영영 안 먹는 것을 아무도 모른다
    assert swallowed == []


def test_the_guard_job_exports_the_skip_flag_of_its_guard_step():
    # Arrange
    outputs = _job("guard")["outputs"]
    step = _guard_step()

    # Act
    reference = re.fullmatch(r"\$\{\{ steps\.([\w-]+)\.outputs\.skip \}\}", outputs["skip"])

    # Assert — 출력 이름·스텝 id 가 어긋나면 skip 이 늘 비어 있어 수집 잡이 건너뛰지 않는다
    assert set(outputs) == {"skip"}
    assert reference and reference.group(1) == step["id"]


def test_the_guard_job_installs_nothing_so_the_script_may_import_only_the_standard_library():
    # Arrange
    guard = _job("guard")
    commands = "\n".join(s.get("run", "") for s in guard["steps"])
    actions = [s["uses"] for s in guard["steps"] if "uses" in s]

    # Act
    imported = _imported_modules(GUARD_SCRIPT)
    outside_stdlib = imported - set(sys.stdlib_module_names)

    # Assert
    assert not re.search(r"\b(pip3?|pipx|uv|poetry|conda|npm|apt(-get)?)\b", commands) and not any("setup-" in a for a in actions)
    assert {"json", "urllib", "datetime"} <= imported, "import 목록을 못 읽었다 — 파싱이 깨졌다(vacuous pass 방지)"
    assert outside_stdlib == set(), f"설치 없이 도는 잡에서 표준 라이브러리 밖 모듈을 쓴다: {sorted(outside_stdlib)}"


# ── collect-forecast 잡 ──────────────────────────────────────────────────────
def test_the_collect_job_waits_for_the_guard_job():
    # Act
    needs = _job("collect-forecast")["needs"]

    # Assert
    assert ([needs] if isinstance(needs, str) else needs) == ["guard"]


def test_the_collect_condition_skips_on_a_handled_slot_but_survives_a_failed_guard():
    # Arrange
    condition = _job("collect-forecast")["if"]

    # Act
    clauses = [c.strip() for c in condition.split("&&")]

    # Assert — 세 조건뿐이다. `!cancelled()` 가 기본 조건(success())을 대신해 guard 가 실패해도 수집이 돈다
    assert isinstance(condition, str)
    assert set(clauses) == {REPO_CLAUSE, "needs.guard.outputs.skip != 'true'", "!cancelled()"} and len(clauses) == 3
    assert clauses[-1] == "!cancelled()"  # `!` 로 시작하는 식은 YAML 태그로 읽힌다 — 끝에 둔다
    assert not condition.lstrip().startswith("!")


def test_the_collect_job_does_not_ask_for_the_guard_result_directly():
    # Arrange
    condition = _job("collect-forecast")["if"]

    # Assert — `needs.guard.result == 'success'` 를 요구하면 guard 가 죽을 때 수집도 막힌다
    assert "needs.guard.result" not in condition and "success()" not in condition


# ── 트리거·동시성·이름 ───────────────────────────────────────────────────────
def test_force_is_a_boolean_input_that_defaults_to_false():
    # Arrange
    dispatch = _triggers(_load(WORKFLOW))["workflow_dispatch"]

    # Act
    force = dispatch["inputs"]["force"]

    # Assert — 기본값이 true 면 VM 의 정시 dispatch 가 가드를 늘 건너뛴다
    assert set(dispatch["inputs"]) == {"force"}
    assert force["type"] == "boolean" and force["default"] is False
    assert isinstance(force["description"], str) and force["description"].strip()
    assert not force.get("required")


def test_the_scheduled_backup_is_kept_as_it_was():
    # Act
    triggers = _triggers(_load(WORKFLOW))

    # Assert — 정시 dispatch 가 죽어도 늦게나마 도는 길이다
    assert triggers["schedule"] == [{"cron": "15 */6 * * *"}]
    assert set(triggers) == {"schedule", "workflow_dispatch"}


def test_runs_stay_serialized_so_a_late_guard_sees_the_earlier_collection():
    # Act
    doc = _load(WORKFLOW)

    # Assert — 병렬로 돌면 두 run 의 가드가 동시에 "아직 수집 없음" 을 보고 둘 다 수집한다. 취소도 안 된다(수집 중간에 끊기면 안 된다)
    assert doc["concurrency"] == {"group": "forecast-collect", "cancel-in-progress": False}


def test_the_workflow_permissions_stay_read_only():
    # Act
    doc = _load(WORKFLOW)

    # Assert
    assert doc["permissions"] == {"contents": "read"}


def test_band_forecast_still_follows_this_workflow_by_its_name():
    # Arrange
    name = _load(WORKFLOW)["name"]

    # Act
    follows = _triggers(_load(BAND_WORKFLOW))["workflow_run"]

    # Assert — 건너뛴 run 도 guard 잡이 성공해서 success 로 끝나, band 의 workflow_run 은 그 뒤에도 돈다(band 쪽 가드가 중복을 거른다).
    # 이름이 어긋나면 band 가 예보 수집을 더는 따라가지 않는다.
    assert follows["workflows"] == [name]
    assert follows["types"] == ["completed"]
