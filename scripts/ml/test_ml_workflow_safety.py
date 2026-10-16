"""ML 추론 워크플로 가드 시험 (AAA, YAML 파싱 — 네트워크 0).

이 레포는 **공개**고 추론 코드는 **비공개 wheel** 이다. 워크플로를 잘못 고치면 (1) PR 코드가 비공개 wheel 읽기 토큰을
만지거나 (2) 쓰기 토큰이 의존성 수십 개를 실행하는 잡에 들어가거나 (3) wheel·venv 가 캐시에 올라 PR run 이 복원하거나
(4) 후보 모드가 운영 경로를 건드리는 사고가 난다. 이 파일은 그 경계를 YAML 구조로 고정한다 — 사람이 리뷰에서 놓쳐도
`pytest scripts` 가 막는다.

구성
  A. 전체 워크플로 공통 (PR 트리거 ↔ 비공개 시크릿, `pull_request_target`, 캐시 경로, SHA 고정)
  B. ML 워크플로 공통 (repo 가드, 시간 제한, 권한, 동시성, 발행 모드, 시크릿 위치)
  C. `ml-predict.yml` 전용 (2잡 분리, 핀, run_entry 경유, 설치 플래그, artifact 내용)
  D. 환경 고정 파일 (constraints ↔ 해시 고정본, publish 잡의 해시 고정본)
  E. 공개 위생 (로컬 경로·운영 환경 이름·내부 문서 번호·토큰 모양·IP 가 새 파일에 없고, 저장소 id 는 허용목록 안에 있다)
     + 그 검사기 자체의 시험
  F. 감시 워크플로 / keepalive
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
ML_DIR = ROOT / "scripts" / "ml"
PREDICT_LOCK = ML_DIR / "requirements-predict.lock.txt"
PUBLISH_LOCK = ROOT / "scripts" / "etl" / "requirements-publish.lock.txt"  # 발행기를 쓰는 다른 워크플로가 재사용한다

ALL_WORKFLOWS = sorted(WORKFLOW_DIR.glob("*.yml"))
# band-forecast.yml 은 후속 변경이 추가한다 — 생기면 같은 공통 규칙이 자동으로 적용된다.
ML_WORKFLOW_NAMES = [n for n in ("ml-predict.yml", "band-forecast.yml") if (WORKFLOW_DIR / n).exists()]

REPO_GUARD = "github.repository == 'AirLens-cloud/airlens-data'"
MAIN_REF_GUARD = "github.ref == 'refs/heads/main'"
SHA_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
VERSION_RE = re.compile(r"[0-9]+(\.[0-9]+)*")
MAX_TIMEOUT_MIN = 45
PUBLISH_MODES = {"candidate", "production"}
WHEEL_SECRET_PREFIX = "HF_MODELS_"
WRITE_SECRET = "HF_TOKEN"
CONSOLE_SCRIPTS = ("airlens-fetch-model", "airlens-predict-grid", "airlens-dqss-check", "airlens-drift-monitor")


# ── 도우미 ───────────────────────────────────────────────────────────────────
def load(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), f"{path.name}: 최상위가 매핑이 아니다"
    return doc


def triggers(doc: dict) -> dict:
    """PyYAML(YAML 1.1)은 bare `on` 을 True 로 읽는다. 문자열·목록·매핑 형태를 모두 매핑으로 통일한다."""
    raw = doc.get("on", doc.get(True))
    if isinstance(raw, str):
        return {raw: None}
    if isinstance(raw, list):
        return {name: None for name in raw}
    return dict(raw or {})


def pr_like(doc: dict) -> bool:
    """외부 기여자가 일으킬 수 있는 트리거(PR·PR 댓글·리뷰)."""
    return any(t.startswith(("pull_request", "issue_comment")) for t in triggers(doc))


def jobs(doc: dict) -> dict[str, dict]:
    return doc["jobs"]


def steps(job: dict) -> list[dict]:
    return job.get("steps", [])


def all_steps(doc: dict):
    for job_id, job in jobs(doc).items():
        for index, step in enumerate(steps(job)):
            yield job_id, index, step


def run_scripts(doc: dict):
    for job_id, index, step in all_steps(doc):
        if "run" in step:
            yield job_id, index, step["run"]


def logical_lines(script: str) -> list[str]:
    """주석 줄을 빼고 `\\` 이음줄을 한 줄로 합친다."""
    lines: list[str] = []
    buf = ""
    for raw in script.splitlines():
        stripped = raw.strip()
        if not buf and stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
            continue
        lines.append((buf + stripped).strip())
        buf = ""
    if buf:
        lines.append(buf.strip())
    return [line for line in lines if line]


def secrets_in(value: object) -> set[str]:
    return set(re.findall(r"secrets\.([A-Za-z0-9_]+)", json.dumps(value, ensure_ascii=False, default=str)))


def step_name(step: dict, index: int) -> str:
    return str(step.get("name") or step.get("id") or f"#{index}")


ml_params = pytest.mark.parametrize("name", ML_WORKFLOW_NAMES)


@pytest.fixture(scope="module")
def predict() -> dict:
    return load(WORKFLOW_DIR / "ml-predict.yml")


# ── A. 전체 워크플로 공통 ────────────────────────────────────────────────────
def test_ml_predict_workflow_exists_and_is_listed():
    # Arrange / Act / Assert — 아래 모든 ML 시험이 이 목록으로 도는데 목록이 비면 무음 통과한다
    assert "ml-predict.yml" in ML_WORKFLOW_NAMES


@pytest.mark.parametrize("path", ALL_WORKFLOWS, ids=lambda p: p.name)
def test_no_workflow_uses_pull_request_target(path):
    # Arrange
    doc = load(path)

    # Act / Assert — fork 코드를 시크릿 컨텍스트에서 실행하는 가장 위험한 트리거
    assert "pull_request_target" not in triggers(doc)


@pytest.mark.parametrize("path", [p for p in ALL_WORKFLOWS if pr_like(load(p))], ids=lambda p: p.name)
def test_workflows_triggered_by_pull_requests_never_reference_hf_secrets(path):
    # Arrange
    text = path.read_text(encoding="utf-8")

    # Act / Assert — 비공개 wheel 읽기 토큰·HF 쓰기 토큰이 PR 에서 도는 코드에 닿지 않는다
    assert WHEEL_SECRET_PREFIX not in text
    assert WRITE_SECRET not in text


def test_the_pull_request_secret_check_actually_sees_the_pr_workflows():
    # Arrange / Act
    names = {p.name for p in ALL_WORKFLOWS if pr_like(load(p))}

    # Assert — 위 시험의 parametrize 가 비어 무음 통과하지 않도록
    assert {"pr-check.yml"} <= names


def _cache_paths(doc: dict) -> list[str]:
    out = []
    for _job, _i, step in all_steps(doc):
        if str(step.get("uses", "")).startswith("actions/cache"):
            out.append(str((step.get("with") or {}).get("path", "")))
    return out


def _forbidden_cache_path(path: str) -> bool:
    return bool(re.search(r"wheel|venv|site-packages|\.cache/pip|\bpip\b|\.whl", path, re.IGNORECASE))


@pytest.mark.parametrize("path", ALL_WORKFLOWS, ids=lambda p: p.name)
def test_no_cache_step_stores_wheel_venv_or_pip_cache(path):
    # Arrange / Act
    paths = _cache_paths(load(path))

    # Assert — PR run 이 기본 브랜치 캐시를 복원할 수 있다
    assert not [p for p in paths if _forbidden_cache_path(p)]


def test_the_cache_path_check_would_catch_a_violation():
    # Arrange — 위 검사기 자체의 시험(대조군)
    doc = {"jobs": {"j": {"steps": [{"uses": "actions/cache@abc", "with": {"path": "~/.cache/pip\n.venv"}}]}}}

    # Act / Assert
    assert _forbidden_cache_path(_cache_paths(doc)[0])
    assert not _forbidden_cache_path("node_modules")


@pytest.mark.parametrize("path", ALL_WORKFLOWS, ids=lambda p: p.name)
def test_setup_python_never_enables_a_cache(path):
    # Arrange
    doc = load(path)

    # Act
    cached = [(j, i) for j, i, s in all_steps(doc)
              if str(s.get("uses", "")).startswith("actions/setup-python") and "cache" in (s.get("with") or {})]

    # Assert
    assert cached == []


@pytest.mark.parametrize("path", ALL_WORKFLOWS, ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_commit_sha(path):
    # Arrange
    doc = load(path)

    # Act
    unpinned = [s["uses"] for _j, _i, s in all_steps(doc)
                if "uses" in s and not str(s["uses"]).startswith("./") and not SHA_RE.fullmatch(str(s["uses"]).split("@", 1)[-1])]

    # Assert
    assert unpinned == []


# ── B. ML 워크플로 공통 ──────────────────────────────────────────────────────
@ml_params
def test_ml_workflow_has_no_pull_request_style_trigger(name):
    # Arrange
    doc = load(WORKFLOW_DIR / name)

    # Act / Assert
    assert not pr_like(doc)
    assert set(triggers(doc)) <= {"schedule", "workflow_dispatch"}


def guard_clauses(job: dict) -> list[str]:
    """job.if 를 `&&` 로 나눈 절(`${{ }}` 감싸기·공백 차이는 무시). `||` 가 있으면 AND 가 보장되지 않아 빈 목록이다."""
    condition = re.sub(r"^\s*\$\{\{|\}\}\s*$", "", str(job.get("if", ""))).strip()
    if "||" in condition:
        return []
    return [clause.strip() for clause in condition.split("&&")]


@ml_params
def test_every_ml_job_has_the_repository_guard_and_runs_only_on_main(name):
    # Arrange
    doc = load(WORKFLOW_DIR / name)

    # Act / Assert — fork 에서 스케줄이 돌아도 이 잡들은 건너뛴다. workflow_dispatch 는 임의의 브랜치를 고를 수 있으므로
    # main 이 아닌 ref 에서는 잡이 돌지 않아야 한다(브랜치에서 모드·핀을 바꾼 사본이 토큰을 쓰지 못하게)
    for job_id, job in jobs(doc).items():
        clauses = guard_clauses(job)
        assert REPO_GUARD in clauses, f"{name}:{job_id} repo 가드 없음"
        assert MAIN_REF_GUARD in clauses, f"{name}:{job_id} main 가드 없음(dispatch 로 브랜치 사본이 돈다)"


@pytest.mark.parametrize(
    ("condition", "ok"),
    [
        (f"{REPO_GUARD} && {MAIN_REF_GUARD}", True),
        (f"${{{{ {REPO_GUARD} && {MAIN_REF_GUARD} }}}}", True),
        (f"{MAIN_REF_GUARD} && {REPO_GUARD}", True),
        (REPO_GUARD, False),  # main 가드 없음
        (MAIN_REF_GUARD, False),  # repo 가드 없음
        (f"{REPO_GUARD} || {MAIN_REF_GUARD}", False),  # OR 이면 한쪽만 맞아도 돈다
        (f"{REPO_GUARD} && github.event_name == 'schedule' || github.event_name == 'workflow_dispatch' && {MAIN_REF_GUARD}",
         False),  # 두 가드가 각각 따로 있어도 가운데 || 가 있으면 AND 가 아니다(뒤쪽 묶음만 참이어도 돈다)
        (f"{REPO_GUARD} && github.ref == 'refs/heads/main-x'", False),  # 다른 ref
        (f"{REPO_GUARD} && github.ref_name == 'main'", False),  # 같은 뜻이어도 정해진 표현만 인정한다
        ("", False),
    ],
)
def test_the_job_guard_check_accepts_only_an_and_of_both_guards(condition, ok):
    # Arrange / Act
    clauses = guard_clauses({"if": condition})

    # Assert — 검사기 자체가 느슨해지지 않았음을 보인다
    assert (REPO_GUARD in clauses and MAIN_REF_GUARD in clauses) is ok


@ml_params
def test_every_ml_job_has_a_bounded_timeout(name):
    # Arrange
    doc = load(WORKFLOW_DIR / name)

    # Act / Assert
    for job_id, job in jobs(doc).items():
        timeout = job.get("timeout-minutes")
        assert isinstance(timeout, int) and 0 < timeout <= MAX_TIMEOUT_MIN, f"{name}:{job_id} timeout={timeout!r}"


@ml_params
def test_ml_workflow_permissions_are_explicit_and_read_only(name):
    # Arrange
    doc = load(WORKFLOW_DIR / name)

    # Act
    declared = [("workflow", doc.get("permissions"))] + [(j, job.get("permissions")) for j, job in jobs(doc).items() if "permissions" in job]

    # Assert
    assert doc.get("permissions") == {"contents": "read"}
    for where, perms in declared:
        assert perms is not None
        assert "write" not in json.dumps(perms), f"{name}:{where} 에 write 권한"


@ml_params
def test_ml_workflow_serializes_runs_without_cancelling_a_publish(name):
    # Arrange
    doc = load(WORKFLOW_DIR / name)

    # Act
    concurrency = doc.get("concurrency")

    # Assert — 발행 도중 취소되면 반쪽 상태가 남는다
    assert isinstance(concurrency, dict) and concurrency.get("group")
    assert concurrency.get("cancel-in-progress") is False


@ml_params
def test_publish_mode_is_valid_lives_in_workflow_env_and_is_not_a_dispatch_input(name):
    # Arrange
    path = WORKFLOW_DIR / name
    doc = load(path)
    inputs = (triggers(doc).get("workflow_dispatch") or {}).get("inputs") or {}
    text = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#"))

    # Act
    mode = (doc.get("env") or {}).get("PUBLISH_MODE")

    # Assert — 컷오버 = 이 값을 바꾸는 PR 하나(main 에 머지된 값만 쓰인다 — 두 잡이 main 에서만 돈다).
    #          dispatch 입력으로 모드를 바꿀 수 없다.
    if mode is None:
        pytest.skip(f"{name} 은 발행 모드를 쓰지 않는다")
    assert mode in PUBLISH_MODES
    assert "PUBLISH_MODE" not in inputs
    assert not re.search(r"inputs\.PUBLISH_MODE", text, re.IGNORECASE)


@ml_params
def test_ml_workflow_never_sets_the_live_repo_override(name):
    # Arrange
    doc = load(WORKFLOW_DIR / name)

    # Act — 발행·가중치 조달·관측 읽기의 대상 저장소가 함께 바뀌는 변수다. 설정(env 키·셸 대입)만 막는다.
    env_keys: set[str] = set(doc.get("env") or {})
    for job in jobs(doc).values():
        env_keys |= set(job.get("env") or {})
        for step in steps(job):
            env_keys |= set(step.get("env") or {})
    assigned = [line for _j, _i, script in run_scripts(doc) for line in logical_lines(script)
                if re.search(r"HF_LIVE_REPO\s*=|export\s+HF_LIVE_REPO", line)]

    # Assert
    assert "HF_LIVE_REPO" not in env_keys
    assert assigned == []


@ml_params
def test_secrets_are_only_in_the_env_of_the_step_that_uses_them(name):
    # Arrange
    doc = load(WORKFLOW_DIR / name)

    # Act / Assert — 워크플로·잡 env 에 두면 그 범위의 모든 스텝(= 서드파티 의존성 실행)이 본다
    assert secrets_in(doc.get("env")) == set(), f"{name}: 워크플로 env 에 secrets"
    for job_id, job in jobs(doc).items():
        assert secrets_in(job.get("env")) == set(), f"{name}:{job_id} 잡 env 에 secrets"
        for index, step in enumerate(steps(job)):
            in_run = re.findall(r"secrets\.[A-Za-z0-9_]+", str(step.get("run", "")))
            assert in_run == [], f"{name}:{job_id}:{step_name(step, index)} run 본문에 secrets 직접 삽입"


@ml_params
def test_run_scripts_do_not_interpolate_expressions_or_trace_commands(name):
    # Arrange
    doc = load(WORKFLOW_DIR / name)

    # Act / Assert
    for job_id, index, script in run_scripts(doc):
        lines = logical_lines(script)
        assert not any("${{" in line for line in lines), f"{name}:{job_id}#{index} 표현식을 셸에 직접 삽입(주입 위험, env 로 받는다)"
        assert not any(re.search(r"set\s+-[a-zA-Z]*x|\bxtrace\b|\bprintenv\b", line) for line in lines), f"{name}:{job_id}#{index} 명령 추적/환경 출력"


@ml_params
def test_secret_values_are_never_echoed_by_run_scripts(name):
    # Arrange
    doc = load(WORKFLOW_DIR / name)
    pattern = re.compile(r"\b(echo|printf|cat|tee)\b[^\n]*\$\{?(HF_TOKEN|HF_MODELS_[A-Z]+)\b")

    # Act
    leaks = [line for _j, _i, script in run_scripts(doc) for line in logical_lines(script) if pattern.search(line)]

    # Assert
    assert leaks == []


# ── C. ml-predict.yml 전용 ───────────────────────────────────────────────────
def test_predict_trigger_is_the_documented_schedule_plus_manual_dispatch_with_one_input(predict):
    # Arrange / Act
    trig = triggers(predict)

    # Assert
    assert trig["schedule"] == [{"cron": "45 */3 * * *"}]
    assert set((trig["workflow_dispatch"] or {}).get("inputs") or {}) == {"max_age_hours"}


def test_predict_has_separate_infer_and_publish_jobs_with_publish_after_infer(predict):
    # Act / Assert
    assert set(jobs(predict)) == {"infer", "publish"}
    assert jobs(predict)["publish"].get("needs") == "infer"


def test_infer_job_keeps_joblib_from_starting_a_process_pool(predict):
    # Arrange / Act
    env = jobs(predict)["infer"].get("env") or {}

    # Assert — 프로세스 풀 worker 의 traceback 은 소스 줄이 든 문자열로 예외에 실려 sys.tracebacklimit 이 닿지 않는다.
    #          잡 env 에 두어 모든 스텝이 같은 값을 본다(비밀 아님 — 잡 env 의 secrets 금지 시험이 따로 있다).
    assert str(env.get("JOBLIB_MULTIPROCESSING")) == "0"


def test_wheel_read_token_exists_only_in_the_wheel_fetch_step_of_the_infer_job(predict):
    # Arrange / Act
    holders = [(j, s.get("id")) for j, _i, s in all_steps(predict) if any(k.startswith(WHEEL_SECRET_PREFIX) for k in (s.get("env") or {}))]
    names = {n for _j, _i, s in all_steps(predict) for n in secrets_in(s) if n.startswith(WHEEL_SECRET_PREFIX)}

    # Assert — 읽기 토큰은 한 스텝에만, publish 잡에는 없다
    assert holders == [("infer", "wheel")]
    assert names == {"HF_MODELS_READ", "HF_MODELS_REPO"}


def test_write_token_exists_only_in_publish_steps_that_run_the_publisher(predict):
    # Arrange / Act
    holders = [(j, i, s) for j, i, s in all_steps(predict) if WRITE_SECRET in (s.get("env") or {})]

    # Assert — infer 잡(서드파티 의존성 + 비공개 wheel 실행)에는 쓰기 토큰이 없다
    assert holders, "발행 스텝이 쓰기 토큰을 받지 못한다"
    assert {j for j, _i, _s in holders} == {"publish"}
    for _j, _i, step in holders:
        assert "hf_publish.py" in step["run"]
    assert all(WRITE_SECRET not in json.dumps(s, default=str) for j, _i, s in all_steps(predict) if j == "infer")


def test_publish_job_has_no_wheel_and_infer_job_has_no_publisher(predict):
    # Arrange
    publish_text = json.dumps(jobs(predict)["publish"], ensure_ascii=False, default=str)
    infer_text = json.dumps(jobs(predict)["infer"], ensure_ascii=False, default=str)

    # Act / Assert
    for forbidden in ("fetch_inference_wheel", "run_entry.py", "requirements-predict.lock", "airlens-predict-grid"):
        assert forbidden not in publish_text
    assert "hf_publish.py" not in infer_text


def test_pins_are_exact_and_never_latest(predict):
    # Arrange
    env = predict["env"]

    # Act / Assert
    assert SHA256_RE.fullmatch(str(env["INFERENCE_WHEEL_SHA256"]))
    assert VERSION_RE.fullmatch(str(env["INFERENCE_VERSION"]))
    assert "latest" not in json.dumps(env).lower()


def test_wheel_fetch_uses_the_pins_and_exports_the_wheel_path(predict):
    # Arrange
    step = next(s for _j, _i, s in all_steps(predict) if s.get("id") == "wheel")

    # Act
    line = " ".join(logical_lines(step["run"]))

    # Assert
    assert "fetch_inference_wheel.py" in line
    assert '--version "$INFERENCE_VERSION"' in line and '--sha256 "$INFERENCE_WHEEL_SHA256"' in line
    install = next(s for _j, _i, s in all_steps(predict) if s.get("name") == "Install pinned environment")
    assert install["env"]["WHEEL"] == "${{ steps.wheel.outputs.wheel }}"


def test_every_airlens_command_runs_through_run_entry_in_isolated_unbuffered_mode(predict):
    # Arrange / Act
    command_lines = [line for _j, _i, script in run_scripts(predict) for line in logical_lines(script)
                     if any(name in line for name in CONSOLE_SCRIPTS)]

    # Assert — 실패해도 비공개 코드 줄이 공개 로그에 남지 않게(traceback 에서 프레임·소스 줄 제거)
    assert len(command_lines) == len(CONSOLE_SCRIPTS), command_lines
    for line in command_lines:
        assert "scripts/ml/run_entry.py" in line, f"run_entry 를 거치지 않는 호출: {line}"
        assert " -I " in line and " -u " in line, f"-I(격리)·-u(비버퍼) 없음: {line}"


def test_install_steps_use_hash_pinned_binary_only_uncached_installs(predict):
    # Arrange
    installs = [line for j, _i, script in run_scripts(predict) if j == "infer" for line in logical_lines(script) if "pip install" in line]

    # Act
    lock_line = next(line for line in installs if "requirements-predict.lock.txt" in line)
    wheel_line = next(line for line in installs if '"$WHEEL"' in line)

    # Assert
    assert len(installs) == 2
    assert all("--no-cache-dir" in line for line in installs)
    for flag in ("--require-hashes", "--no-deps", "--only-binary :all:"):
        assert flag in lock_line
    assert "--no-deps" in wheel_line


def test_environment_check_and_pip_listing_follow_the_installs(predict):
    # Arrange
    script = next(s["run"] for _j, _i, s in all_steps(predict) if s.get("name") == "Install pinned environment")
    lines = logical_lines(script)

    # Act
    positions = {key: next(i for i, line in enumerate(lines) if key in line) for key in ("requirements-predict.lock.txt", '"$WHEEL"', "pip check")}

    # Assert — wheel 의 의존성이 고정 목록과 어긋나면 pip check 가 여기서 실패한다
    assert positions["requirements-predict.lock.txt"] < positions['"$WHEEL"'] < positions["pip check"]


def test_only_json_outputs_are_staged_and_uploaded(predict):
    # Arrange
    stage = next(s for _j, _i, s in all_steps(predict) if s.get("name") == "Stage JSON outputs")
    upload = next(s for _j, _i, s in all_steps(predict) if str(s.get("uses", "")).startswith("actions/upload-artifact"))

    # Act
    copies = [line for line in logical_lines(stage["run"]) if line.startswith("cp ")]
    sources = [tok for line in copies for tok in line.split()[1:-1]]
    upload_path = str(upload["with"]["path"])

    # Assert
    assert copies
    assert all(src.strip('"').endswith(".json") for src in sources), sources
    assert not re.search(r"wheel|venv|\.whl|\.log|lock", upload_path, re.IGNORECASE)
    assert upload["with"].get("if-no-files-found") == "error"
    assert 0 < int(upload["with"]["retention-days"]) <= 7


def test_artifact_name_matches_between_upload_and_download(predict):
    # Arrange
    uploads = [s["with"]["name"] for _j, _i, s in all_steps(predict) if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    downloads = [s["with"]["name"] for _j, _i, s in all_steps(predict) if str(s.get("uses", "")).startswith("actions/download-artifact")]

    # Act / Assert
    assert uploads == downloads and len(uploads) == 1


def test_publish_uses_the_mapping_script_for_every_destination_and_validates_before_upload(predict):
    # Arrange
    publish_scripts = [script for j, _i, script in run_scripts(predict) if j == "publish"]
    lines = [line for script in publish_scripts for line in logical_lines(script)]

    # Act
    batch_lines = [line for line in lines if "hf_publish.py batch" in line]

    # Assert — 목적지 경로를 워크플로에 손으로 적지 않는다(모드에 따라 publish_targets.py 가 정한다)
    assert len(batch_lines) == 3
    assert all("--schema auto" in line for line in batch_lines)
    assert all("aq-data/" not in line for line in lines), "목적지 경로가 워크플로에 하드코딩됐다"
    assert sum("publish_targets.py maps" in line for line in lines) == 3


def _step_lines(doc: dict, name: str) -> list[str]:
    step = next(s for _j, _i, s in all_steps(doc) if s.get("name") == name)
    return logical_lines(step["run"])


def test_required_publishes_fail_the_run(predict):
    # Arrange / Act
    grid = " ".join(_step_lines(predict, "Publish grid"))
    dqss = " ".join(_step_lines(predict, "Publish DQSS"))

    # Assert — grid·DQSS 는 필수: 업로드가 실패하면 run 이 실패한다
    for joined in (grid, dqss):
        assert '|| { echo "::error::' in joined and "exit 1" in joined


def test_advisory_publish_steps_turn_upload_failures_into_warnings(predict):
    # Arrange
    shadow = next(line for line in _step_lines(predict, "Publish shadow snapshot (advisory)") if "hf_publish.py batch" in line)
    prune = next(line for line in _step_lines(predict, "Prune history snapshots") if "hf_publish.py prune" in line)

    # Act / Assert — 섀도우·prune 은 advisory: 이미 올라간 점수를 되돌리지 않고, 발행 체인을 막지 않는다
    for line in (shadow, prune):
        assert '|| echo "::warning::' in line
        assert "exit 1" not in line


def test_predict_steps_use_pipefail_so_a_failing_command_is_not_hidden_by_tee(predict):
    # Arrange
    step = next(s for _j, _i, s in all_steps(predict) if s.get("name") == "Predict observed stations (OpenAQ shadow feed)")

    # Act
    lines = logical_lines(step["run"])

    # Assert
    assert lines[0] == "set -euo pipefail"
    assert any("| tee predict.log" in line for line in lines)
    assert step["env"]["MAX_AGE_HOURS"] == "${{ inputs.max_age_hours || '24' }}"


def test_dqss_gate_withholds_on_fail_and_crashes_propagate(predict):
    # Arrange
    lines = _step_lines(predict, "DQSS information gate")

    # Act / Assert — FAIL 은 withheld payload 로 바꿔 발행, 스크립트 크래시는 set -e 로 이 스텝을 실패시킨다
    assert lines[0] == "set -euo pipefail"
    assert any("airlens-dqss-check" in line and "--withhold-on-fail" in line for line in lines)


DRIFT_STEP = "Drift monitor (advisory)"
# 리포트에 들어 있을 법한 값(보정 상수·모델 sha·기준 지표)을 흉내 낸 표지. 공개 로그·요약에 나오면 안 된다.
DRIFT_SENTINELS = ("4.123456789", "0.987654321", "sha_sentinel_deadbeef")


def _drift_step(doc: dict) -> dict:
    return next(s for _j, _i, s in all_steps(doc) if s.get("name") == DRIFT_STEP)


def test_drift_monitor_step_cannot_fail_the_job_by_construction(predict):
    # Arrange
    step = _drift_step(predict)
    lines = logical_lines(step["run"])

    # Act / Assert — advisory 스텝: continue-on-error, 기본 셸의 -e 를 맨 앞에서 끄고 다시 켜지 않으며, 비정상 종료 경로가 없다
    assert step.get("continue-on-error") is True
    assert step.get("id") == "drift"  # id 없는 continue-on-error 스텝은 레포 가드(test_build_mac_index)가 막는다
    assert lines[0] == "set +e"
    assert not [line for line in lines if re.match(r"set\s+-\S*e", line)], "errexit 를 다시 켜는 줄이 있다"
    assert not any(re.search(r"\bexit\s+[1-9]", line) for line in lines)
    assert lines[-1] == "exit 0"


def test_drift_report_contents_never_reach_the_public_log_summary_or_artifact(predict):
    # Arrange
    lines = logical_lines(_drift_step(predict)["run"])
    stage = _step_lines(predict, "Stage JSON outputs")

    # Act
    mentions = [line for line in lines if "drift_report.json" in line]

    # Assert — 리포트에는 보정 상수·모델 sha·기준 지표가 있다. 공개 로그·요약·artifact 에는 전문을 싣지 않는다.
    assert not [line for line in lines if re.match(r"(cat|head|tail|tee|jq|cp|mv)\b", line)]
    assert len(mentions) == 3  # 모니터에 주는 --output 인자 1 + 불리언 두 개를 읽는 python 2
    assert all("isinstance(v,bool)" in line for line in mentions if line.startswith(("DRIFT=", "DUAL=")))
    assert not any("drift_report" in line for line in stage)


def run_drift_step(tmp_path: Path, predict: dict, *, stub_rc: int = 0, report: str | None = None,
                   work_exists: bool = True, summary_is_dir: bool = False) -> tuple[subprocess.CompletedProcess, str]:
    """스텝의 run 스크립트를 GitHub 의 bash 기본 옵션(`-eo pipefail`)으로 실제 실행한다. 모니터(`$VENV/bin/python`)는 stub."""
    work, venv_bin, tool_bin = tmp_path / "work", tmp_path / "venv" / "bin", tmp_path / "tools"
    for directory in (venv_bin, tool_bin):
        directory.mkdir(parents=True)
    if work_exists:
        (work / "output").mkdir(parents=True)
    stub = venv_bin / "python"
    stub.write_text('#!/bin/sh\n[ -f "$STUB_REPORT" ] && cp "$STUB_REPORT" output/drift_report.json\nexit "$STUB_RC"\n', encoding="utf-8")
    python3 = tool_bin / "python3"
    python3.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    for tool in (stub, python3):
        tool.chmod(0o755)
    report_file = tmp_path / "report.json"
    if report is not None:
        report_file.write_text(report, encoding="utf-8")
    summary = tmp_path if summary_is_dir else tmp_path / "summary.md"
    script = tmp_path / "step.sh"
    script.write_text(_drift_step(predict)["run"], encoding="utf-8")
    env = {"PATH": f"{tool_bin}:/usr/bin:/bin", "WORK": str(work), "VENV": str(tmp_path / "venv"),
           "GITHUB_WORKSPACE": str(tmp_path / "workspace"), "GITHUB_STEP_SUMMARY": str(summary),
           "STUB_RC": str(stub_rc), "STUB_REPORT": str(report_file)}
    proc = subprocess.run([shutil.which("bash") or "bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)],
                          env=env, capture_output=True, text=True, timeout=60)
    return proc, (summary.read_text(encoding="utf-8") if summary.is_file() else "")


def _report(**over: object) -> str:
    base = {"overall_drift": False, "model_sha": "sha_sentinel_deadbeef",
            "predictions_drift": {"dual_metrics_present": True, "baseline_rmse": 4.123456789, "calibration": {"a": 0.987654321}}}
    base.update(over)
    return json.dumps(base)


@pytest.mark.parametrize(
    ("label", "options", "expected_log"),
    [
        ("monitor-fails", {"stub_rc": 1}, "drift monitor 실행 실패 (advisory — rc=1)"),
        ("monitor-fails-after-writing-a-report", {"stub_rc": 3, "report": "{}"}, "rc=3"),
        ("no-report", {}, "overall_drift=unknown dual_metrics_present=unknown"),
        ("broken-json", {"report": "{not json"}, "overall_drift=unknown dual_metrics_present=unknown"),
        ("report-is-a-list", {"report": "[1, 2]"}, "overall_drift=unknown dual_metrics_present=unknown"),
        ("predictions-drift-is-a-string", {"report": '{"overall_drift": true, "predictions_drift": "x"}'},
         "overall_drift=True dual_metrics_present=unknown"),
        ("predictions-drift-is-null", {"report": '{"overall_drift": false, "predictions_drift": null}'},
         "overall_drift=False dual_metrics_present=unknown"),
        ("work-directory-missing", {"work_exists": False}, "작업 폴더 없음"),
        ("summary-file-unwritable", {"report": _report(), "summary_is_dir": True}, "drift summary:"),
    ],
)
def test_drift_step_exits_zero_whatever_goes_wrong(tmp_path, predict, label, options, expected_log):
    # Arrange / Act — GitHub 기본 셸(-e)에서도 후처리 실패가 스텝·job 을 실패시키지 않는다
    proc, _summary = run_drift_step(tmp_path, predict, **options)

    # Assert
    assert proc.returncode == 0, f"{label}: {proc.stderr}"
    assert expected_log in proc.stdout


@pytest.mark.parametrize("report", ["{not json", "[1, 2]", '{"overall_drift": true, "predictions_drift": "x"}'],
                         ids=["broken-json", "list", "predictions-drift-is-a-string"])
def test_drift_step_hides_python_tracebacks_for_unreadable_reports(tmp_path, predict, report):
    # Arrange / Act
    proc, _summary = run_drift_step(tmp_path, predict, report=report)

    # Assert — 읽을 수 없는 리포트는 unknown 으로만 보이고, 공개 로그에 traceback 이 남지 않는다
    assert (proc.returncode, proc.stderr) == (0, "")


def test_drift_step_failing_monitor_writes_no_summary_and_no_derived_values(tmp_path, predict):
    # Arrange / Act
    proc, summary = run_drift_step(tmp_path, predict, stub_rc=1, report=_report(overall_drift=True))

    # Assert — 모니터가 실패했으면 남아 있을지 모를 이전 리포트를 읽지 않는다
    assert (proc.returncode, summary) == (0, "")
    assert "drift summary" not in proc.stdout


def test_drift_step_logs_derived_booleans_and_warnings_but_never_the_report(tmp_path, predict):
    # Arrange
    report = _report(overall_drift=True, predictions_drift={"dual_metrics_present": False, "baseline_rmse": 4.123456789,
                                                              "calibration": {"a": 0.987654321}})

    # Act
    proc, summary = run_drift_step(tmp_path, predict, report=report)

    # Assert
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "drift summary: overall_drift=True dual_metrics_present=False" in proc.stdout
    assert out.count("::warning::") == 2
    for text in (out, summary):
        assert not [sentinel for sentinel in DRIFT_SENTINELS if sentinel in text]
    assert "overall_drift: `True`" in summary and "dual_metrics_present: `False`" in summary


def test_drift_step_without_drift_prints_no_warnings(tmp_path, predict):
    # Arrange / Act
    proc, _summary = run_drift_step(tmp_path, predict, report=_report())

    # Assert
    assert proc.returncode == 0
    assert "::warning::" not in proc.stdout
    assert "overall_drift=False dual_metrics_present=True" in proc.stdout


def test_drift_step_does_not_copy_a_non_boolean_report_value_into_the_log(tmp_path, predict):
    # Arrange — 리포트의 임의 값(문자열 등)이 로그로 새는 길을 막는다: 불리언만 옮긴다
    report = json.dumps({"overall_drift": "sha_sentinel_deadbeef", "predictions_drift": {"dual_metrics_present": "4.123456789"}})

    # Act
    proc, summary = run_drift_step(tmp_path, predict, report=report)

    # Assert
    assert proc.returncode == 0
    assert "overall_drift=unknown dual_metrics_present=unknown" in proc.stdout
    assert not [sentinel for sentinel in DRIFT_SENTINELS if sentinel in proc.stdout + summary]


def test_checkout_steps_do_not_persist_credentials(predict):
    # Arrange
    checkouts = [s for _j, _i, s in all_steps(predict) if str(s.get("uses", "")).startswith("actions/checkout")]

    # Act / Assert
    assert len(checkouts) == 2
    assert all((s.get("with") or {}).get("persist-credentials") is False for s in checkouts)


def test_python_versions_follow_the_documented_split(predict):
    # Arrange
    versions = {j: str((s.get("with") or {}).get("python-version")) for j, _i, s in all_steps(predict) if str(s.get("uses", "")).startswith("actions/setup-python")}

    # Act / Assert — 추론은 3.12(numpy·scipy 최소 요구), 발행기는 기존 워크플로와 같은 3.11
    assert versions == {"infer": "3.12", "publish": "3.11"}


# ── D. 환경 고정 파일 ────────────────────────────────────────────────────────
def _constraint_pins() -> dict[str, str]:
    pins = {}
    for line in (ML_DIR / "constraints-predict.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        assert match, f"고정 형식이 아닌 줄: {line!r}"
        pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def _lock_entries(lock: Path = PREDICT_LOCK) -> dict[str, list[str]]:
    entries: dict[str, list[str]] = {}
    current = None
    for raw in lock.read_text(encoding="utf-8").splitlines():
        line = raw.strip().rstrip("\\").strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--hash=sha256:"):
            assert current is not None, "해시가 요구사항보다 앞에 있다"
            entries[current].append(line.removeprefix("--hash=sha256:"))
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        assert match, f"고정 형식이 아닌 줄: {line!r}"
        current = f"{match.group(1).lower().replace('_', '-')}=={match.group(2)}"
        entries[current] = []
    return entries


def test_constraints_and_hash_lock_pin_the_same_versions():
    # Arrange
    pins = {f"{name}=={version}" for name, version in _constraint_pins().items()}

    # Act
    lock = set(_lock_entries())

    # Assert — 한쪽만 올리면 여기서 걸린다 (올릴 때는 constraints → lock 재생성 순서)
    assert pins == lock


@pytest.mark.parametrize(("lock", "minimum"), [(PREDICT_LOCK, 15), (PUBLISH_LOCK, 10)], ids=["predict", "publish"])
def test_every_lock_entry_has_sha256_hashes_and_nothing_can_redirect_the_index(lock, minimum):
    # Arrange
    entries = _lock_entries(lock)
    text = lock.read_text(encoding="utf-8")

    # Act / Assert
    assert len(entries) >= minimum
    for requirement, hashes in entries.items():
        assert hashes and all(SHA256_RE.fullmatch(h) for h in hashes), requirement
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    for forbidden in ("--index-url", "--extra-index-url", "--find-links", "--trusted-host", "-e ", "git+", "@ http"):
        assert forbidden not in code


def test_publish_lock_pins_huggingface_hub_inside_the_supported_range_with_its_transfer_backend():
    # Arrange
    pins = {name.partition("==")[0]: name.partition("==")[2] for name in _lock_entries(PUBLISH_LOCK)}

    # Act
    major, minor = (int(part) for part in pins["huggingface-hub"].split(".")[:2])

    # Assert — 발행기는 0.34 이상 2 미만을 지원한다. hf-xet 은 환경 마커 의존성이라 빠지기 쉽다(재생성 때 플랫폼 지정 실수)
    assert (major, minor) >= (0, 34) and major < 2
    assert "hf-xet" in pins


def test_publish_job_installs_only_the_hash_locked_file_and_checks_the_environment_after(predict):
    # Arrange
    lines = [line for j, _i, script in run_scripts(predict) if j == "publish" for line in logical_lines(script)]

    # Act
    installs = [line for line in lines if re.search(r"\binstall\b", line)]

    # Assert — 쓰기 토큰이 있는 잡: 설치는 해시 고정본 한 줄뿐이고, 범위 지정·직접 이름 설치가 없다
    assert len(installs) == 1, installs
    for flag in ("--require-hashes", "--no-deps", "--only-binary :all:", "--no-cache-dir"):
        assert flag in installs[0]
    assert installs[0].endswith(f"-r {PUBLISH_LOCK.relative_to(ROOT).as_posix()}")  # 아래 시험들이 검증하는 파일과 같다
    assert lines.index(installs[0]) < lines.index("python3 -m pip check")


def test_publish_step_installing_the_lock_has_no_secrets(predict):
    # Arrange
    step = next(s for j, _i, s in all_steps(predict) if j == "publish" and "requirements-publish.lock.txt" in s.get("run", ""))

    # Act / Assert — 의존성을 깔고 실행하는 스텝에는 토큰이 없다(토큰은 발행기를 실행하는 스텝에만)
    assert not secrets_in(step)


def test_constraints_have_no_range_specifiers():
    # Arrange
    text = (ML_DIR / "constraints-predict.txt").read_text(encoding="utf-8")
    code = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]

    # Act / Assert
    assert code and all("==" in line and not re.search(r"[<>~!]=?|\*", line.split("==", 1)[1]) for line in code)


# ── E. 공개 위생 ─────────────────────────────────────────────────────────────
# 금지어 정규식은 문자 클래스로 짜서 이 파일 자체에 리터럴이 남지 않게 한다(자기 자신을 검사해도 걸리지 않는다).
# 저장소 id 는 허용목록으로 본다: 이 레포가 실제로 참조하는 공개 저장소만 통과하고 나머지는(오타든 다른 저장소든) 실패한다.
# 금지할 이름을 나열하지 않는다 — 나열 자체가 공개 파일에 남기 때문이다.
HYGIENE_PATTERNS: dict[str, list[str]] = {
    "로컬 경로": [r"/Use[r]s/", r"/hom[e]/(?!runner/)[a-z]+/", r"/[Vv]olumes/[A-Za-z]", r"Dev_[A-Za-z]+"],
    "운영 환경 이름": [r"Orac[l]e", r"self-[h]osted", r"\bE[2]\b", r"air[l]ens-e[2]\b", r"\bV[M]\b"],
    "내부 문서 번호": [r"ORG-[0-9]{4}", r"ADR[ -]?[0-9]{4}"],
    "토큰 모양": [
        r"hf_[A-Za-z0-9]{20,}", r"gh[pousr]_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}",
        r"sk-[A-Za-z0-9_-]{20,}", r"AKIA[0-9A-Z]{16}", r"eyJ[A-Za-z0-9_-]{20,}\.", r"Bearer\s+[A-Za-z0-9._-]{20,}",
    ],
}
IP_PATTERN = r"\b(?!127\.0\.0\.1\b)[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b"
# 저장소 id 허용목록(소문자). 새 파일이 다른 공개 저장소를 참조하게 되면 여기에 더한다(쓰이지 않는 항목은 아래 시험이 잡는다).
ALLOWED_REPO_IDS: dict[str, set[str]] = {
    "robeedau": {"airlens-live"},
    "airlens-cloud": {"airlens-data"},
}
REPO_ID_PATTERN = r"(?i)(Robeedau|AirLens-cloud)/([A-Za-z0-9_.-]+)"  # ASCII 만: 한글 조사가 바로 붙어도 id 에 딸려 오지 않게
UNLISTED_REPO = "허용되지 않은 저장소 id"
SCANNED_SUFFIXES = {".py", ".yml", ".txt", ".md", ".json"}


def repo_ids(text: str) -> list[tuple[str, str, str]]:
    """(원문, 소유자, 저장소) — 소문자, 문장 끝 `.` · `-` 와 `.git` 접미사는 뗀다."""
    return [(m.group(0), m.group(1).lower(), m.group(2).rstrip(".-").lower().removesuffix(".git"))
            for m in re.finditer(REPO_ID_PATTERN, text)]


def hygiene_findings(text: str, *, check_ips: bool = True) -> list[tuple[str, str]]:
    found = []
    for category, patterns in HYGIENE_PATTERNS.items():
        for pattern in patterns:
            found += [(category, m.group(0)) for m in re.finditer(pattern, text)]
    found += [(UNLISTED_REPO, raw) for raw, owner, repo in repo_ids(text) if repo not in ALLOWED_REPO_IDS[owner]]
    if check_ips:
        found += [("IP 주소", m.group(0)) for m in re.finditer(IP_PATTERN, text)]
    return found


def public_files() -> list[Path]:
    files = [
        WORKFLOW_DIR / "ml-predict.yml",
        WORKFLOW_DIR / "predictions-freshness-healthcheck.yml",
        ROOT / "scripts" / "etl" / "probe_predictions_freshness.py",
        ROOT / "scripts" / "etl" / "test_probe_predictions_freshness.py",
        PUBLISH_LOCK,
        *sorted(p for p in ML_DIR.iterdir() if p.is_file() and p.suffix in SCANNED_SUFFIXES),
    ]
    return [f for f in files if f.exists()]


@pytest.mark.parametrize("path", public_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_public_files_pass_the_hygiene_scan(path):
    # Arrange
    text = path.read_text(encoding="utf-8")

    # Act
    findings = hygiene_findings(text, check_ips=path.suffix in {".py", ".yml"})

    # Assert — 값 자체는 출력하지 않고 종류와 개수만 (실패 메시지가 로그에 남는다)
    assert not findings, f"{path.name}: " + ", ".join(sorted({category for category, _ in findings}))


def test_the_hygiene_scan_covers_the_expected_files():
    # Arrange
    names = {p.name for p in public_files()}

    # Act / Assert — 글롭이 비면 위 parametrize 가 무음 통과한다
    assert {"ml-predict.yml", "predictions-freshness-healthcheck.yml", "probe_predictions_freshness.py",
            "fetch_inference_wheel.py", "run_entry.py", "publish_targets.py", "compare_candidate.py",
            "summarize_predict_log.py", "constraints-predict.txt", "requirements-predict.lock.txt",
            "requirements-publish.lock.txt", "test_ml_workflow_safety.py"} <= names


def test_every_allowed_repo_id_is_referenced_by_a_scanned_file():
    # Arrange — 허용목록은 "새 파일이 실제로 참조하는 공개 저장소" 만 둔다. 참조가 사라지면 항목도 뺀다.
    referenced = {(owner, repo) for path in public_files() for _raw, owner, repo in repo_ids(path.read_text(encoding="utf-8"))}

    # Act
    unused = sorted((owner, repo) for owner, repos in ALLOWED_REPO_IDS.items() for repo in repos if (owner, repo) not in referenced)

    # Assert
    assert not unused, f"참조하는 파일이 없는 허용 항목: {unused}"


@pytest.mark.parametrize(
    ("category", "sample"),
    [
        ("로컬 경로", "/Us" + "ers/someone/project"),
        ("로컬 경로", "/ho" + "me/someone/project"),
        ("로컬 경로", "/Vol" + "umes/External/data"),
        ("로컬 경로", "/vol" + "umes/x"),
        ("로컬 경로", "Dev_" + "anything"),
        ("운영 환경 이름", "Ora" + "cle host"),
        ("운영 환경 이름", "self-" + "hosted"),
        ("운영 환경 이름", "the E" + "2 host"),
        ("운영 환경 이름", "airlens-e" + "2"),
        ("운영 환경 이름", "a V" + "M here"),
        ("내부 문서 번호", "ORG-" + "0020"),
        ("내부 문서 번호", "ADR " + "0018"),
        (UNLISTED_REPO, "Robeedau/" + "some-other-name"),
        (UNLISTED_REPO, "robeedau/" + "some-other-name"),  # 대소문자는 구분하지 않는다
        (UNLISTED_REPO, "AirLens-cloud/" + "some-other-name"),
        (UNLISTED_REPO, "airlens-cloud/" + "some-other-name"),
        (UNLISTED_REPO, "Robeedau/" + "airlens-live-extra"),  # 허용 id 로 시작해도 다르면 실패한다
        (UNLISTED_REPO, "AirLens-cloud/" + "airlens-data2"),
        (UNLISTED_REPO, "AirLens-cloud/" + "airlens-data.other"),  # 점 뒤가 이어지면 다른 저장소다(`.git` 접미사만 뗀다)
        (UNLISTED_REPO, "https://huggingface.co/datasets/Robeedau/" + "other/resolve/main/x.json"),
        ("토큰 모양", "hf" + "_" + "a" * 24),
        ("토큰 모양", "gh" + "p_" + "b" * 24),
        ("토큰 모양", "github" + "_pat_" + "c" * 24),
        ("토큰 모양", "sk" + "-" + "d" * 24),
        ("토큰 모양", "AK" + "IA" + "E" * 16),
        ("토큰 모양", "ey" + "J" + "f" * 24 + "."),
        ("토큰 모양", "Bear" + "er " + "g" * 24),
        ("IP 주소", "10." + "0.0.5"),
    ],
)
def test_the_hygiene_scan_detects_each_forbidden_shape(category, sample):
    # Arrange / Act
    found = {c for c, _ in hygiene_findings(f"text {sample} text")}

    # Assert — 검사기가 정규식 오타로 무력해지지 않았음을 보인다
    assert category in found


@pytest.mark.parametrize(
    "sample",
    [
        "127.0.0.1 and 2026.7.22 and 0.1.0",
        "Bearer {token} and Bearer unit-test",
        "ORG-12 and ADR 12 (번호는 4자리일 때만 센다)",
        "Robeedau/" + "airlens-live",
        "robeedau/" + "AirLens-Live",  # 대소문자는 구분하지 않는다(허용 id 도 마찬가지)
        "https://huggingface.co/datasets/Robeedau/" + "airlens-live/resolve/main/aq-data/x.json",
        "AirLens-cloud/" + "airlens-data",
        "github.repository == 'AirLens-cloud/" + "airlens-data'",
        "see AirLens-cloud/" + "airlens-data.",  # 문장 끝 마침표
        "https://github.com/AirLens-cloud/" + "airlens-data.git",
        "AirLens-cloud/" + "airlens-data" + "에서",  # 한글 조사가 바로 붙은 경우
        "/home/" + "runner/work/x",  # 러너의 표준 경로
    ],
)
def test_the_hygiene_scan_does_not_flag_allowed_shapes(sample):
    # Arrange / Act / Assert
    assert hygiene_findings(sample) == []


# ── F. 감시 워크플로 / keepalive ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def probe_workflow() -> dict:
    return load(WORKFLOW_DIR / "predictions-freshness-healthcheck.yml")


def test_probe_workflow_has_repo_guard_bounded_time_and_least_privilege(probe_workflow):
    # Arrange / Act
    job = jobs(probe_workflow)["probe"]

    # Assert — 이슈를 쓰고(issues: write) 워크플로 state 를 읽는다(actions: read). 그 밖의 권한은 없다.
    assert REPO_GUARD in str(job["if"])
    assert 0 < job["timeout-minutes"] <= MAX_TIMEOUT_MIN
    assert probe_workflow["permissions"] == {"contents": "read", "issues": "write", "actions": "read"}
    assert not pr_like(probe_workflow)


def test_probe_workflow_watches_the_publishing_workflows_and_uses_the_web_staleness_limit(probe_workflow):
    # Arrange
    probe = next(s for _j, _i, s in all_steps(probe_workflow) if s.get("id") == "probe")

    # Act
    env = probe["env"]

    # Assert
    assert env["WATCHED_WORKFLOWS"].split(",") == ["ml-predict.yml", "band-forecast.yml"]
    assert env["BAND_MAX_AGE_HOURS"] == "12"
    assert env["GH_TOKEN"] == "${{ github.token }}"


def test_probe_workflow_keeps_tokens_in_step_env_and_interpolates_nothing_into_scripts(probe_workflow):
    # Arrange / Act / Assert
    assert "GH_TOKEN" not in (probe_workflow.get("env") or {})
    assert "GH_TOKEN" not in (jobs(probe_workflow)["probe"].get("env") or {})
    for job_id, index, script in run_scripts(probe_workflow):
        assert "${{" not in script, f"{job_id}#{index} 표현식을 셸에 직접 삽입"


def test_probe_workflow_does_not_checkout_with_persisted_credentials(probe_workflow):
    # Arrange
    checkout = next(s for _j, _i, s in all_steps(probe_workflow) if str(s.get("uses", "")).startswith("actions/checkout"))

    # Act / Assert
    assert checkout["with"]["persist-credentials"] is False


# 감시 대상 중 이 레포에 아직 없는 것. 예측 밴드 발행(band-forecast.yml)이 이 레포로 옮겨 오기 전까지만 "미배포" 로 읽혀도 정상이다.
# 그 변경이 파일을 추가하면서 이 집합에서 뺀다(아래 마지막 시험이 남아 있는 항목을 잡는다).
PLANNED_WATCHED = {"band-forecast.yml"}


def _watched_names(doc: dict) -> list[str]:
    probe = next(s for _j, _i, s in all_steps(doc) if s.get("id") == "probe")
    return probe["env"]["WATCHED_WORKFLOWS"].split(",")


def test_watched_workflow_names_are_real_files_or_explicitly_planned(probe_workflow):
    # Arrange
    names = _watched_names(probe_workflow)

    # Act
    unknown = [name for name in names if not (WORKFLOW_DIR / name).is_file() and name not in PLANNED_WATCHED]

    # Assert — 오타·이름 변경이면 영원히 "미배포" 로 읽혀 경보가 안 난다. 아직 없는 파일은 예정 집합에만 허용한다.
    assert not unknown, unknown
    assert "ml-predict.yml" in names
    assert len(names) == len(set(names))


def test_probe_default_watched_list_matches_the_workflow_env(probe_workflow):
    # Arrange — 프로브 스크립트는 표준 라이브러리만 쓴다. 파일 경로로 적재한다(전역 import 경로를 건드리지 않는다).
    path = ROOT / "scripts" / "etl" / "probe_predictions_freshness.py"
    spec = importlib.util.spec_from_file_location("_probe_for_workflow_guard", path)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    # Act / Assert — env 를 비웠을 때(로컬 실행) 보는 목록과 워크플로가 주는 목록이 같다
    assert probe.DEFAULT_WATCHED.split(",") == _watched_names(probe_workflow)


def test_alert_issue_title_names_both_signals_and_lookup_and_create_share_it(probe_workflow):
    # Arrange
    step = next(s for _j, _i, s in all_steps(probe_workflow) if s.get("name") == "Open, refresh, or close the alert issue")
    title = step["env"]["TITLE"]
    lines = logical_lines(step["run"])

    # Act / Assert — 이 이슈는 grid(ml-predict)와 예측 밴드 두 신호를 함께 알리므로 제목도 둘을 말한다
    assert "ml-predict" in title and "예측 밴드" in title
    # 제목 문자열은 env 에만 있다 — 찾는 줄과 만드는 줄이 같은 값을 쓰지 않으면 이슈를 못 찾고 새로 쌓는다
    assert "예측 발행 신선도 경보" not in step["run"]
    assert any("select(.title == env.TITLE and " in line for line in lines)
    assert any(line.startswith("gh issue create") and '--title "$TITLE"' in line for line in lines)


def _alert_lookup(probe_workflow: dict) -> tuple[str, str, str]:
    """(`gh issue list` 줄, 그 줄의 --jq 식, 제목)."""
    step = next(s for _j, _i, s in all_steps(probe_workflow) if s.get("name") == "Open, refresh, or close the alert issue")
    lookups = [line for line in logical_lines(step["run"]) if "gh issue list" in line]
    assert len(lookups) == 1, lookups
    match = re.search(r"--jq '([^']*)'", lookups[0])
    assert match, lookups[0]
    return lookups[0], match.group(1), step["env"]["TITLE"]


def test_alert_issue_lookup_reads_the_author_and_uses_no_server_side_bot_filter(probe_workflow):
    # Arrange / Act
    line, _jq, _title = _alert_lookup(probe_workflow)

    # Assert — 서버 쪽 `--author`·`--app` 은 봇 표기가 맞지 않으면 조용히 0건이 된다(gh 2.102 에서 `--author app/github-actions` 가
    # 봇이 만든 이슈에 0건이었다). 그러면 경보 이슈를 못 찾아 매번 새로 쌓는다. 응답의 author 를 직접 본다.
    assert "--json number,title,author" in line
    assert not re.search(r"(^|\s)(--author|--app|--search|-A|-S)(\s|=)", line), line


_BOT = {"login": "app/github-actions", "is_bot": True}


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq 가 없다")
@pytest.mark.parametrize(
    ("issues", "expected"),
    [
        ([{"number": 7, "title": "T", "author": _BOT}], "7"),
        ([{"number": 9, "title": "T", "author": _BOT}, {"number": 7, "title": "T", "author": _BOT}], "9"),  # 목록은 최신순 — 가장 앞(최근) 이슈
        ([{"number": 9, "title": "T", "author": {"login": "mallory"}}, {"number": 7, "title": "T", "author": _BOT}], "7"),  # 제3자가 더 최근
        ([{"number": 9, "title": "T", "author": {"login": "mallory"}}], ""),  # 같은 제목이어도 제3자의 이슈는 건드리지 않는다
        ([{"number": 7, "title": "other", "author": _BOT}], ""),  # 봇이 만들었어도 제목이 다르면 아니다
        ([{"number": 7, "title": "T", "author": {"login": "github-actions"}}], "7"),  # `app/` 접두가 없는 표기도 같은 봇이다
        ([{"number": 7, "title": "T", "author": {"login": "app/other-bot"}}], ""),  # 다른 앱
        ([{"number": 7, "title": "T", "author": None}, {"number": 8, "title": "T"}], ""),  # 삭제된 사용자 · author 없음 — 오류 없이 건너뜀
        ([], ""),
    ],
    ids=["bot", "two-bots-newest-first", "bot-and-newer-squatter", "squatter-only", "bot-other-title", "bare-login", "other-app",
         "null-author", "none"],
)
def test_alert_issue_lookup_picks_only_the_issue_this_workflow_created(probe_workflow, issues, expected):
    # Arrange — 워크플로가 쓰는 --jq 식을 그대로 jq 로 돌린다(gh 는 같은 식을 응답 JSON 에 적용한다)
    _line, jq_expr, _title = _alert_lookup(probe_workflow)

    # Act
    proc = subprocess.run(["jq", "-r", jq_expr], input=json.dumps(issues), capture_output=True, text=True, timeout=30,
                          env={**os.environ, "TITLE": "T"})

    # Assert
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == expected


def test_planned_watched_workflows_are_removed_from_the_planned_set_once_the_file_exists():
    # Arrange / Act
    stale = sorted(name for name in PLANNED_WATCHED if (WORKFLOW_DIR / name).is_file())

    # Assert — 예정 집합이 실물보다 오래 살아남으면 오타 방지 검사가 조용히 넓어진다
    assert not stale, f"{stale} 파일이 생겼다 — PLANNED_WATCHED 에서 뺀다"


def test_keepalive_enumerates_workflows_through_the_api_instead_of_a_hardcoded_list():
    # Arrange
    doc = load(WORKFLOW_DIR / "keepalive.yml")
    script = "\n".join(line for _j, _i, s in run_scripts(doc) for line in logical_lines(s))

    # Act / Assert — 새 워크플로를 추가해도 keepalive 를 고칠 필요가 없다(전체 목록을 API 로 순회하고 파일명을 적지 않는다)
    assert 'gh api --paginate "repos/${REPO}/actions/workflows"' in script
    assert not re.search(r"[A-Za-z0-9_-]+\.ya?ml\b", script)
