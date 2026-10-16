"""HF 쓰기 토큰이 있는 잡은 `pip install` 을 해시 고정본으로만 한다 (AAA, YAML 파싱 — 네트워크 0).

**왜 이 파일이 있나.** 이 레포는 공개고, `HF_TOKEN`(HF 쓰기 토큰)을 받는 잡이 여럿이다. 그런 잡이
`pip install "huggingface_hub>=0.34,<2"` 처럼 범위로 깔면 cron 이 돌 때마다 그 시점 PyPI 의 최신 릴리스와
전이 의존 전부를 받아 토큰과 함께 실행한다 — 의존성 하나가 오염된 릴리스를 내면 다음 실행이 곧바로 토큰을
넘겨 준다. 그래서 이 잡들은 `scripts/etl/requirements-publish.lock.txt`(버전 + 파일 sha256 고정)만
`--require-hashes --no-deps --only-binary :all:` 로 깐다.

이 워크플로들은 PR 에서 돌지 않는다(cron·수동). 설치 줄이 조용히 범위 지정으로 되돌아가거나 고정 없이 새
워크플로가 생겨도 머지된 뒤 cron 이 돌 때까지 아무도 모른다 — 이 파일이 그 경계를 `pytest scripts` 로 못박는다.

규칙 — `secrets.HF_TOKEN` 이 범위 안에 있는 잡(워크플로 env · 잡 env · 어느 스텝의 env/with 에 있음)은
  1. `pip install` 이 전부 `--no-cache-dir --require-hashes --no-deps --only-binary :all: -r scripts/….lock.txt`
     한 모양이고(그 파일이 있다),
  2. 그 설치가 체크아웃 뒤에 있으며(고정본은 체크아웃에 있다),
  3. 설치 뒤에 `pip check` 를 돈다.
토큰이 없는 잡은 이 규칙 밖이다 — ml-predict 의 infer 잡이 받아 둔 wheel 을 `--no-deps` 로 까는 것처럼(그쪽은
sha256 핀으로 따로 검증한다).

한계: 워크플로 `run` 스크립트 안의 `pip install` 만 본다. 스크립트 파일이나 합성 액션이 설치하는 것은 못 본다
(지금은 그런 설치가 없다 — 생기면 이 시험도 그쪽을 읽게 넓힌다).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
ALL_WORKFLOWS = sorted(WORKFLOW_DIR.glob("*.yml"))

WRITE_SECRET = "HF_TOKEN"
INSTALL_FLAGS = ("--no-cache-dir", "--require-hashes", "--no-deps", "--only-binary :all:")
# huggingface_hub 로 HF 에 쓰는 스크립트 — 이걸 돌리는 잡이 깐 고정본에는 huggingface-hub 가 있어야 한다.
PUBLISHERS = ("hf_publish.py", "publish_contracts.py")
# 자기 고정본이 아직 없는 워크플로: 발행기용 고정본 하나로는 모자라는 의존성이 더 있다(pandas·cdsapi·pyarrow·
# jsonschema·pdfplumber 등). 각자 고정본을 만드는 후속 변경이 이 목록에서 뺀다 — 이미 고쳐졌는데 남아 있으면
# 아래 시험이 실패해 알려 준다.
UNPINNED_ALLOWLIST = ("contracts-publish.yml", "mac-data-publish.yml", "policy-collect.yml")

PIP_INSTALL_RE = re.compile(r"\bpip3?\b[^;&|]*?\binstall\b")
PIP_CHECK_RE = re.compile(r"\bpip3?\s+check\b")
LOCK_ARG_RE = re.compile(r"\s-r\s+(scripts/\S+\.lock\.txt)\s*$")


# ── 도우미 ───────────────────────────────────────────────────────────────────
def load(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), f"{path.name}: 최상위가 매핑이 아니다"
    return doc


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


def token_jobs(doc: dict) -> dict[str, dict]:
    """토큰이 범위 안에 있는 잡. 워크플로 env 에 있으면 모든 잡, 아니면 잡 안(잡·스텝의 env/with 등) 어디든 참조하는 잡."""
    workflow_wide = WRITE_SECRET in secrets_in(doc.get("env"))
    return {job_id: job for job_id, job in doc["jobs"].items() if workflow_wide or WRITE_SECRET in secrets_in(job)}


def commands(job: dict):
    """(스텝 번호, 실행 순번, 명령) — `&&`·`||`·`;`·`|` 로 이은 명령은 따로 센다."""
    order = 0
    for step_index, step in enumerate(job.get("steps", [])):
        for line in logical_lines(str(step.get("run", ""))):
            for command in re.split(r"&&|\|\||;|\|", line):
                order += 1
                yield step_index, order, command.strip()


def step_name(step: dict, index: int) -> str:
    return str(step.get("name") or step.get("id") or f"#{index}")


def lock_packages(lock: Path) -> set[str]:
    """고정본이 핀한 패키지 이름(소문자, `_` → `-`)."""
    return {m.group(1).lower().replace("_", "-") for m in re.finditer(r"^([A-Za-z0-9_.-]+)==", lock.read_text(encoding="utf-8"), re.M)}


def install_problems(command: str, root: Path = ROOT) -> list[str]:
    """`pip install` 한 명령이 해시 고정 설치 모양에서 어긋난 점. 플래그 넷 + `-r scripts/….lock.txt` 말고는 아무것도 없어야 한다."""
    rest = command[PIP_INSTALL_RE.search(command).end():]
    lock = LOCK_ARG_RE.search(rest)
    options = rest[: lock.start()] if lock else rest
    problems = []
    for flag in INSTALL_FLAGS:
        if flag not in options:
            problems.append(f"{flag} 없음")
        options = options.replace(flag, " ")
    if options.split():
        problems.append(f"고정본 밖 인자 {options.split()}")  # 직접 이름 설치·`--index-url` 같은 것
    if lock is None:
        problems.append("끝이 `-r scripts/….lock.txt` 가 아니다")
    elif not (root / lock.group(1)).is_file():
        problems.append(f"{lock.group(1)} 이 없다")
    return problems


def violations(doc: dict, root: Path = ROOT) -> list[str]:
    """토큰이 있는 잡마다 규칙을 어긴 곳을 `잡 / 스텝: 이유` 문장으로 돌려준다. 비어 있으면 통과."""
    found: list[str] = []
    for job_id, job in token_jobs(doc).items():
        steps = job.get("steps", [])
        checkout = next((i for i, s in enumerate(steps) if str(s.get("uses", "")).startswith("actions/checkout@")), len(steps))
        ran = list(commands(job))
        checks = [order for _step, order, command in ran if PIP_CHECK_RE.search(command)]
        locks: set[str] = set()
        for step_index, order, command in ran:
            if not PIP_INSTALL_RE.search(command):
                continue
            where = f"{job_id} / {step_name(steps[step_index], step_index)}"
            found += [f"{where}: {problem}" for problem in install_problems(command, root)]
            if step_index < checkout:
                found.append(f"{where}: 체크아웃보다 앞이다 — 고정본이 아직 없다")
            if not any(check > order for check in checks):
                found.append(f"{where}: 뒤에 `pip check` 가 없다")
            lock = LOCK_ARG_RE.search(command)
            if lock and (root / lock.group(1)).is_file():
                locks.add(lock.group(1))
        runs_publisher = any(name in command for _step, _order, command in ran for name in PUBLISHERS)
        if runs_publisher and locks and not any("huggingface-hub" in lock_packages(root / rel) for rel in locks):
            found.append(f"{job_id}: 발행기를 돌리는데 설치하는 고정본({', '.join(sorted(locks))})에 huggingface-hub 가 없다")
    return found


# ── 실제 워크플로 ────────────────────────────────────────────────────────────
def test_the_scan_sees_the_token_in_every_placement_the_workflows_use():
    # Arrange / Act
    holders = {p.name for p in ALL_WORKFLOWS if token_jobs(load(p))}

    # Assert — 글롭·범위 판정이 깨져 대상이 빠지면 아래 parametrize 가 무음 통과한다.
    # 잡 env(data-collect-hourly) · 스텝 env(openaq-bulk-collect) · 잡 분리(ml-predict) + 허용목록
    assert {"data-collect-hourly.yml", "openaq-bulk-collect.yml", "ml-predict.yml", *UNPINNED_ALLOWLIST} <= holders


@pytest.mark.parametrize("path", [p for p in ALL_WORKFLOWS if p.name not in UNPINNED_ALLOWLIST], ids=lambda p: p.name)
def test_jobs_with_the_write_token_install_only_the_hash_locked_file(path):
    # Arrange
    doc = load(path)

    # Act
    problems = violations(doc)

    # Assert — 토큰이 없는 워크플로는 빈 목록으로 통과한다
    assert not problems, f"{path.name}:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("name", UNPINNED_ALLOWLIST)
def test_allowlisted_workflow_exists_and_is_still_unpinned(name):
    # Arrange
    path = WORKFLOW_DIR / name

    # Act / Assert
    assert path.is_file(), f"{name} 이 없다 — UNPINNED_ALLOWLIST 에서 뺀다"
    assert violations(load(path)), f"{name} 의 설치가 이미 해시 고정이다 — UNPINNED_ALLOWLIST 에서 뺀다(뺀 뒤엔 위 시험이 이 워크플로를 지킨다)"


def test_ml_predict_keeps_the_token_out_of_the_infer_job_whose_local_wheel_install_is_not_hash_locked():
    # Arrange
    doc = load(WORKFLOW_DIR / "ml-predict.yml")

    # Act
    scoped = set(token_jobs(doc))

    # Assert — infer 잡은 받아 둔 wheel 을 `--no-deps` 로 깐다(sha256 핀으로 따로 검증). 토큰이 거기로 들어가면 이 규칙이 그 설치를 잡는다
    assert scoped == {"publish"}
    assert violations(doc) == []


# ── 검사기 자체의 시험 ───────────────────────────────────────────────────────
# 검사기가 정규식 오타로 무력해지지 않았음을 합성 워크플로로 보인다(실제 파일에 대한 변이 시험은 따로 돌린다).
TOKEN = "${{ secrets.HF_TOKEN }}"
CHECKOUT = {"uses": "actions/checkout@" + "0" * 40}
LOCKED = "python3 -m pip install --no-cache-dir --require-hashes --no-deps --only-binary :all: -r scripts/with-hub.lock.txt"
PIP_CHECK = "python3 -m pip check"
PUBLISH_STEP = {"run": "python3 scripts/etl/hf_publish.py upload --src a.json --dest b.json"}


def run(script: str) -> dict:
    return {"run": script}


def job_doc(*steps: dict, token: str = "job") -> dict:
    """토큰이 있는 잡 하나짜리 워크플로 — `token` 은 어디에 둘지(workflow · job · step · with)."""
    publish = dict(PUBLISH_STEP)
    job: dict = {"steps": [*steps, publish]}
    doc: dict = {"jobs": {"publish": job}}
    if token == "workflow":
        doc["env"] = {WRITE_SECRET: TOKEN}
    elif token == "job":
        job["env"] = {WRITE_SECRET: TOKEN}
    elif token == "step":
        publish["env"] = {WRITE_SECRET: TOKEN}
    elif token == "with":
        publish["with"] = {"token": TOKEN}
    return doc


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    for name, package in (("with-hub", "huggingface-hub==1.0.0"), ("without-hub", "numpy==2.0.0")):
        (tmp_path / "scripts" / f"{name}.lock.txt").write_text(f"{package} \\\n    --hash=sha256:{'0' * 64}\n", encoding="utf-8")
    return tmp_path


def test_the_checker_accepts_a_locked_install_followed_by_pip_check(root):
    # Arrange — 한 줄이든 `\` 이음줄이든, 같은 스크립트 안이든 다음 스텝이든
    split_install = LOCKED.replace(" --no-deps", " \\\n  --no-deps")
    docs = [
        job_doc(CHECKOUT, run(f"{LOCKED}\n{PIP_CHECK}")),
        job_doc(CHECKOUT, run(split_install), run(PIP_CHECK)),
        job_doc(CHECKOUT, run(f"set -euo pipefail\n# 주석\n{LOCKED}\n{PIP_CHECK}")),
    ]

    # Act / Assert
    assert [violations(doc, root) for doc in docs] == [[], [], []]


def checked(script: str) -> dict:
    """설치 스크립트 뒤에 `pip check` 를 붙인 스텝."""
    return run(f"{script}\n{PIP_CHECK}")


# 사례 이름 → (잡의 스텝들, 문제 목록에 들어 있어야 할 문구). 문구까지 보므로 엉뚱한 이유로 잡혀도 통과하지 않는다.
DEVIATIONS = {
    "range-specifier": ([CHECKOUT, checked('pip install "huggingface_hub>=0.34,<2"')], "--require-hashes 없음"),
    "python-m-pip-range-specifier": ([CHECKOUT, checked('python3 -m pip install --quiet "huggingface_hub>=0.34,<2"')], "--require-hashes 없음"),
    **{f"missing{flag.split()[0]}": ([CHECKOUT, checked(LOCKED.replace(f"{flag} ", ""))], f"{flag} 없음") for flag in INSTALL_FLAGS},
    "named-requirement": ([CHECKOUT, checked(LOCKED.replace("-r ", "requests -r "))], "고정본 밖 인자"),
    "index-override": ([CHECKOUT, checked(LOCKED.replace(" -r ", " --extra-index-url https://example.invalid/simple -r "))], "고정본 밖 인자"),
    "chained-unpinned-install": ([CHECKOUT, checked(f"{LOCKED} && pip install requests")], "--require-hashes 없음"),
    "later-step-unpinned-install": ([CHECKOUT, checked(LOCKED), run("pip install requests")], "--require-hashes 없음"),
    "not-a-lock-file": ([CHECKOUT, checked(LOCKED.replace("scripts/with-hub.lock.txt", "requirements.txt"))], "끝이 `-r scripts/….lock.txt` 가 아니다"),
    "missing-lock-file": ([CHECKOUT, checked(LOCKED.replace("with-hub", "missing"))], "이 없다"),
    "lock-without-the-hub": ([CHECKOUT, checked(LOCKED.replace("with-hub", "without-hub"))], "huggingface-hub 가 없다"),
    "no-pip-check": ([CHECKOUT, run(LOCKED)], "pip check"),
    "pip-check-before-install": ([CHECKOUT, run(f"{PIP_CHECK}\n{LOCKED}")], "pip check"),
    "install-before-checkout": ([checked(LOCKED), CHECKOUT], "체크아웃보다 앞이다"),
    "no-checkout": ([checked(LOCKED)], "체크아웃보다 앞이다"),
}


@pytest.mark.parametrize(("steps", "expected"), DEVIATIONS.values(), ids=list(DEVIATIONS))
def test_the_checker_flags_each_deviation_for_the_right_reason(root, steps, expected):
    # Arrange
    doc = job_doc(*steps)

    # Act
    problems = violations(doc, root)

    # Assert
    assert any(expected in problem for problem in problems), problems


@pytest.mark.parametrize("token", ["workflow", "job", "step", "with"])
def test_the_scope_check_sees_the_token_wherever_it_is_set(root, token):
    # Arrange — 설치 스텝이 아닌 곳에 토큰이 있어도(openaq-bulk-collect 처럼) 잡 전체가 대상이다
    doc = job_doc(CHECKOUT, run('pip install "huggingface_hub>=0.34,<2"'), token=token)

    # Act
    problems = violations(doc, root)

    # Assert
    assert problems


def test_a_job_without_the_write_token_is_not_checked(root):
    # Arrange — 다른 시크릿(비공개 wheel 읽기 토큰)은 쓰기 토큰이 아니다
    doc = {"jobs": {"infer": {
        "env": {"HF_MODELS_READ": "${{ secrets.HF_MODELS_READ }}"},
        "steps": [CHECKOUT, run('python3 -m pip install --no-cache-dir --no-deps "$WHEEL"')],
    }}}

    # Act
    scoped = token_jobs(doc)

    # Assert
    assert scoped == {}
    assert violations(doc, root) == []
