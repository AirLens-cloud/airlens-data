"""HF 쓰기 토큰이 있는 잡은 `pip install` 을 해시 고정본으로만 한다 (AAA, YAML 파싱 — 네트워크 0).

**왜 이 파일이 있나.** 이 레포는 공개고, `HF_TOKEN`(HF 쓰기 토큰)을 받는 잡이 여럿이다. 그런 잡이
`pip install "huggingface_hub>=0.34,<2"` 처럼 범위로 깔면 cron 이 돌 때마다 그 시점 PyPI 의 최신 릴리스와
전이 의존 전부를 받아 토큰과 함께 실행한다 — 의존성 하나가 오염된 릴리스를 내면 다음 실행이 곧바로 토큰을
넘겨 준다. 그래서 이 잡들은 `scripts/etl/requirements-publish.lock.txt`(버전 + 파일 sha256 고정)만
`--require-hashes --no-deps --only-binary :all:` 로 깐다.

이 워크플로들은 PR 에서 돌지 않는다(cron·수동). 설치 줄이 조용히 범위 지정으로 되돌아가거나 고정 없이 새
워크플로가 생겨도 머지된 뒤 cron 이 돌 때까지 아무도 모른다 — 이 파일이 그 경계를 `pytest scripts` 로 못박는다.

규칙 — 토큰이 범위 안에 있는 잡(`secrets.HF_TOKEN` 을 워크플로 env · 잡 env · 어느 스텝의 env/with 에서 참조)은
  1. `pip install` 이 전부 `--no-cache-dir --require-hashes --no-deps --only-binary :all: -r scripts/….lock.txt`
     한 모양이고(그 파일이 있다) 정확히 이 철자만 허용한다 — `--only-binary=:all:` 같은 동등 표기도 거부한다,
  2. 그 설치가 체크아웃 뒤에 있으며(고정본은 체크아웃에 있다),
  3. 설치 뒤에 `pip check` 를 돈다(명령이 `pip`·`pip3`·`python3 -m pip` 로 시작해야 인정 — `echo "pip check"` 는 아니다),
  4. 설치·`pip check` 의 실패를 가리지 않는다(그 줄에 `||`·`|`, 그 스텝에 `continue-on-error`),
  5. 발행기(`hf_publish.py` 등)를 돌면 고정본 설치가 있고 그 고정본에 huggingface-hub 가 들어 있으며,
  6. 해시 고정본(requirements-publish.lock.txt)은 만들어진 CPython 3.11 로 깐다(앞선 setup-python 이 3.11),
  7. pip 밖 설치기(pipx·uv·poetry·conda·mamba·npx·`curl|sh`)로 받지 않는다.
토큰 범위는 `secrets.NAME`·`secrets['NAME']`(이름은 대소문자 무시)·`toJSON(secrets)`·`secrets: inherit` 로 읽는다.
재사용 워크플로(`uses:`)로 토큰을 넘기는 잡은 안쪽을 못 보므로 그것만으로 위반이다.
토큰이 없는 잡은 이 규칙 밖이다 — ml-predict 의 infer 잡이 받아 둔 wheel 을 `--no-deps` 로 까는 것처럼(그쪽은
sha256 핀으로 따로 검증한다).

한계: 워크플로 `run` 스크립트 안의 설치만 본다. 스크립트 파일(`bash scripts/x.sh`)이나 액션(`uses:` 스텝, 합성 액션)
안에서 하는 설치, 위 목록 밖 설치기, 동적 색인(`secrets[format(...)]`)은 못 본다(지금은 그런 곳이 없다 — 생기면 이
시험도 그쪽을 읽게 넓힌다). `.github/workflows` 의 `.yml`·`.yaml` 만 읽고, 다른 이름의 파일이 생기면 시험이 실패한다.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
ALL_WORKFLOWS = sorted([*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")])

WRITE_SECRET = "HF_TOKEN"
PUBLISH_LOCK = "scripts/etl/requirements-publish.lock.txt"
# 고정본 → 그 고정본을 만든 CPython. 새 고정본이 생기면 여기에 한 줄 더한다.
LOCK_PYTHON = {PUBLISH_LOCK: "3.11"}
INSTALL_FLAGS = ("--no-cache-dir", "--require-hashes", "--no-deps", "--only-binary :all:")
# huggingface_hub 로 HF 에 쓰는 스크립트 — 이걸 돌리는 잡이 깐 고정본에는 huggingface-hub 가 있어야 한다.
PUBLISHERS = ("hf_publish.py", "publish_contracts.py")

PIP_INSTALL_RE = re.compile(r"\bpip3?\b[^;&|]*?\binstall\b")
PIP_USE_RE = re.compile(r"\bpip3?\b[^;&|]*?\b(?:install|check)\b")
PIP_CHECK_RE = re.compile(r"^(?:python[0-9.]*\s+-m\s+)?pip3?\s+check\b")
LOCK_ARG_RE = re.compile(r"\s-r\s+(scripts/\S+\.lock\.txt)\s*$")
# pip 밖에서 패키지·스크립트를 받는 도구. apt 는 여기 없다(고정은 별개 문제).
OTHER_INSTALLER_RE = re.compile(
    r"\b(?:pipx|poetry|conda|(?:micro)?mamba|npx)\b"
    r"|\buv\s+(?:pip|sync|run|tool|add)\b"
    r"|\b(?:curl|wget)\b[^\n]*\|\s*(?:sudo\s+)?(?:ba|z)?sh\b"
)
# `secrets.NAME` · `secrets['NAME']` · `secrets["NAME"]` — 이름은 대소문자를 가리지 않는다
SECRET_RE = re.compile(r"\bsecrets\s*\.\s*([A-Za-z0-9_]+)|\bsecrets\s*\[\s*['\"]([A-Za-z0-9_]+)['\"]\s*\]", re.I)
ALL_SECRETS_RE = re.compile(r"\btoJSON\s*\(\s*secrets\s*\)", re.I)


# ── 도우미 ───────────────────────────────────────────────────────────────────
# ponytail: ROOT·WORKFLOW_DIR·load·logical_lines·step_name 은 scripts/ml/test_ml_workflow_safety.py 사본, secrets_in 과 ALL_WORKFLOWS 는 그 사본보다 넓게 읽는다(secrets['X']·대소문자·toJSON·.yaml) — upgrade: 그 파일 수정이 머지된 뒤 scripts/ 공용 모듈로 합친다.
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


def strings(value: object):
    """중첩된 매핑·목록 안의 모든 문자열(키 포함)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from strings(key)
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def secrets_in(value: object) -> set[str]:
    """`value` 안 문자열이 가리키는 시크릿 이름(대문자). `toJSON(secrets)` 는 전부라는 뜻의 `*`."""
    names: set[str] = set()
    for text in strings(value):
        names |= {(dotted or indexed).upper() for dotted, indexed in SECRET_RE.findall(text)}
        if ALL_SECRETS_RE.search(text):
            names.add("*")
    return names


def holds_token(value: object) -> bool:
    names = secrets_in(value)
    return WRITE_SECRET in names or "*" in names


def token_jobs(doc: dict) -> dict[str, dict]:
    """토큰이 범위 안에 있는 잡. 워크플로 env 에 있으면 모든 잡, 아니면 잡 안(잡·스텝의 env/with 등) 어디든 참조하거나
    `secrets: inherit` 로 전부 물려받는 잡."""
    workflow_wide = holds_token(doc.get("env"))
    return {
        job_id: job for job_id, job in doc["jobs"].items()
        if workflow_wide or holds_token(job) or job.get("secrets") == "inherit"
    }


def lines(job: dict):
    """(스텝 번호, 논리 줄) — run 스크립트의 줄."""
    for step_index, step in enumerate(job.get("steps", [])):
        for line in logical_lines(str(step.get("run", ""))):
            yield step_index, line


def commands(job: dict):
    """(스텝 번호, 실행 순번, 명령) — `&&`·`||`·`;`·`|` 로 이은 명령은 따로 센다."""
    order = 0
    for step_index, line in lines(job):
        for command in re.split(r"&&|\|\||;|\|", line):
            order += 1
            yield step_index, order, command.strip()


def step_name(step: dict, index: int) -> str:
    return str(step.get("name") or step.get("id") or f"#{index}")


def lock_packages(lock: Path) -> set[str]:
    """고정본이 핀한 패키지 이름(소문자, `_` → `-`)."""
    return {m.group(1).lower().replace("_", "-") for m in re.finditer(r"^([A-Za-z0-9_.-]+)==", lock.read_text(encoding="utf-8"), re.M)}


def python_version(steps: list[dict]) -> str:
    """`steps` 중 마지막 setup-python 의 python-version(없으면 빈 문자열)."""
    versions = [
        str((step.get("with") or {}).get("python-version", ""))
        for step in steps if str(step.get("uses", "")).startswith("actions/setup-python@")
    ]
    return versions[-1] if versions else ""


def install_problems(command: str, root: Path = ROOT) -> tuple[list[str], str | None]:
    """`pip install` 한 명령이 해시 고정 설치 모양에서 어긋난 점과, 마지막 인자가 가리키는(있는) 고정본 경로.
    플래그 넷 + `-r scripts/….lock.txt` 말고는 아무것도 없어야 한다."""
    rest = command[PIP_INSTALL_RE.search(command).end():]
    arg = LOCK_ARG_RE.search(rest)
    options = rest[: arg.start()] if arg else rest
    problems = []
    for flag in INSTALL_FLAGS:
        if flag not in options:
            problems.append(f"{flag} 없음")
        options = options.replace(flag, " ")
    if options.split():
        problems.append(f"고정본 밖 인자 {options.split()}")  # 직접 이름 설치·`--index-url` 같은 것
    lock = arg.group(1) if arg else None
    if lock is None:
        problems.append("끝이 `-r scripts/….lock.txt` 가 아니다(따옴표·리다이렉트·주석 없이 마지막 인자여야 한다)")
    elif not (root / lock).is_file():
        problems.append(f"{lock} 이 없다")
        lock = None
    return problems, lock


def job_violations(job_id: str, job: dict, root: Path) -> list[str]:
    steps = job.get("steps", [])
    found: list[str] = []

    def where(index: int) -> str:
        return f"{job_id} / {step_name(steps[index], index)}"

    checkout = next((i for i, s in enumerate(steps) if str(s.get("uses", "")).startswith("actions/checkout@")), len(steps))
    ran = list(commands(job))
    checks = [order for _step, order, command in ran if PIP_CHECK_RE.search(command)]
    locks: set[str] = set()
    for index, order, command in ran:
        if not PIP_INSTALL_RE.search(command):
            continue
        problems, lock = install_problems(command, root)
        found += [f"{where(index)}: {problem}" for problem in problems]
        if index < checkout:
            found.append(f"{where(index)}: 체크아웃보다 앞이다 — 고정본이 아직 없다")
        if not any(check > order for check in checks):
            found.append(f"{where(index)}: 뒤에 `pip check` 가 없다")
        if lock:
            locks.add(lock)
            wanted = LOCK_PYTHON.get(lock)
            version = python_version(steps[:index])
            if wanted and not re.fullmatch(re.escape(wanted) + r"(\.\d+)?", version):
                found.append(f"{where(index)}: {lock} 은 CPython {wanted} 용이다 — 앞선 setup-python 의 python-version: {version or '(없음)'}")

    for index, line in lines(job):
        installer = OTHER_INSTALLER_RE.search(line)
        if installer:
            found.append(f"{where(index)}: pip 밖 설치기 `{installer.group()}` — 고정본 밖에서 받는다")
        if "|" in line and PIP_USE_RE.search(line):
            found.append(f"{where(index)}: `||`·`|` 가 `pip install`/`pip check` 의 실패를 가린다")

    for index in sorted({index for index, _order, command in ran if PIP_USE_RE.search(command)}):
        flag = steps[index].get("continue-on-error")
        if flag is not None and str(flag).strip().lower() != "false":
            found.append(f"{where(index)}: continue-on-error 가 `pip install`/`pip check` 의 실패를 가린다")

    if any(name in command for _step, _order, command in ran for name in PUBLISHERS):
        if not locks:
            found.append(f"{job_id}: 발행기를 돌리는데 고정본 설치가 없다")
        elif not any("huggingface-hub" in lock_packages(root / rel) for rel in locks):
            found.append(f"{job_id}: 발행기를 돌리는데 설치하는 고정본({', '.join(sorted(locks))})에 huggingface-hub 가 없다")
    return found


def violations(doc: dict, root: Path = ROOT) -> list[str]:
    """토큰이 있는 잡마다 규칙을 어긴 곳을 `잡 / 스텝: 이유` 문장으로 돌려준다. 비어 있으면 통과."""
    found: list[str] = []
    for job_id, job in token_jobs(doc).items():
        if "uses" in job:
            found.append(f"{job_id}: 재사용 워크플로(`uses:`)로 토큰이 넘어가는 잡이라 안쪽 설치를 검사할 수 없다")
        else:
            found += job_violations(job_id, job, root)
    return found


# ── 실제 워크플로 ────────────────────────────────────────────────────────────
# 해시 고정본으로 바꾼 워크플로 — 토큰 범위 판정이 이 목록을 놓치면 아래 시험이 무음 통과하므로 따로 못박는다.
HASH_LOCKED_WORKFLOWS = (
    "data-collect-hourly.yml", "firms-collect.yml", "forecast-collect.yml", "hf-live-squash.yml",
    "news-collect.yml", "openaq-bulk-collect.yml", "shadow-retention.yml", "timeline-collect.yml",
)

# 자기 고정본이 아직 없는 워크플로: 발행기용 고정본 하나로는 모자라는 의존성이 더 있다(pandas·cdsapi·pyarrow·
# jsonschema·pdfplumber 등). 값은 지금 이 워크플로가 가진 위반 문장 전부다 — 새 위반이 생기면(고친다), 위반이 줄거나
# 사라지면(목록을 줄이거나 이 워크플로를 뺀다) 시험이 실패해 알려 준다. 각자 고정본을 만드는 후속 변경이 뺀다.
_CONTRACTS = "publish-contracts / Install dependencies"
_MAC_DEPS = "collect-and-build / Install Python deps (versions pinned to match models/uv.lock's ubuntu-x86_64/py3.11 resolution)"
_MAC_HUB = "collect-and-build / Install huggingface_hub"
_POLICY = "collect-policies / Install dependencies"
_NOT_LOCK = "끝이 `-r scripts/….lock.txt` 가 아니다(따옴표·리다이렉트·주석 없이 마지막 인자여야 한다)"
UNPINNED_ALLOWLIST: dict[str, list[str]] = {
    "contracts-publish.yml": [
        f"{_CONTRACTS}: --no-cache-dir 없음",
        f"{_CONTRACTS}: --require-hashes 없음",
        f"{_CONTRACTS}: --no-deps 없음",
        f"{_CONTRACTS}: --only-binary :all: 없음",
        f"{_CONTRACTS}: 고정본 밖 인자 ['--quiet', '-r', 'requirements.txt']",
        f"{_CONTRACTS}: {_NOT_LOCK}",
        f"{_CONTRACTS}: 뒤에 `pip check` 가 없다",
        "publish-contracts: 발행기를 돌리는데 고정본 설치가 없다",
    ],
    "mac-data-publish.yml": [
        f"{_MAC_DEPS}: --no-cache-dir 없음",
        f"{_MAC_DEPS}: --require-hashes 없음",
        f"{_MAC_DEPS}: --no-deps 없음",
        f"{_MAC_DEPS}: --only-binary :all: 없음",
        f"{_MAC_DEPS}: 고정본 밖 인자 ['cdsapi==0.7.7', 'pandas==3.0.2', 'pyarrow==23.0.1']",
        f"{_MAC_DEPS}: {_NOT_LOCK}",
        f"{_MAC_DEPS}: 뒤에 `pip check` 가 없다",
        f"{_MAC_HUB}: --no-cache-dir 없음",
        f"{_MAC_HUB}: --require-hashes 없음",
        f"{_MAC_HUB}: --no-deps 없음",
        f"{_MAC_HUB}: --only-binary :all: 없음",
        f"{_MAC_HUB}: 고정본 밖 인자 ['\"huggingface_hub>=0.34,<2\"']",
        f"{_MAC_HUB}: {_NOT_LOCK}",
        f"{_MAC_HUB}: 뒤에 `pip check` 가 없다",
        "collect-and-build: 발행기를 돌리는데 고정본 설치가 없다",
    ],
    "policy-collect.yml": [
        f"{_POLICY}: --no-cache-dir 없음",
        f"{_POLICY}: --require-hashes 없음",
        f"{_POLICY}: --no-deps 없음",
        f"{_POLICY}: --only-binary :all: 없음",
        f"{_POLICY}: 고정본 밖 인자 ['--quiet', 'requests', 'pyarrow', 'pdfplumber', '\"huggingface_hub>=0.34,<2\"']",
        f"{_POLICY}: {_NOT_LOCK}",
        f"{_POLICY}: 뒤에 `pip check` 가 없다",
        "collect-policies: 발행기를 돌리는데 고정본 설치가 없다",
    ],
}


def test_the_glob_covers_every_file_in_the_workflow_directory():
    # Arrange
    on_disk = {p.name for p in WORKFLOW_DIR.iterdir() if p.is_file()}

    # Act
    globbed = {p.name for p in ALL_WORKFLOWS}

    # Assert — GitHub 은 .yml/.yaml 만 워크플로로 읽는다. 다른 이름이 글롭 밖에 있으면 아래 시험이 그 파일을 무음으로 건너뛴다
    assert on_disk == globbed, f"글롭 밖 파일: {sorted(on_disk - globbed)}"


def test_the_scan_sees_the_token_in_every_placement_the_workflows_use():
    # Arrange / Act
    holders = {p.name for p in ALL_WORKFLOWS if token_jobs(load(p))}

    # Assert — 글롭·범위 판정이 깨져 대상이 빠지면 아래 parametrize 가 무음 통과한다.
    # 잡 env(data-collect-hourly) · 스텝 env(openaq-bulk-collect) · 잡 분리(ml-predict) + 고정본으로 바꾼 8개 + 허용목록
    assert {*HASH_LOCKED_WORKFLOWS, "ml-predict.yml", *UNPINNED_ALLOWLIST} <= holders


@pytest.mark.parametrize("path", [p for p in ALL_WORKFLOWS if p.name not in UNPINNED_ALLOWLIST], ids=lambda p: p.name)
def test_jobs_with_the_write_token_install_only_the_hash_locked_file(path):
    # Arrange
    doc = load(path)

    # Act
    problems = violations(doc)

    # Assert — 토큰이 없는 워크플로는 빈 목록으로 통과한다
    assert not problems, f"{path.name}:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("name", UNPINNED_ALLOWLIST)
def test_allowlisted_workflow_has_exactly_its_known_violations(name):
    # Arrange
    path = WORKFLOW_DIR / name
    assert path.is_file(), f"{name} 이 없다 — UNPINNED_ALLOWLIST 에서 뺀다"
    expected = UNPINNED_ALLOWLIST[name]

    # Act
    actual = violations(load(path))
    new, gone = Counter(actual) - Counter(expected), Counter(expected) - Counter(actual)

    # Assert — 허용목록은 지금 있는 위반만 봐준다. 새 위반은 고치고, 사라진 위반은 목록에서 지운다
    report = [f"새 위반이 생겼다 — 고친다: {v}" for v in new] + [f"위반이 사라졌다 — 목록에서 지운다: {v}" for v in gone]
    if expected and not actual:
        report.append("위반이 하나도 없다(고쳐졌거나 토큰이 없어졌다) — UNPINNED_ALLOWLIST 에서 뺀다. 뺀 뒤엔 위 시험이 이 워크플로를 지킨다")
    assert not report, f"{name}:\n  " + "\n  ".join(report)


def test_ml_predict_keeps_the_token_out_of_the_infer_job_whose_local_wheel_install_is_not_hash_locked():
    # Arrange
    doc = load(WORKFLOW_DIR / "ml-predict.yml")

    # Act
    scoped = set(token_jobs(doc))

    # Assert — infer 잡은 받아 둔 wheel 을 `--no-deps` 로 깐다(sha256 핀으로 따로 검증). 토큰이 거기로 들어가면 이 규칙이 그 설치를 잡는다
    assert scoped == {"publish"}


# ── 검사기 자체의 시험 ───────────────────────────────────────────────────────
# 검사기가 정규식 오타로 무력해지지 않았음을 합성 워크플로로 보인다.
TOKEN = "${{ secrets.HF_TOKEN }}"
UNPINNED_HUB = 'pip install "huggingface_hub>=0.34,<2"'
CHECKOUT = {"uses": "actions/checkout@" + "0" * 40}
SETUP_PYTHON = {"uses": "actions/setup-python@" + "0" * 40, "with": {"python-version": "3.11"}}
PREP = [CHECKOUT, SETUP_PYTHON]
LOCKED = f"python3 -m pip install --no-cache-dir --require-hashes --no-deps --only-binary :all: -r {PUBLISH_LOCK}"
PIP_CHECK = "python3 -m pip check"
PUBLISH_STEP = {"run": "python3 scripts/etl/hf_publish.py upload --src a.json --dest b.json"}


def run(script: str) -> dict:
    return {"run": script}


def checked(script: str) -> dict:
    """설치 스크립트 뒤에 `pip check` 를 붙인 스텝."""
    return run(f"{script}\n{PIP_CHECK}")


def job_doc(*steps: dict, token: str = "job", expr: str = TOKEN) -> dict:
    """토큰이 있는 잡 하나짜리 워크플로 — `token` 은 어디에 둘지(workflow · job · step · with), `expr` 는 어떻게 쓸지."""
    publish = dict(PUBLISH_STEP)
    job: dict = {"steps": [*steps, publish]}
    doc: dict = {"jobs": {"publish": job}}
    if token == "workflow":
        doc["env"] = {WRITE_SECRET: expr}
    elif token == "job":
        job["env"] = {WRITE_SECRET: expr}
    elif token == "step":
        publish["env"] = {WRITE_SECRET: expr}
    elif token == "with":
        publish["with"] = {"token": expr}
    return doc


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for rel, package in ((PUBLISH_LOCK, "huggingface-hub==1.0.0"), ("scripts/without-hub.lock.txt", "numpy==2.0.0")):
        lock = tmp_path / rel
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(f"{package} \\\n    --hash=sha256:{'0' * 64}\n", encoding="utf-8")
    return tmp_path


def test_the_checker_accepts_a_locked_install_followed_by_pip_check(root):
    # Arrange — 한 줄이든 `\` 이음줄이든, 같은 스크립트 안이든 다음 스텝이든, `pip check` 의 어느 호출 형태든,
    # 파이썬 버전이 3.11 계열이든, apt 설치가 곁들여졌든
    split_install = LOCKED.replace(" --no-deps", " \\\n  --no-deps")
    docs = [
        job_doc(*PREP, run(f"{LOCKED}\n{PIP_CHECK}")),
        job_doc(*PREP, run(split_install), run(PIP_CHECK)),
        job_doc(*PREP, run(f"set -euo pipefail\n# 주석\n{LOCKED}\n{PIP_CHECK}")),
        job_doc(*PREP, run(f"{LOCKED}\npip check")),
        job_doc(*PREP, run(f"{LOCKED}\npip3 check")),
        job_doc(*PREP, run(f"{LOCKED}\npython3.11 -m pip check")),
        job_doc(*PREP, run(f"sudo apt-get install -y -qq libeccodes-tools\n{LOCKED}\n{PIP_CHECK}")),
        job_doc(*PREP, {**checked(LOCKED), "continue-on-error": False}),
        job_doc(CHECKOUT, {**SETUP_PYTHON, "with": {"python-version": "3.11.9"}}, checked(LOCKED)),
        job_doc(CHECKOUT, {**SETUP_PYTHON, "with": {"python-version": 3.11}}, checked(LOCKED)),
        job_doc(CHECKOUT, {**SETUP_PYTHON, "with": {"python-version": "3.12"}}, SETUP_PYTHON, checked(LOCKED)),
    ]

    # Act
    found = [violations(doc, root) for doc in docs]

    # Assert
    assert found == [[]] * len(docs)


OTHER_INSTALLERS = {
    "pipx": "pipx install requests",
    "uv-sync": "uv sync",
    "uv-run": "uv run script.py",
    "uv-tool": "uv tool install ruff",
    "uv-add": "uv add requests",
    "poetry": "poetry install",
    "conda": "conda install numpy",
    "mamba": "mamba install numpy",
    "npx": "npx cowsay hi",
    "curl-pipe-sh": "curl -fsSL https://example.invalid/install.sh | sh",
    "wget-pipe-sudo-bash": "wget -qO- https://example.invalid/install.sh | sudo bash",
}

# 사례 이름 → (잡의 스텝들, 문제 목록에 들어 있어야 할 문구). 문구까지 보므로 엉뚱한 이유로 잡혀도 통과하지 않는다.
DEVIATIONS = {
    "range-specifier": ([*PREP, checked(UNPINNED_HUB)], "--require-hashes 없음"),
    "python-m-pip-range-specifier": ([*PREP, checked('python3 -m pip install --quiet "huggingface_hub>=0.34,<2"')], "--require-hashes 없음"),
    **{f"missing-{flag.split()[0].lstrip('-')}": ([*PREP, checked(LOCKED.replace(f"{flag} ", ""))], f"{flag} 없음") for flag in INSTALL_FLAGS},
    "only-binary-equals-form": ([*PREP, checked(LOCKED.replace("--only-binary :all:", "--only-binary=:all:"))], "--only-binary :all: 없음"),
    "named-requirement": ([*PREP, checked(LOCKED.replace("-r ", "requests -r "))], "고정본 밖 인자"),
    "index-override": ([*PREP, checked(LOCKED.replace(" -r ", " --extra-index-url https://example.invalid/simple -r "))], "고정본 밖 인자"),
    "chained-unpinned-install": ([*PREP, checked(f"{LOCKED} && pip install requests")], "--require-hashes 없음"),
    "later-step-unpinned-install": ([*PREP, checked(LOCKED), run("pip install requests")], "--require-hashes 없음"),
    "uv-pip-install": ([*PREP, checked(LOCKED), run("uv pip install requests")], "--require-hashes 없음"),
    "not-a-lock-file": ([*PREP, checked(LOCKED.replace(PUBLISH_LOCK, "requirements.txt"))], "끝이 `-r scripts/….lock.txt` 가 아니다"),
    "quoted-lock-path": ([*PREP, checked(LOCKED.replace(PUBLISH_LOCK, f'"{PUBLISH_LOCK}"'))], "마지막 인자여야 한다"),
    "redirect-after-lock-path": ([*PREP, checked(f"{LOCKED} > install.log")], "마지막 인자여야 한다"),
    "missing-lock-file": ([*PREP, checked(LOCKED.replace("requirements-publish", "missing"))], "missing.lock.txt 이 없다"),
    "lock-without-the-hub": ([*PREP, checked(LOCKED.replace(PUBLISH_LOCK, "scripts/without-hub.lock.txt"))], "huggingface-hub 가 없다"),
    "no-pip-check": ([*PREP, run(LOCKED)], "뒤에 `pip check` 가 없다"),
    "pip-check-before-install": ([*PREP, run(f"{PIP_CHECK}\n{LOCKED}")], "뒤에 `pip check` 가 없다"),
    "pip-check-only-echoed": ([*PREP, run(f'{LOCKED}\necho "pip check"')], "뒤에 `pip check` 가 없다"),
    "install-before-checkout": ([SETUP_PYTHON, checked(LOCKED), CHECKOUT], "체크아웃보다 앞이다"),
    "no-checkout": ([SETUP_PYTHON, checked(LOCKED)], "체크아웃보다 앞이다"),
    "install-or-true": ([*PREP, run(f"{LOCKED} || true\n{PIP_CHECK}")], "`||`·`|`"),
    "pip-check-or-true": ([*PREP, run(f"{LOCKED}\n{PIP_CHECK} || true")], "`||`·`|`"),
    "install-piped-to-tee": ([*PREP, run(f"{LOCKED} | tee install.log\n{PIP_CHECK}")], "`||`·`|`"),
    "install-continue-on-error": ([*PREP, {**checked(LOCKED), "continue-on-error": True}], "continue-on-error"),
    "pip-check-continue-on-error": ([*PREP, run(LOCKED), {**run(PIP_CHECK), "continue-on-error": "true"}], "continue-on-error"),
    "continue-on-error-expression": ([*PREP, {**checked(LOCKED), "continue-on-error": "${{ github.event_name == 'schedule' }}"}], "continue-on-error"),
    **{f"other-installer-{name}": ([*PREP, checked(LOCKED), run(script)], "pip 밖 설치기") for name, script in OTHER_INSTALLERS.items()},
    "publisher-without-any-install": ([*PREP], "발행기를 돌리는데 고정본 설치가 없다"),
    "publisher-with-an-unrecognized-installer-only": ([*PREP, run("pipx install huggingface_hub")], "발행기를 돌리는데 고정본 설치가 없다"),
    "python-3-12": ([CHECKOUT, {**SETUP_PYTHON, "with": {"python-version": "3.12"}}, checked(LOCKED)], "CPython 3.11"),
    "no-setup-python": ([CHECKOUT, checked(LOCKED)], "CPython 3.11"),
    "setup-python-after-install": ([CHECKOUT, checked(LOCKED), SETUP_PYTHON], "CPython 3.11"),
    "python-version-expression": ([CHECKOUT, {**SETUP_PYTHON, "with": {"python-version": "${{ matrix.python }}"}}, checked(LOCKED)], "CPython 3.11"),
}


@pytest.mark.parametrize(("steps", "expected"), DEVIATIONS.values(), ids=list(DEVIATIONS))
def test_the_checker_flags_each_deviation_for_the_right_reason(root, steps, expected):
    # Arrange
    doc = job_doc(*steps)

    # Act
    problems = violations(doc, root)

    # Assert
    assert any(expected in problem for problem in problems), problems


TOKEN_FORMS = {
    "dotted": TOKEN,
    "indexed-single-quote": "${{ secrets['HF_TOKEN'] }}",
    "indexed-double-quote": '${{ secrets["HF_TOKEN"] }}',
    "lowercase-name": "${{ secrets.hf_token }}",
    "mixed-case-context": "${{ Secrets.Hf_Token }}",
    "all-secrets": "${{ toJSON(secrets) }}",
}


@pytest.mark.parametrize("expr", TOKEN_FORMS.values(), ids=list(TOKEN_FORMS))
@pytest.mark.parametrize("token", ["workflow", "job", "step", "with"])
def test_the_scope_check_sees_the_token_wherever_and_however_it_is_set(root, token, expr):
    # Arrange — 설치 스텝이 아닌 곳에 토큰이 있어도(openaq-bulk-collect 처럼) 잡 전체가 대상이다
    doc = job_doc(*PREP, run(UNPINNED_HUB), token=token, expr=expr)

    # Act
    problems = violations(doc, root)

    # Assert
    assert any("--require-hashes 없음" in problem for problem in problems), problems


@pytest.mark.parametrize("secrets", ["inherit", {WRITE_SECRET: TOKEN}], ids=["inherit", "explicit"])
def test_a_reusable_workflow_job_that_receives_the_token_is_a_violation(root, secrets):
    # Arrange — 안쪽 워크플로의 설치는 이 시험이 못 본다
    doc = {"jobs": {"call": {"uses": "./.github/workflows/publish.yml", "secrets": secrets}}}

    # Act
    problems = violations(doc, root)

    # Assert
    assert any("재사용 워크플로" in problem for problem in problems), problems


def test_a_reusable_workflow_job_without_the_token_is_not_checked(root):
    # Arrange
    doc = {"jobs": {"call": {"uses": "./.github/workflows/build.yml", "with": {"flag": "1"}}}}

    # Act / Assert
    assert token_jobs(doc) == {}
    assert violations(doc, root) == []


@pytest.mark.parametrize(
    "expr",
    ["${{ secrets.HF_MODELS_READ }}", "${{ secrets['HF_MODELS_READ'] }}", "${{ secrets.HF_TOKEN_READONLY }}"],
    ids=["dotted", "indexed", "same-prefix"],
)
def test_a_job_without_the_write_token_is_not_checked(root, expr):
    # Arrange — 다른 시크릿(비공개 wheel 읽기 토큰, 이름이 앞부분만 같은 것)은 쓰기 토큰이 아니다
    doc = {"jobs": {"infer": {
        "env": {"HF_MODELS_READ": expr},
        "steps": [CHECKOUT, run('python3 -m pip install --no-cache-dir --no-deps "$WHEEL"')],
    }}}

    # Act
    scoped = token_jobs(doc)

    # Assert
    assert scoped == {}
    assert violations(doc, root) == []
