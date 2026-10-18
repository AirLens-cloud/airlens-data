"""E2 VM 의 `dispatch-forecast-collect.sh` 를 가짜 `curl` 로 돌려 호출 모양·실패 처리·토큰 비노출을 고정한다 (AAA, 네트워크 0).

**왜 이 파일이 있나.** 이 스크립트는 VM 의 systemd 타이머가 6시간마다 불러 `forecast-collect.yml` 을 `workflow_dispatch` API 로 시작시킨다.
VM 에서만 돌고 PR 에서는 아무도 안 돌리는 코드라서, 어긋나도 다음 슬롯에 로그 한 줄이 실패로 찍힐 뿐 아무도 모른다. 실패해도 워크플로의
`schedule` 백업이 몇 시간 늦게 대신 수집하므로 "정시 수집이 조용히 사라지는" 것이 가장 나쁜 실패다. 그리고 이 스크립트는 레포 쓰기 권한이 있는
토큰을 다룬다 — 토큰이 명령줄·환경·로그 어디에든 새면 같은 VM 의 다른 사용자(`ps`·journal)가 읽을 수 있다.

실제 `bash`(이 머신에 있는 서로 다른 bash 전부 — macOS 의 3.2 포함)로 스크립트를 돌리고, PATH 맨 앞에 가짜 `curl` 을 둔다. 가짜 `curl` 은
받은 인자·표준입력·"토큰이 환경에 있었나" 를 기록하고, 각본대로 HTTP 상태와 본문을 돌려준다(실제 curl 처럼 본문이 비면 `-o` 파일을 건드리지 않는다).

규칙
  1. 토큰은 `curl -q -K -` 의 표준입력 설정으로만 간다 — 인자·환경·표준출력·표준오류 어디에도 없다. `-q` 는 첫 인자라서 `~/.curlrc` 를 읽지 않는다.
     앞뒤 공백·CRLF 는 잘라내고, `[A-Za-z0-9_]` 밖의 글자가 있으면 curl 을 부르기 전에 실패한다(값은 찍지 않는다).
  2. 204·200 은 성공이다. 200 의 본문에 정수 `workflow_run_id` 가 있으면 그 숫자만 로그에 `run <id>` 로 남기고(없으면 ID 없이 성공),
     본문 원문은 싣지 않는다. 그 밖의 응답(거절)은 상태 코드·문구와 상관없이 워크플로의 `state` 를 읽는다(GET) — 꺼진 워크플로에 dispatch 했을 때의
     응답은 문서에 없어서 거기에 기대지 않는다. 정확히 `disabled_inactivity`(60일 무활동으로 꺼짐)일 때만 켜고(PUT enable) 딱 한 번 다시 부른다
     (이 길에서도 같다). 사람이 끈 것(`disabled_manually`)·그 밖의 상태·읽을 수 없는 상태·조회 실패는 켜지 않고 실패한다. 실패 줄은 처음 거절의
     HTTP 코드와 본문의 첫 `message`, 읽은 상태(또는 조회 코드)를 한 줄로 남긴다 — 인쇄 가능한 ASCII 200자 이하의 `message` 이고 토큰이 가려져 있다.
  3. 임시 본문 파일은 어떤 경로로 끝나도 남지 않는다. 이전 요청의 본문이 다음 오류 메시지에 섞이지 않는다.
  4. `set -x` 를 쓰지 않는다. 토큰 검사 문자 범위가 로케일에 흔들리지 않는다.

한계: 실제 GitHub API 의 응답은 못 본다(상태 코드·본문 문구는 각본이다). curl 의 `-K -` 설정 해석과 `-w`·`-o` 동작은 실제 curl 로 따로 확인했다.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "e2-sidecar/bin/dispatch-forecast-collect.sh"
SCRIPT_TEXT = SCRIPT.read_text(encoding="utf-8")
WORKFLOW_DIR = ROOT / ".github/workflows"

CREDENTIAL = "gh_dispatch_pat"
TOKEN = "placeholder_alpha_bravo_charlie_delta_echo_foxtrot"  # 시험용 가짜 값 — 스크립트의 문자 검사를 통과하는 모양일 뿐이다
API = "https://api.github.com/repos/AirLens-cloud/airlens-data/actions/workflows/forecast-collect.yml"
DISPATCH_URL = f"{API}/dispatches"
STATE_URL = API  # 워크플로 객체 조회(GET)
ENABLE_URL = f"{API}/enable"
LOG = "dispatch-forecast-collect: "
BAD_TOKEN = LOG + "credential gh_dispatch_pat is empty or has characters outside [A-Za-z0-9_]\n"
DISABLED = json.dumps({"message": "Cannot trigger a workflow_dispatch on a disabled workflow"})
OK = f"{LOG}dispatched forecast-collect.yml on main"  # 성공 줄의 앞부분 — 뒤에 ` (HTTP …)` 가 붙는다
RUN_ID = "12345678901"  # 지금 GitHub run ID 가 갖는 자릿수 정도의 값
RUN_BODY = json.dumps({
    "workflow_run_id": int(RUN_ID),
    "run_url": f"https://api.github.com/repos/AirLens-cloud/airlens-data/actions/runs/{RUN_ID}",
    "html_url": f"https://github.com/AirLens-cloud/airlens-data/actions/runs/{RUN_ID}",
})


def _discover_bashes() -> list:
    """이 머신의 서로 다른 bash 실행 파일(실제 경로 기준)마다 하나씩 — 버전은 시험 이름에 붙인다."""
    found: dict[str, str] = {}
    for candidate in ("/bin/bash", shutil.which("bash")):
        if not candidate:
            continue
        real = os.path.realpath(candidate)
        if real in found:
            continue
        probe = subprocess.run(
            [real, "-c", 'printf "%s.%s" "${BASH_VERSINFO[0]}" "${BASH_VERSINFO[1]}"'],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if probe.returncode == 0 and probe.stdout:
            found[real] = f"bash-{probe.stdout}"
    return [pytest.param(path, id=label) for path, label in found.items()]


BASHES = _discover_bashes()


def _utf8_locale() -> str | None:
    """이 머신에 있는 UTF-8 로케일 이름(없으면 None)."""
    done = subprocess.run(["locale", "-a"], capture_output=True, text=True, timeout=30, check=False) if shutil.which("locale") else None
    names = done.stdout.split() if done and done.returncode == 0 else []
    return next((n for n in names if re.search(r"utf-?8", n, re.I)), None)


# 호출을 기록하고 각본대로 응답하는 curl 대역. 토큰 값은 "환경에 있었나" 를 판정하는 데만 쓰고 어디에도 찍지 않는다.
FAKE_CURL = r'''#!{python}
import json
import os
import sys

with open(os.environ["FAKE_CURL_STATE"], encoding="utf-8") as fh:
    state = json.load(fh)

argv = sys.argv[1:]
stdin_text = "" if sys.stdin.isatty() else sys.stdin.buffer.read().decode("utf-8", "replace")
token = state["token"]
visible = [*os.environ.keys(), *os.environ.values()]
record = {
    "argv": argv,
    "stdin": stdin_text,
    "token_in_env": any(token in item for item in visible),
    "body_file_existed": "-o" in argv and os.path.exists(argv[argv.index("-o") + 1]),
}

index = 0
if os.path.exists(state["calls"]):
    with open(state["calls"], encoding="utf-8") as fh:
        index = sum(1 for _ in fh)
with open(state["calls"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps(record) + "\n")

if index >= len(state["responses"]):
    sys.stderr.write("fake curl: no scripted response for call %d\n" % index)
    sys.exit(99)
reply = state["responses"][index]
if reply.get("exit"):
    sys.stdout.write("000")
    sys.stderr.write(reply["stderr"])
    sys.exit(reply["exit"])
body = bytes.fromhex(reply["body_hex"])
if body:
    with open(argv[argv.index("-o") + 1], "wb") as out:
        out.write(body)
sys.stdout.write(str(reply["code"]))
'''


def reply(code: int, body: str | bytes = "") -> dict:
    data = body.encode("utf-8") if isinstance(body, str) else body
    return {"code": code, "body_hex": data.hex()}


def curl_fails(exit_code: int = 7) -> dict:
    return {"exit": exit_code, "stderr": "curl: (7) Failed to connect to api.github.com port 443\n"}


def message(text: str) -> str:
    return json.dumps({"message": text})


def state_is(state: str) -> dict:
    """워크플로 조회(GET)의 200 응답 — 문서의 필드 모양에서 `state` 만 바꾼다."""
    return reply(200, json.dumps({
        "id": 1234567,
        "name": "Forecast Collection (Open-Meteo CAMS PM2.5)",
        "path": ".github/workflows/forecast-collect.yml",
        "state": state,
        "url": API,
    }))


INACTIVE = state_is("disabled_inactivity")  # GitHub 이 60일 무활동으로 끈 워크플로


@dataclass(frozen=True)
class Call:
    argv: list[str]
    stdin: str
    token_in_env: bool
    body_file_existed: bool

    @property
    def method(self) -> str:
        return self.argv[self.argv.index("-X") + 1]

    @property
    def url(self) -> str:
        return self.argv[-1]

    @property
    def body_file(self) -> str:
        return self.argv[self.argv.index("-o") + 1]

    @property
    def data(self) -> str | None:
        return self.argv[self.argv.index("--data") + 1] if "--data" in self.argv else None


@dataclass(frozen=True)
class Run:
    code: int
    out: str
    err: str
    calls: list[Call]

    @property
    def requests(self) -> list[tuple[str, str]]:
        return [(c.method, c.url) for c in self.calls]


class Lab:
    """스크립트 한 번을 돌리는 작업 공간: 가짜 curl 이 앞선 PATH, 자격 증명 디렉터리, 비어 있는 TMPDIR."""

    def __init__(self, root: Path, bash: str) -> None:
        self.bash = bash
        self.bin_dir, self.cred_dir, self.tmp_dir, self.work_dir = (root / name for name in ("fake-bin", "credentials", "tmp", "work"))
        for directory in (self.bin_dir, self.cred_dir, self.tmp_dir, self.work_dir):
            directory.mkdir()
        self.state, self.calls_file = root / "state.json", root / "calls.jsonl"
        fake = self.bin_dir / "curl"
        fake.write_text(FAKE_CURL.replace("{python}", sys.executable), encoding="utf-8")
        fake.chmod(0o755)
        self.write_token(TOKEN + "\n")
        self.respond()

    def write_token(self, content: str | bytes) -> None:
        data = content.encode("utf-8") if isinstance(content, str) else content
        (self.cred_dir / CREDENTIAL).write_bytes(data)

    def respond(self, *replies: dict) -> None:
        self.state.write_text(json.dumps({"token": TOKEN, "calls": str(self.calls_file), "responses": list(replies)}), encoding="utf-8")

    def run(self, *, credentials: bool = True, env: dict[str, str] | None = None) -> Run:
        full_env = {
            "PATH": f"{self.bin_dir}:/usr/bin:/bin",
            "HOME": str(self.work_dir),
            "TMPDIR": str(self.tmp_dir),
            "FAKE_CURL_STATE": str(self.state),
        }
        if credentials:
            full_env["CREDENTIALS_DIRECTORY"] = str(self.cred_dir)
        full_env.update(env or {})
        done = subprocess.run(
            [self.bash, str(SCRIPT)], env=full_env, cwd=self.work_dir, stdin=subprocess.DEVNULL,
            capture_output=True, timeout=60, check=False,
        )
        lines = self.calls_file.read_text(encoding="utf-8").splitlines() if self.calls_file.exists() else []
        calls = [Call(**json.loads(line)) for line in lines]
        # 엄격한 UTF-8 디코딩 — 원격이 준 바이트가 깨진 채 새어 나오면 여기서 시험이 실패한다
        return Run(done.returncode, done.stdout.decode("utf-8"), done.stderr.decode("utf-8"), calls)


@pytest.fixture(params=BASHES)
def lab(request, tmp_path) -> Lab:
    return Lab(tmp_path, request.param)


def leaks(text: str, secret: str = TOKEN, window: int = 8) -> bool:
    """secret 의 한 조각(window 글자)이라도 text 에 있으면 True — 일부만 새는 것도 잡는다."""
    return any(secret[i:i + window] in text for i in range(len(secret) - window + 1))


# ── 성공 ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("status", [204, 200])
def test_a_204_or_200_answer_means_the_dispatch_was_accepted(lab, status):
    # Arrange — 본문 없는 204 가 예전 응답이고, 200 도 본문이 비어 있어도 성공이다
    lab.respond(reply(status))

    # Act
    run = lab.run()

    # Assert
    assert run.code == 0
    assert run.out == f"{OK} (HTTP {status})\n"
    assert run.err == ""
    assert run.requests == [("POST", DISPATCH_URL)]


def test_a_200_answer_logs_the_run_id_from_its_body_and_nothing_else_of_it(lab):
    # Arrange — 지금 dispatch API 는 200 과 새 run 의 ID·URL 을 돌려준다
    lab.respond(reply(200, RUN_BODY))

    # Act
    run = lab.run()

    # Assert — URL 은 싣지 않는다(본문 원문은 로그에 없다)
    assert run.code == 0
    assert run.out == f"{OK} (HTTP 200, run {RUN_ID})\n"
    assert run.err == ""
    assert run.requests == [("POST", DISPATCH_URL)]


RUN_ID_LAYOUTS = {
    "compact-first": ('{"workflow_run_id":42,"run_url":"u","html_url":"h"}', "42"),
    "compact-last": ('{"run_url":"u","html_url":"h","workflow_run_id":42}', "42"),
    "spaces-around-the-colon-and-the-number": ('{"run_url":"u", "workflow_run_id" :  42 , "html_url":"h"}', "42"),
    "pretty-printed-middle": ('{\n  "run_url": "u",\n  "workflow_run_id": 42,\n  "html_url": "h"\n}\n', "42"),
    "pretty-printed-last": ('{\n  "run_url": "u",\n  "workflow_run_id": 42\n}\n', "42"),
    "pretty-printed-last-crlf": ('{\r\n  "run_url": "u",\r\n  "workflow_run_id": 42\r\n}\r\n', "42"),
    "tab-indented": ('{\n\t"workflow_run_id":\t42,\n\t"run_url": "u"\n}', "42"),
    "after-a-nested-object": ('{"links":{"self":"s"},"workflow_run_id":42}', "42"),
    "one-digit": ('{"workflow_run_id":7}', "7"),
    "zero": ('{"workflow_run_id":0}', "0"),
    "twenty-digits": ('{"workflow_run_id":12345678901234567890}', "12345678901234567890"),
    "first-of-two-keys": ('{"workflow_run_id":1,"nested":{"workflow_run_id":2}}', "1"),
    "hostile-text-around-it": ('{"message":"\r\n::error::x \x1b[31m","workflow_run_id":42,"html_url":"https://example.invalid/<script>"}', "42"),
    "binary-noise-around-it": (b'\x00\x01\xff{"workflow_run_id":42}\x00\xfe', "42"),
}


@pytest.mark.parametrize("name", list(RUN_ID_LAYOUTS))
def test_the_run_id_is_read_from_any_json_layout_the_api_may_use(lab, name):
    # Arrange
    body, expected = RUN_ID_LAYOUTS[name]
    lab.respond(reply(200, body))

    # Act
    run = lab.run()

    # Assert
    assert run.code == 0
    assert run.out == f"{OK} (HTTP 200, run {expected})\n"


NO_RUN_ID_BODIES = {
    "empty": b"",
    "not-json": b"all good",
    "html": b"<html>200 OK</html>",
    "empty-object": b"{}",
    "message-only": b'{"message":"ok"}',
    "null": b'{"workflow_run_id":null}',
    "string": b'{"workflow_run_id":"123"}',
    "float": b'{"workflow_run_id":12.5}',
    "exponent": b'{"workflow_run_id":1e5}',
    "negative": b'{"workflow_run_id":-5}',
    "array": b'{"workflow_run_id":[1]}',
    "no-value": b'{"workflow_run_id":}',
    "word": b'{"workflow_run_id":abc}',
    "two-numbers": b'{"workflow_run_id": 123 456}',
    "over-twenty-digits": b'{"workflow_run_id":123456789012345678901}',
    "other-key": b'{"other_workflow_run_id":123,"run_id":456}',
    "different-case": b'{"Workflow_Run_Id":123}',
    "key-inside-a-string": b'{"message":"say \\"workflow_run_id\\": 5, ok"}',
    "binary": b'\x00\x01\xff{"workflow_run_id":\x00123,}',
    "hostile-text": b'{"message":"\r\n::error::x \x1b[31m \xe2\x80\xa8","html_url":"https://example.invalid/<script>"}',
}


@pytest.mark.parametrize("name", list(NO_RUN_ID_BODIES))
def test_a_200_answer_without_a_usable_run_id_is_still_a_success_without_one(lab, name):
    # Arrange
    lab.respond(reply(200, NO_RUN_ID_BODIES[name]))

    # Act
    run = lab.run()

    # Assert — ID 를 못 뽑아도 dispatch 는 받아들여진 것이다. 본문의 다른 글자는 로그에 없다
    assert run.code == 0
    assert run.out == f"{OK} (HTTP 200)\n"
    assert run.err == ""


def test_the_run_id_is_read_only_from_a_200_answer(lab):
    # Arrange — 204 는 본문이 없다. 파일에 뭔가 있어도(그럴 수 없지만) ID 로 읽지 않는다
    lab.respond(reply(204, RUN_BODY))

    # Act
    run = lab.run()

    # Assert
    assert run.out == f"{OK} (HTTP 204)\n"


def test_the_dispatch_request_has_exactly_these_curl_options(lab):
    # Arrange
    lab.respond(reply(204))

    # Act
    call = lab.run().calls[0]

    # Assert — 인자 전체가 이 목록이다: 토큰·헤더가 인자에 슬쩍 늘어나지 않고, 시간 상한(--max-time)이 남는다
    assert call.argv == [
        "-q", "-K", "-", "-sS", "--max-time", "30", "-o", call.body_file, "-w", "%{http_code}", "-X", "POST",
        "-H", "Accept: application/vnd.github+json", "-H", "X-GitHub-Api-Version: 2022-11-28", "-A", "airlens-e2-dispatch",
        "--data", '{"ref":"main"}', DISPATCH_URL,
    ]
    assert Path(call.body_file).is_absolute()  # mktemp 가 만든 파일 — macOS 의 mktemp 는 TMPDIR 을 무시하고 사용자 임시 폴더를 쓴다


def test_curl_gets_q_as_its_very_first_argument_in_every_request(lab):
    # Arrange — 켜고 재시도하는 가장 긴 경로의 요청 4개 모두
    lab.respond(reply(422, DISABLED), INACTIVE, reply(204), reply(204))

    # Act
    run = lab.run()

    # Assert — `-q` 는 첫 인자일 때만 `~/.curlrc` 를 막는다. 그 파일이 trace·proxy 를 더하면 표준입력으로 준 토큰 헤더가 어딘가에 남는다
    assert run.requests == FLOW
    assert [call.argv[0] for call in run.calls] == ["-q"] * 4
    assert all(call.argv.count("-q") == 1 for call in run.calls)


@pytest.mark.parametrize("content", [TOKEN, TOKEN + "\n", TOKEN + "\r\n", f"  {TOKEN} \t\n\n", f"\n{TOKEN}", f"\t{TOKEN}\r\n\r\n"])
def test_whitespace_around_the_token_is_trimmed(lab, content):
    # Arrange
    lab.write_token(content)
    lab.respond(reply(204))

    # Act
    run = lab.run()

    # Assert — 줄바꿈이 헤더 값에 섞이면 curl 설정이 깨지거나 헤더가 둘로 갈라진다
    assert run.code == 0
    assert run.calls[0].stdin == f'header = "Authorization: Bearer {TOKEN}"\n'


# ── 거절 → 상태를 읽고, 무활동으로 꺼진 것만 켜서 한 번 재시도 ──────────────────
REFUSAL = "HTTP 422 (message: Cannot trigger a workflow_dispatch on a disabled workflow)"  # `DISABLED` 응답이 로그에 실리는 모양
FLOW = [("POST", DISPATCH_URL), ("GET", STATE_URL), ("PUT", ENABLE_URL), ("POST", DISPATCH_URL)]  # 켜고 재시도하는 가장 긴 경로


def enabling(refusal: str = REFUSAL) -> str:
    """켜기 직전의 로그 줄 — 처음 거절의 코드·message 가 앞에 붙는다."""
    return f"{LOG}dispatch refused: {refusal}; forecast-collect.yml is disabled_inactivity — enabling it and retrying the dispatch once\n"


def left_as_it_is(seen: str, refusal: str = REFUSAL) -> str:
    """켜지 않고 실패하는 stderr 한 줄 — 처음 거절, 그리고 읽은 상태(또는 조회 코드)."""
    return f"{LOG}dispatch failed: {refusal}; {seen} — not enabling it (only an inactivity disable is undone here)\n"


ENABLING = enabling()


@pytest.mark.parametrize(
    ("body", "refusal"),
    [
        (DISABLED, REFUSAL),
        (message("Workflow is DISABLED"), "HTTP 422 (message: Workflow is DISABLED)"),
        (message("Disabled by GitHub after 60 days of inactivity"), "HTTP 422 (message: Disabled by GitHub after 60 days of inactivity)"),
        ("this workflow is disabled", "HTTP 422"),
    ],
    ids=["api-wording", "upper-case", "capitalised", "plain-text"],
)
def test_a_workflow_disabled_for_inactivity_is_enabled_and_the_dispatch_retried_once(lab, body, refusal):
    # Arrange
    lab.respond(reply(422, body), INACTIVE, reply(204), reply(204))

    # Act
    run = lab.run()

    # Assert — dispatch → 상태 조회(GET) → 켜기(PUT) → 다시 dispatch. 켜기 직전 줄에 처음 거절이 남는다
    assert run.code == 0
    assert run.requests == FLOW
    assert run.out == enabling(refusal) + f"{OK} after enabling it (HTTP 204)\n"
    assert run.err == ""


# GitHub 문서는 꺼진 워크플로에 dispatch 했을 때의 상태 코드·문구를 알려 주지 않는다. 그래서 어떤 거절이든 판정은 워크플로의 `state` 가 한다
REFUSALS = [
    (401, message("Bad credentials"), "HTTP 401 (message: Bad credentials)"),
    (403, message("Resource not accessible by personal access token"), "HTTP 403 (message: Resource not accessible by personal access token)"),
    (404, message("Not Found"), "HTTP 404 (message: Not Found)"),
    (404, message("Not Found, workflow is disabled?"), "HTTP 404 (message: Not Found, workflow is disabled?)"),
    (409, "", "HTTP 409"),
    (422, DISABLED, REFUSAL),
    (422, message("Validation Failed"), "HTTP 422 (message: Validation Failed)"),
    (422, message("Workflow does not have 'workflow_dispatch' trigger"), "HTTP 422 (message: Workflow does not have 'workflow_dispatch' trigger)"),
    (422, "", "HTTP 422"),
    (500, "", "HTTP 500"),
    (500, "disabled", "HTTP 500"),
    (502, "<html>502 Bad Gateway</html>", "HTTP 502"),
]
REFUSAL_IDS = [
    "401", "403", "404", "404-with-the-word", "409-empty", "422-api-wording", "422-validation", "422-no-trigger", "422-empty",
    "500-empty", "500-plain-with-the-word", "502-html",
]


@pytest.mark.parametrize(("status", "body", "refusal"), REFUSALS, ids=REFUSAL_IDS)
def test_a_workflow_found_disabled_for_inactivity_is_enabled_whatever_the_refusal_said(lab, status, body, refusal):
    # Arrange
    lab.respond(reply(status, body), INACTIVE, reply(204), reply(204))

    # Act
    run = lab.run()

    # Assert — "disabled" 가 없는 422·404·500 에서도 켜고 재시도한다. 상태 조회는 한 번이다
    assert run.code == 0
    assert run.requests == FLOW
    assert run.out == enabling(refusal) + f"{OK} after enabling it (HTTP 204)\n"
    assert run.err == ""


@pytest.mark.parametrize("state", ["active", "disabled_manually"])
@pytest.mark.parametrize(("status", "body", "refusal"), REFUSALS, ids=REFUSAL_IDS)
def test_any_refusal_gets_one_state_lookup_and_no_enable_unless_the_state_is_inactivity(lab, status, body, refusal, state):
    # Arrange
    lab.respond(reply(status, body), state_is(state), reply(204), reply(204))

    # Act
    run = lab.run()

    # Assert — 거절이면 조회는 늘 한 번 하고, 켜기(PUT)는 `disabled_inactivity` 일 때만이다. 로그에는 처음 거절의 코드·message 가 남는다
    assert run.code == 1
    assert run.requests == FLOW[:2]
    assert run.out == ""
    assert run.err == left_as_it_is(f"workflow state {state}", refusal)


def test_the_state_lookup_is_a_plain_get_without_a_body_or_a_length_header(lab):
    # Arrange
    lab.respond(reply(422, DISABLED), INACTIVE, reply(204), reply(204))

    # Act
    lookup = lab.run().calls[1]

    # Assert — `Content-Length: 0` 은 본문 없는 PUT 에만 붙는다
    assert lookup.argv == [
        "-q", "-K", "-", "-sS", "--max-time", "30", "-o", lookup.body_file, "-w", "%{http_code}", "-X", "GET",
        "-H", "Accept: application/vnd.github+json", "-H", "X-GitHub-Api-Version: 2022-11-28", "-A", "airlens-e2-dispatch",
        STATE_URL,
    ]
    assert lookup.data is None


def test_the_enable_request_has_no_body_and_an_explicit_zero_length(lab):
    # Arrange
    lab.respond(reply(422, DISABLED), INACTIVE, reply(204), reply(204))

    # Act
    enable = lab.run().calls[2]

    # Assert — 본문 없는 PUT 은 길이를 0 으로 밝힌다(안 그러면 411 로 거절하는 엔드포인트가 있다)
    assert enable.argv == [
        "-q", "-K", "-", "-sS", "--max-time", "30", "-o", enable.body_file, "-w", "%{http_code}", "-X", "PUT",
        "-H", "Accept: application/vnd.github+json", "-H", "X-GitHub-Api-Version: 2022-11-28", "-A", "airlens-e2-dispatch",
        "-H", "Content-Length: 0", ENABLE_URL,
    ]
    assert enable.data is None


def test_a_200_answer_to_the_retry_logs_the_run_id_like_a_first_try_would(lab):
    # Arrange
    lab.respond(reply(422, DISABLED), INACTIVE, reply(204), reply(200, RUN_BODY))

    # Act
    run = lab.run()

    # Assert
    assert run.code == 0
    assert run.requests == FLOW
    assert run.out == ENABLING + f"{OK} after enabling it (HTTP 200, run {RUN_ID})\n"
    assert run.err == ""


@pytest.mark.parametrize("body", [b"", b"all good", b'{"workflow_run_id":null}', b'{"workflow_run_id":4.2}'], ids=["empty", "not-json", "null", "float"])
def test_a_200_answer_to_the_retry_without_a_usable_run_id_is_still_a_success(lab, body):
    # Arrange
    lab.respond(reply(422, DISABLED), INACTIVE, reply(204), reply(200, body))

    # Act
    run = lab.run()

    # Assert
    assert run.code == 0
    assert run.out.endswith(f"{OK} after enabling it (HTTP 200)\n")
    assert run.err == ""


def test_a_run_id_in_an_earlier_response_is_not_taken_for_the_retry_s_own(lab):
    # Arrange — 상태 조회 본문에 ID 모양이 있고, 켜기(204)와 재시도(200)는 본문이 비었다.
    # 실제 curl 은 받은 바이트가 없으면 -o 파일을 건드리지 않으므로, 요청마다 비우지 않으면 앞 본문의 숫자가 남는다
    stale = json.dumps({"state": "disabled_inactivity", "workflow_run_id": 999})
    lab.respond(reply(422, DISABLED), reply(200, stale), reply(204), reply(200))

    # Act
    run = lab.run()

    # Assert
    assert run.code == 0
    assert run.out.endswith(f"{OK} after enabling it (HTTP 200)\n")


def test_the_retry_after_enabling_is_only_one_and_a_second_422_fails(lab):
    # Arrange
    lab.respond(reply(422, DISABLED), INACTIVE, reply(204), reply(422, DISABLED), reply(204))

    # Act
    run = lab.run()

    # Assert — 켜졌는데도 거절하면 거기서 멈춘다(상태 조회·켜기를 되풀이하지 않는다)
    assert run.code == 1
    assert run.requests == FLOW
    assert run.err == f"{LOG}dispatch failed: HTTP 422 (message: Cannot trigger a workflow_dispatch on a disabled workflow)\n"


@pytest.mark.parametrize("status", [200, 201, 403, 404, 500])
def test_enabling_must_answer_204_or_the_run_fails_without_a_retry(lab, status):
    # Arrange
    lab.respond(reply(422, DISABLED), INACTIVE, reply(status, message("enable refused")), reply(204))

    # Act
    run = lab.run()

    # Assert
    assert run.code == 1
    assert run.requests == FLOW[:3]
    assert run.err == f"{LOG}enable failed: HTTP {status} (message: enable refused)\n"


@pytest.mark.parametrize(
    "state",
    ["disabled_manually", "active", "deleted", "disabled_fork", "disabled", "disabled_inactivity_", "xdisabled_inactivity"],
)
def test_any_state_but_inactivity_is_left_as_it_is_and_the_run_fails_naming_the_state(lab, state):
    # Arrange — 사람이 끈 것(사고 대응·롤백 절차)을 다음 :50 에 되살리지 않는다. 이미 켜진(active, 경합) 것도 여기서 재시도하지 않는다 — 유닛의 Restart 가 한다.
    # 값은 정확히 `disabled_inactivity` 여야 한다(접두·접미가 더 붙은 값은 아니다)
    lab.respond(reply(422, DISABLED), state_is(state), reply(204), reply(204))

    # Act
    run = lab.run()

    # Assert
    assert run.code == 1
    assert run.requests == FLOW[:2]
    assert run.out == ""
    assert run.err == left_as_it_is(f"workflow state {state}")


def test_the_first_state_in_the_body_is_the_one_used(lab):
    # Arrange — 뒤쪽의 중첩된 값이 아니라 워크플로 자신의 값이 판정이다
    lab.respond(reply(422, DISABLED), reply(200, '{"state":"disabled_manually","note":{"state":"disabled_inactivity"}}'), reply(204), reply(204))

    # Act
    run = lab.run()

    # Assert
    assert run.code == 1
    assert run.requests == FLOW[:2]
    assert run.err == left_as_it_is("workflow state disabled_manually")


STATE_LAYOUTS = {
    "compact": b'{"id":1,"state":"disabled_inactivity","name":"x"}',
    "spaces-around-the-colon": b'{"id": 1, "state" :  "disabled_inactivity" , "name": "x"}',
    "pretty-printed": b'{\n  "id": 1,\n  "state": "disabled_inactivity",\n  "name": "x"\n}\n',
    "last-with-crlf": b'{\r\n  "id": 1,\r\n  "state": "disabled_inactivity"\r\n}\r\n',
    "tab-indented": b'{\n\t"state":\t"disabled_inactivity"\n}',
    "binary-noise-around-it": b'\x00\x01\xff{"state":"disabled_inactivity"}\x00\xfe',
}


@pytest.mark.parametrize("name", list(STATE_LAYOUTS))
def test_the_state_is_read_from_any_json_layout_the_api_may_use(lab, name):
    # Arrange
    lab.respond(reply(422, DISABLED), reply(200, STATE_LAYOUTS[name]), reply(204), reply(204))

    # Act
    run = lab.run()

    # Assert
    assert run.code == 0
    assert run.requests == FLOW
    assert run.out == ENABLING + f"{OK} after enabling it (HTTP 204)\n"


UNREADABLE_STATES = {
    "empty-body": b"",
    "not-json": b"<html>200 OK</html>",
    "no-state": b'{"id": 1, "name": "x"}',
    "null": b'{"state": null}',
    "number": b'{"state": 1}',
    "empty-string": b'{"state": ""}',
    "upper-case": b'{"state": "DISABLED_INACTIVITY"}',
    "capitalised": b'{"state": "Disabled_inactivity"}',
    "dash": b'{"state": "disabled-inactivity"}',
    "digit": b'{"state": "disabled_inactivity2"}',
    "space-inside": b'{"state": "disabled_inactivity "}',
    "too-long": b'{"state": "' + b"a" * 41 + b'"}',
    "array": b'{"state": ["disabled_inactivity"]}',
    "other-keys": b'{"badge_state": "disabled_inactivity", "state_note": "disabled_inactivity"}',
    "hostile-text": b'{"state": "disabled_inactivity\r\n::error::x \x1b[31m"}',
}


@pytest.mark.parametrize("name", list(UNREADABLE_STATES))
def test_a_state_that_cannot_be_read_is_not_enabled_and_no_remote_text_reaches_the_log(lab, name):
    # Arrange
    lab.respond(reply(422, DISABLED), reply(200, UNREADABLE_STATES[name]), reply(204), reply(204))

    # Act
    run = lab.run()

    # Assert — 읽을 수 없는 값은 켜지 않는다. 로그는 고정 문구뿐이라 원격이 준 글자가 없다
    assert run.code == 1
    assert run.requests == FLOW[:2]
    assert run.out == ""
    assert run.err == left_as_it_is("workflow state not readable")


@pytest.mark.parametrize("status", [201, 204, 301, 401, 403, 404, 500, 502])
def test_a_state_lookup_that_does_not_answer_200_fails_without_enabling_anything(lab, status):
    # Arrange — 200 이 아니면 본문에 그럴듯한 state 가 있어도 믿지 않는다
    body = json.dumps({"message": "lookup refused", "state": "disabled_inactivity"})
    lab.respond(reply(422, DISABLED), reply(status, body), reply(204), reply(204))

    # Act
    run = lab.run()

    # Assert — 로그에는 처음 거절과 조회 코드가 남고, 조회 응답의 message 는 싣지 않는다
    assert run.code == 1
    assert run.requests == FLOW[:2]
    assert run.out == ""
    assert run.err == left_as_it_is(f"state lookup HTTP {status}")
    assert "lookup refused" not in run.err


def test_the_refusal_in_the_log_is_the_first_one_and_not_a_message_of_the_lookup(lab):
    # Arrange — 조회(GET)가 $BODY 를 덮어쓰기 전에 처음 거절을 담아 둬야 한다. 조회 본문에도 message 가 있다
    lab.respond(reply(404, message("first refusal")), reply(200, '{"state":"active","message":"lookup message"}'), reply(204), reply(204))

    # Act
    run = lab.run()

    # Assert
    assert run.err == left_as_it_is("workflow state active", "HTTP 404 (message: first refusal)")
    assert "lookup message" not in run.err


def test_a_curl_failure_on_the_state_lookup_stops_the_run(lab):
    # Arrange
    lab.respond(reply(422, DISABLED), curl_fails(), reply(204), reply(204))

    # Act
    run = lab.run()

    # Assert
    assert run.code == 1 and run.out == ""
    assert run.requests == FLOW[:2]
    assert run.err == f"curl: (7) Failed to connect to api.github.com port 443\n{LOG}curl failed (exit 7) for GET {STATE_URL}\n"


# ── 실패 보고 ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("status", "body", "suffix"),
    [
        (401, message("Bad credentials"), " (message: Bad credentials)"),
        (403, message("Resource not accessible by personal access token"), " (message: Resource not accessible by personal access token)"),
        (404, message("Not Found"), " (message: Not Found)"),
        (422, message("Required input 'x' not provided"), " (message: Required input 'x' not provided)"),
        (500, "", ""),
        (502, "<html>502 Bad Gateway</html>", ""),
        (503, '{"error":"unavailable"}', ""),
    ],
)
def test_other_answers_fail_with_the_status_code_the_api_message_and_the_state(lab, status, body, suffix):
    # Arrange — 워크플로는 켜져 있다(active): 거절의 원인이 꺼짐이 아니다
    lab.respond(reply(status, body), state_is("active"), reply(204))

    # Act
    run = lab.run()

    # Assert — 상태 조회 한 번 뒤에 재시도 없이 실패한다(재시도는 유닛의 Restart 가 한다)
    assert run.code == 1
    assert run.out == ""
    assert run.err == left_as_it_is("workflow state active", f"HTTP {status}{suffix}")
    assert run.requests == FLOW[:2]


def test_a_curl_failure_is_reported_with_its_exit_code_and_stops_the_run(lab):
    # Arrange
    lab.respond(curl_fails(7), reply(204))

    # Act
    run = lab.run()

    # Assert
    assert run.code == 1 and run.out == ""
    assert run.err == f"curl: (7) Failed to connect to api.github.com port 443\n{LOG}curl failed (exit 7) for POST {DISPATCH_URL}\n"
    assert len(run.calls) == 1


def test_the_first_message_of_the_error_body_is_the_one_reported(lab):
    # Arrange — 중첩된 errors[].message 보다 위쪽 message 가 이 응답의 이유다
    lab.respond(reply(422, '{"message":"Validation Failed","errors":[{"message":"inner reason"}],"documentation_url":"https://example.invalid/x"}'), state_is("active"))

    # Act
    run = lab.run()

    # Assert
    assert run.err == left_as_it_is("workflow state active", "HTTP 422 (message: Validation Failed)")


HOSTILE_BODIES = [
    (b'{"message":"first\\nsecond"}', "first\\nsecond"),  # JSON 이스케이프는 풀지 않고 글자 그대로 둔다
    (b'{"message":"a\rb"}', "ab"),
    (b'{"message":"a\x1b[31mred\x1b[0m"}', "a[31mred[0m"),
    (b'{"message":"a\x07b\x00c"}', "abc"),
    ('{"message":"a b\u0085céd 한글"}'.encode(), "abcd "),
    (b'{"message":"say \\"hi\\" now","documentation_url":"https://example.invalid"}', 'say \\"hi\\" now'),
    (b'{"message":"' + b"x" * 500 + b'"}', "x" * 200),
    (b'{"message":"a\nb"}', None),  # 값 한가운데의 진짜 줄바꿈 — 한 줄로 못 읽으니 메시지를 싣지 않는다
    (b'{"message":123}', None),
    (b'{"message":""}', None),
    (b"<html>502 Bad Gateway</html>", None),
    (b"", None),
]
HOSTILE_IDS = ["json-escape", "carriage-return", "escape-code", "bell-and-nul", "non-ascii", "escaped-quotes", "long", "raw-newline", "number", "empty-string", "html", "empty"]


@pytest.mark.parametrize(("body", "expected"), HOSTILE_BODIES, ids=HOSTILE_IDS)
def test_the_logged_message_is_one_printable_ascii_line_of_at_most_200_characters(lab, body, expected):
    # Arrange — 응답 본문은 원격이 준 바이트다. journal 에 줄을 위조하거나 터미널 제어 문자를 심지 못해야 한다
    lab.respond(reply(500, body), state_is("active"))

    # Act
    run = lab.run()

    # Assert — 스크립트 자신의 고정 문구에 든 줄표(—)만 ASCII 검사에서 뺀다
    suffix = "" if expected is None else f" (message: {expected})"
    assert run.err == left_as_it_is("workflow state active", f"HTTP 500{suffix}")
    assert run.err.replace("—", "-").isascii() and len(run.err.splitlines()) == 1 and all(ch.isprintable() for ch in run.err[:-1])


@pytest.mark.skipif(_utf8_locale() is None, reason="이 머신에 UTF-8 로케일이 없다")
def test_a_utf8_caller_locale_does_not_change_the_verdicts(lab):
    # Arrange — 호출한 쪽 로케일이 UTF-8 이어도 스크립트는 C 로케일로 못박는다(범위 `[A-Za-z]` 가 악센트 글자를 삼키지 않게)
    locale = {"LC_ALL": _utf8_locale()}
    lab.write_token(TOKEN + "é")

    # Act
    rejected = lab.run(env=locale)
    lab.write_token(TOKEN)
    lab.respond(reply(500, '{"message":"a béc"}'.encode()), state_is("active"))
    reported = lab.run(env=locale)

    # Assert
    assert (rejected.code, rejected.err, rejected.calls) == (1, BAD_TOKEN, [])
    assert reported.err == left_as_it_is("workflow state active", "HTTP 500 (message: abc)")


# ── 토큰 비노출 ──────────────────────────────────────────────────────────────
def test_the_token_travels_only_as_a_curl_config_on_stdin(lab):
    # Arrange — 켜고 재시도하는 가장 긴 경로: 상태 조회(GET)를 포함한 요청 4개 모두 본다
    lab.respond(reply(422, DISABLED), INACTIVE, reply(204), reply(204))

    # Act
    run = lab.run()

    # Assert
    assert run.requests == FLOW
    for call in run.calls:
        argv_text = "\n".join(call.argv)
        assert call.argv[call.argv.index("-K") + 1] == "-"
        assert call.stdin == f'header = "Authorization: Bearer {TOKEN}"\n'
        assert not leaks(argv_text) and "Authorization" not in argv_text and "Bearer" not in argv_text
        assert call.token_in_env is False, "토큰이 curl 의 환경에 있다 — 변수를 export 했다"
    assert not leaks(run.out) and not leaks(run.err)


@pytest.mark.parametrize(
    "echoed",
    [f"Bad credentials {TOKEN} end", TOKEN, "x" * 190 + TOKEN + "tail", f"{TOKEN}{TOKEN}"],
    ids=["inside-text", "alone", "token-would-straddle-the-cut", "twice"],
)
def test_a_token_echoed_back_by_the_server_is_masked_in_the_log(lab, echoed):
    # Arrange — 서버가 받은 토큰을 오류 문구에 되돌려 줘도 로그에는 안 남는다
    lab.respond(reply(401, message(echoed)), state_is("active"))

    # Act
    run = lab.run()

    # Assert
    assert run.code == 1
    assert not leaks(run.err) and "***" in run.err
    logged = re.fullmatch(rf"{re.escape(LOG)}dispatch failed: HTTP 401 \(message: (.*)\); workflow state active .*\n", run.err)
    assert logged and len(logged.group(1)) <= 200


def test_the_message_is_cut_after_the_token_is_masked_so_a_cut_cannot_leave_half_a_token(lab):
    # Arrange — 가린 글이 200자를 넘는다: 198자 + 토큰. 자른 뒤에 가리면 토큰 앞 두 글자가 남고, 가린 뒤에 자르면 가림 표시의 앞 두 글자가 남는다
    lab.respond(reply(401, message("x" * 198 + TOKEN)), state_is("active"))

    # Act
    run = lab.run()

    # Assert
    assert run.err == left_as_it_is("workflow state active", f"HTTP 401 (message: {'x' * 198}**)")


def test_a_token_echoed_back_in_the_refusal_is_masked_in_the_enabling_line_too(lab):
    # Arrange — 켜는 길에서는 처음 거절이 stdout 의 켜기 줄에 실린다
    lab.respond(reply(401, message(f"Bad credentials {TOKEN} end")), INACTIVE, reply(204), reply(204))

    # Act
    run = lab.run()

    # Assert
    assert run.code == 0
    assert not leaks(run.out) and not leaks(run.err)
    assert run.out == enabling("HTTP 401 (message: Bad credentials *** end)") + f"{OK} after enabling it (HTTP 204)\n"


def test_a_token_echoed_back_by_the_state_lookup_never_reaches_the_log(lab):
    # Arrange — 조회 응답의 message 는 싣지 않는다
    lab.respond(reply(422, DISABLED), reply(200, json.dumps({"state": "active", "message": f"echo {TOKEN}"})), reply(204))

    # Act
    run = lab.run()

    # Assert
    assert not leaks(run.out) and not leaks(run.err)
    assert run.err == left_as_it_is("workflow state active")


BAD_TOKENS = {
    "empty": "",
    "newline-only": "\n",
    "spaces-only": "   \n",
    "tab-and-crlf-only": "\t\r\n",
    "inner-space": TOKEN + " more",
    "second-line": TOKEN + "\nsecond-line",
    "second-line-after-blank-crlf": TOKEN + "\r\n\r\nsecond",
    "first-line-then-token": "first\n" + TOKEN,
    "dash": TOKEN + "-dash",
    "dot": TOKEN + ".dot",
    "colon": TOKEN + ":colon",
    "equals": TOKEN + "=eq",
    "slash": TOKEN + "/slash",
    "backslash": TOKEN + "\\",
    "double-quote": TOKEN + '"',
    "single-quote": TOKEN + "'",
    "semicolon": TOKEN + ";",
    "pipe": TOKEN + "|",
    "ampersand": TOKEN + "&",
    "asterisk": TOKEN + "*",
    "question-mark": TOKEN + "?",
    "bracket": TOKEN + "[",
    "percent": TOKEN + "%",
    "command-substitution": TOKEN + "$(touch PWNED)",
    "backtick": TOKEN + "`touch PWNED`",
    "parameter-expansion": TOKEN + "${IFS}",
    "control-01": TOKEN + "\x01",
    "delete-7f": TOKEN + "\x7f",
    "accented": TOKEN + "é",
    "hangul": TOKEN + "토큰",
    "fullwidth": "ｔｏｋｅｎ",
}


@pytest.mark.parametrize("name", list(BAD_TOKENS))
def test_a_token_with_anything_but_ascii_word_characters_fails_before_curl_and_is_not_printed(lab, name):
    # Arrange
    lab.write_token(BAD_TOKENS[name])
    lab.respond(reply(204))

    # Act
    run = lab.run()

    # Assert — 명령 치환·인용 문자가 섞인 값도 실행되거나 출력되지 않는다
    assert run.code == 1
    assert run.calls == []
    assert run.out == "" and run.err == BAD_TOKEN
    assert not (lab.work_dir / "PWNED").exists()


def test_a_binary_token_file_is_rejected_without_a_request(lab):
    # Arrange
    lab.write_token(bytes(range(1, 256)))

    # Act
    run = lab.run()

    # Assert
    assert (run.code, run.calls, run.out, run.err) == (1, [], "", BAD_TOKEN)


# ── 자격 증명 확인 ───────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "env",
    [None, {"CREDENTIALS_DIRECTORY": ""}],
    ids=["unset", "empty"],
)
def test_without_a_credentials_directory_it_stops_before_any_request(lab, env):
    # Arrange
    lab.respond(reply(204))

    # Act
    run = lab.run(credentials=False, env=env)

    # Assert
    assert (run.code, run.calls, run.out) == (1, [], "")
    assert run.err == f"{LOG}CREDENTIALS_DIRECTORY is not set (run through the systemd unit)\n"


def test_a_missing_credential_file_stops_before_any_request(lab):
    # Arrange
    (lab.cred_dir / CREDENTIAL).unlink()
    lab.respond(reply(204))

    # Act
    run = lab.run()

    # Assert
    assert (run.code, run.calls, run.out) == (1, [], "")
    assert run.err == f"{LOG}credential gh_dispatch_pat is not readable\n"


@pytest.mark.skipif(os.geteuid() == 0, reason="root 는 권한 없는 파일도 읽는다")
def test_an_unreadable_credential_file_stops_before_any_request(lab):
    # Arrange
    (lab.cred_dir / CREDENTIAL).chmod(0o000)
    lab.respond(reply(204))

    # Act
    run = lab.run()

    # Assert
    assert (run.code, run.calls, run.out) == (1, [], "")
    assert run.err == f"{LOG}credential gh_dispatch_pat is not readable\n"


# ── 임시 파일 ────────────────────────────────────────────────────────────────
SCENARIOS = {
    "accepted": [reply(204)],
    "accepted-with-a-run-id": [reply(200, RUN_BODY)],
    "enabled-and-retried": [reply(422, DISABLED), INACTIVE, reply(204), reply(204)],
    "enabled-and-retried-with-a-run-id": [reply(422, DISABLED), INACTIVE, reply(204), reply(200, RUN_BODY)],
    "enabled-after-a-refusal-without-the-word": [reply(404, message("Not Found")), INACTIVE, reply(204), reply(204)],
    "enable-refused": [reply(422, DISABLED), INACTIVE, reply(500, message("boom"))],
    "retry-refused": [reply(422, DISABLED), INACTIVE, reply(204), reply(500, message("boom"))],
    "left-off-by-hand": [reply(422, DISABLED), state_is("disabled_manually")],
    "state-unreadable": [reply(422, DISABLED), reply(200, "{}")],
    "state-lookup-refused": [reply(422, DISABLED), reply(403, message("boom"))],
    "state-lookup-curl-failure": [reply(422, DISABLED), curl_fails(28)],
    "http-error": [reply(500, message("boom")), state_is("active")],
    "curl-failure": [curl_fails(28)],
    "other-422": [reply(422, message("Validation Failed")), state_is("active")],
}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_the_response_body_temp_file_does_not_outlive_the_run(lab, scenario):
    # Arrange
    lab.respond(*SCENARIOS[scenario])

    # Act
    run = lab.run()

    # Assert — 첫 요청 때는 파일이 있었고(vacuous pass 방지), 모든 요청이 같은 파일을 썼으며, 끝난 뒤에는 없다
    assert run.calls[0].body_file_existed is True
    assert len({c.body_file for c in run.calls}) == 1
    assert not any(Path(c.body_file).exists() for c in run.calls)


def test_an_earlier_response_body_does_not_leak_into_the_error_message_of_the_second_dispatch(lab):
    # Arrange — 조회 본문에 message 가 있고, 켜기(204)와 두 번째 dispatch(500)의 응답은 본문이 없다.
    # 실제 curl 은 받은 바이트가 없으면 -o 파일을 건드리지 않아, 요청마다 비우지 않으면 앞 요청의 본문이 그대로 남는다
    lab.respond(reply(422, DISABLED), reply(200, '{"state":"disabled_inactivity","message":"stale lookup message"}'), reply(204), reply(500, ""))

    # Act
    run = lab.run()

    # Assert
    assert run.err == f"{LOG}dispatch failed: HTTP 500\n"


def test_the_state_body_does_not_leak_into_the_error_message_of_the_enable_request(lab):
    # Arrange — 조회 본문에 message 가 있고, 켜기(PUT)의 응답은 본문 없이 실패한다
    lab.respond(reply(422, DISABLED), reply(200, '{"state":"disabled_inactivity","message":"stale lookup message"}'), reply(500, ""))

    # Act
    run = lab.run()

    # Assert
    assert run.err == f"{LOG}enable failed: HTTP 500\n"


# ── 스크립트 자체 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bash", BASHES)
def test_the_script_parses_under_each_bash(bash):
    # Act
    done = subprocess.run([bash, "-n", str(SCRIPT)], capture_output=True, text=True, timeout=60, check=False)

    # Assert
    assert done.returncode == 0, done.stderr


def _code_lines() -> list[str]:
    return [line for line in SCRIPT_TEXT.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def test_the_script_is_executable_and_starts_with_a_bash_shebang_and_strict_mode():
    # Arrange
    code = _code_lines()

    # Assert — deploy.sh 는 0755 로 설치하지만, 소스 쪽 모드도 맞춰 둔다(VM 밖에서 실행해 볼 때)
    assert os.access(SCRIPT, os.X_OK)
    assert SCRIPT_TEXT.splitlines()[0] == "#!/usr/bin/env bash"
    assert code[0] == "set -euo pipefail"


def test_the_script_never_traces_its_commands():
    # Arrange
    code = "\n".join(_code_lines())

    # Assert — `set -x` 는 토큰이 든 curl 설정 줄까지 그대로 찍는다
    assert not re.search(r"\bset\s+-[A-Za-z]*x|xtrace|\bbash\s+-[A-Za-z]*x|\bsh\s+-[A-Za-z]*x", code)
    assert " -x" not in SCRIPT_TEXT.splitlines()[0]


def test_the_script_hands_the_token_to_exactly_one_curl_and_only_through_stdin_config():
    # Arrange
    code = _code_lines()

    # Act
    authorization = [line for line in code if "Authorization" in line]
    curls = [line for line in code if re.search(r"(?:^|\||\$\()\s*curl\b", line)]  # 호출만 — 메시지 안의 낱말 curl 은 뺀다

    # Assert
    assert len(authorization) == 1 and authorization[0].lstrip().startswith("HTTP_CODE=\"$(printf 'header = \"Authorization: Bearer %s\"")
    assert len(curls) == 1 and "curl -q -K -" in curls[0]  # `-q` 는 첫 인자여야 한다(붙여 읽으면 위치까지 고정된다)


def test_the_script_exports_nothing_but_the_locale():
    # Arrange
    code = "\n".join(_code_lines())

    # Act
    exported = re.findall(r"^\s*(?:export\b.*|declare\s+-[a-zA-Z]*x.*|set\s+-a\b.*|set\s+-[a-zA-Z]*a[a-zA-Z]*\b.*)$", code, re.M)

    # Assert — 환경에 올라간 변수는 curl 과 그 자식에게 보인다
    assert exported == ["export LC_ALL=C"]


def test_the_response_body_is_read_by_three_greps_and_nothing_else():
    # Arrange
    uses = [line.strip() for line in _code_lines() if "$BODY" in line]

    # Act
    bookkeeping = [use for use in uses if use.startswith(("trap ", ": > ")) or '-o "$BODY"' in use]
    readers = [use for use in uses if use not in bookkeeping]

    # Assert — run ID·워크플로 state·오류 메시지 셋뿐이다. 거절의 문구를 읽는 `disabled` 확인은 없다(그 문구는 문서에 없어 판정에 안 쓴다).
    # `cat "$BODY"` 같은 읽기가 하나라도 더해지면 본문 원문이 로그로 갈 길이 열린다
    assert len(bookkeeping) == 3, bookkeeping
    assert len(readers) == 3 and all(re.search(r"\bgrep -[A-Za-z]*a[A-Za-z]*\b", use) for use in readers), readers
    keys = [m.group(1) for use in readers if (m := re.search(r'"(workflow_run_id|state|message)"', use))]
    assert sorted(keys) == ["message", "state", "workflow_run_id"], readers


def test_the_script_needs_neither_jq_nor_python_on_the_vm():
    # Arrange
    code = "\n".join(_code_lines())

    # Assert — 응답 본문은 정규식으로만 읽는다(VM 에 jq 가 있다고 가정하지 않는다)
    assert not re.search(r"\b(jq|python3?|perl|node)\b", code)


def test_the_script_targets_the_workflow_file_and_repository_of_this_repo():
    # Arrange
    constants = dict(re.findall(r'^(REPO|WORKFLOW|REF)="([^"$]+)"$', SCRIPT_TEXT, re.M))
    assert set(constants) == {"REPO", "WORKFLOW", "REF"}, "스크립트의 상수를 못 찾았다 — 추출이 깨졌다(vacuous pass 방지)"
    workflow = yaml.safe_load((WORKFLOW_DIR / constants["WORKFLOW"]).read_text(encoding="utf-8"))

    # Act
    triggers = workflow.get("on", workflow.get(True))

    # Assert — 이 시험의 URL 은 스크립트와 독립으로 적은 것이다. 파일 이름·저장소·브랜치가 어긋나면 API 가 404/422 로 거절한다
    assert DISPATCH_URL == f"https://api.github.com/repos/{constants['REPO']}/actions/workflows/{constants['WORKFLOW']}/dispatches"
    assert "workflow_dispatch" in triggers, "대상 워크플로가 workflow_dispatch 를 받지 않는다 — API 가 422 로 거절한다"
    assert f"github.repository == '{constants['REPO']}'" in workflow["jobs"]["guard"]["if"]
    assert constants["REF"] == "main"
