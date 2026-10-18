"""E2 VM 의 dispatch 타이머·서비스 유닛과 `deploy.sh` 가 서로 맞물리는지 고정한다 (AAA, 파일 파싱 — 네트워크 0).

**왜 이 파일이 있나.** 이 유닛들은 VM 에서만 돈다. PR 은 `pytest scripts` 하나라서 유닛 파일을 고치다 어긋나도 배포(`./deploy.sh`) 뒤에야, 그것도 6시간
마다 한 번 찍히는 로그로만 드러난다. 어긋남이 조용히 생기는 자리는 넷이다.
  1. 토큰 파일 이름 — 서비스의 `LoadCredential`, 스크립트가 읽는 `$CREDENTIALS_DIRECTORY/<이름>`, `deploy.sh` 의 존재 확인이 같아야 한다.
  2. 시각 — 타이머는 시간별 수집기(:05/:15/:35)와 겹치지 않고, 워크플로의 6시간 칸(UTC 00/06/12/18)과 같은 간격이어야 한다.
  3. 재시도 — `StartLimitBurst` 는 창(`StartLimitIntervalSec`)이 `RestartSec` 보다 길어야 걸린다. 기본 창(10초)은 90초 간격의 재시작을 못 센다.
  4. 실행 계정 — 토큰이 든 credential 디렉터리는 유닛의 UID 만 읽는다. 시간별 수집기(ubuntu)와 UID 를 나누는 `DynamicUser=yes` 여야 하고, 코어 파일은 남기지 않는다.

한계: systemd 가 이 파일들을 실제로 받아들이는지(`systemd-analyze verify`, `OnCalendar` 의 `UTC` 접미사, `LoadCredential` 동작)는 못 본다 —
VM 에서 배포 직후 `systemctl list-timers` 로 확인한다(`e2-sidecar/README.md` 검증 3). 여기서는 선언된 값과 파일 사이의 약속만 본다.
"""
from __future__ import annotations

import configparser
import re
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SIDECAR = ROOT / "e2-sidecar"
UNITS = SIDECAR / "systemd"
SERVICE = UNITS / "airlens-forecast-dispatch.service"
TIMER = UNITS / "airlens-forecast-dispatch.timer"
SCRIPT = SIDECAR / "bin/dispatch-forecast-collect.sh"
DEPLOY = SIDECAR / "deploy.sh"
WORKFLOW = ROOT / ".github/workflows/forecast-collect.yml"

CREDENTIAL_NAME = "gh_dispatch_pat"
CREDENTIAL_PATH = f"/etc/airlens/{CREDENTIAL_NAME}"
TIMER_UNIT = "airlens-forecast-dispatch.timer"
HOURLY_TIMERS = sorted(p for p in UNITS.glob("airlens-*.timer") if p != TIMER)
HARDENING = {
    "NoNewPrivileges": "yes",
    "PrivateTmp": "yes",  # ProtectSystem=strict 아래에서 스크립트의 mktemp 가 쓸 수 있는 곳은 이 개인 /tmp 뿐이다
    "ProtectSystem": "strict",
    "ProtectHome": "yes",
    "ProtectKernelTunables": "yes",
    "ProtectControlGroups": "yes",
    "RestrictSUIDSGID": "yes",
}


def _unit(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(strict=False, interpolation=None, delimiters=("=",))
    parser.optionxform = str  # systemd 키는 대소문자를 구분한다
    parser.read_string(path.read_text(encoding="utf-8"))
    return parser


def _bytes(size: str) -> int:
    number, suffix = re.fullmatch(r"(\d+)([KMG]?)", size).groups()
    return int(number) * {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[suffix]


def _seconds(span: str) -> int:
    number, unit = re.fullmatch(r"(\d+)\s*(s|sec|min|h)", span).groups()
    return int(number) * {"s": 1, "sec": 1, "min": 60, "h": 3600}[unit]


def _hours(spec: str) -> set[int]:
    """systemd 의 시(hour) 필드 일부만 읽는다: `*` · `a/step` · `a,b`."""
    if spec == "*":
        return set(range(24))
    if "/" in spec:
        start, step = (int(n) for n in spec.split("/"))
        return set(range(start, 24, step))
    return {int(n) for n in spec.split(",")}


def _ssh_block() -> str:
    """deploy.sh 가 VM 에서 돌리라고 큰따옴표 문자열로 넘기는 본문(`"${SSH[@]}" "set -euo pipefail` ~ 닫는 `"`)."""
    text = DEPLOY.read_text(encoding="utf-8")
    start = text.index('"${SSH[@]}" "set -euo pipefail')
    body_start = text.index("\n", start) + 1
    end = text.index('\n"\n', body_start)
    return text[body_start:end]


# ── 서비스 ───────────────────────────────────────────────────────────────────
def test_the_service_is_a_oneshot_run_as_a_dynamic_user_of_its_own():
    # Act
    service = _unit(SERVICE)["Service"]

    # Assert — 토큰은 실행하는 동안 이 유닛의 credential 디렉터리에 있고, 그 디렉터리는 유닛의 UID 만 읽는다. 시간별 수집기와 같은 ubuntu UID 면
    # 그 창에 수집기가 읽을 수 있다. `User=` 가 남아 있으면 그 이름이 동적 계정에 붙어 고정 계정 이름이 되살아난다
    assert (service["Type"], service["DynamicUser"]) == ("oneshot", "yes")
    assert "User" not in service and "Group" not in service


def test_the_service_leaves_no_core_file_of_a_process_that_holds_the_token():
    # Act
    service = _unit(SERVICE)["Service"]

    # Assert — bash·curl 의 메모리에 토큰이 있다. 비정상 종료가 코어 파일로 남지 않게 크기 한도를 0 으로 둔다
    assert service["LimitCORE"] == "0"


def test_the_service_runs_the_script_that_deploy_installs():
    # Arrange
    service = _unit(SERVICE)["Service"]
    deploy = DEPLOY.read_text(encoding="utf-8")

    # Act
    exec_start = service["ExecStart"]

    # Assert — deploy.sh 는 bin/*.sh 를 통째로 /usr/local/bin 에 0755 로 깐다: 목록에 파일을 더하지 않아도 스크립트가 따라간다
    assert exec_start == f"/usr/local/bin/{SCRIPT.name}"
    assert SCRIPT.is_file() and SCRIPT.suffix == ".sh"
    assert re.search(r"install -m 0755 -o root -g root \$STAGE/bin/\*\.sh /usr/local/bin/", deploy)
    assert re.search(r"install -m 0644 -o root -g root \$STAGE/systemd/\* /etc/systemd/system/", deploy)


def test_the_token_file_name_is_the_same_in_the_unit_the_script_and_deploy():
    # Arrange
    service = _unit(SERVICE)["Service"]
    script = SCRIPT.read_text(encoding="utf-8")
    deploy = DEPLOY.read_text(encoding="utf-8")

    # Assert — 이름이 어긋나면 스크립트가 "not readable" 로 매번 실패하고, deploy 의 확인은 엉뚱한 파일을 본다
    assert service["LoadCredential"] == f"{CREDENTIAL_NAME}:{CREDENTIAL_PATH}"
    assert f'"$CREDENTIALS_DIRECTORY/{CREDENTIAL_NAME}"' in script
    assert f"sudo test -r {CREDENTIAL_PATH}" in deploy


def test_the_token_is_not_given_to_the_service_in_any_other_way():
    # Arrange
    text = SERVICE.read_text(encoding="utf-8")
    keys = set(_unit(SERVICE)["Service"])

    # Assert — 환경 변수·EnvironmentFile 로 주면 `systemctl show`·/proc 에 보인다
    assert not {"Environment", "EnvironmentFile", "PassEnvironment", "SetCredential"} & keys
    assert not re.search(r"ghp_|github_pat_|gho_|ghs_", text)


def test_the_service_memory_is_capped_far_below_the_collectors():
    # Arrange
    service = _unit(SERVICE)["Service"]
    collectors = [_bytes(_unit(p)["Service"]["MemoryMax"]) for p in UNITS.glob("airlens-*.service") if p != SERVICE]

    # Assert — 1GB 박스에서 curl 한 번이다. OOM 이 이 유닛의 cgroup 안에서 끝나야 한다
    assert service["MemoryAccounting"] == "yes"
    assert _bytes(service["MemoryHigh"]) < _bytes(service["MemoryMax"]) <= _bytes("50M")
    assert collectors, "다른 서비스 유닛을 못 찾았다 — 글롭이 깨졌다(vacuous pass 방지)"
    assert _bytes(service["MemoryMax"]) < min(collectors)


def test_the_service_retries_once_and_its_start_limit_window_is_longer_than_the_retry_delay():
    # Arrange
    unit = _unit(SERVICE)

    # Act
    delay = _seconds(unit["Service"]["RestartSec"])
    window = _seconds(unit["Unit"]["StartLimitIntervalSec"])

    # Assert — 창이 재시작 간격보다 짧으면 두 번째 실패 뒤에도 한도(burst)에 안 걸려 90초마다 영원히 다시 시도한다
    assert unit["Service"]["Restart"] == "on-failure"
    assert (delay, unit["Unit"]["StartLimitBurst"]) == (90, "2")
    assert window > delay
    assert "StartLimitBurst" not in unit["Service"] and "StartLimitIntervalSec" not in unit["Service"]  # [Unit] 이 문서화된 자리다


def test_the_start_timeout_covers_the_longest_run_of_four_requests():
    # Arrange
    timeout = _seconds(_unit(SERVICE)["Service"]["TimeoutStartSec"])
    max_time = int(re.search(r"--max-time (\d+)", SCRIPT.read_text(encoding="utf-8")).group(1))

    # Assert — 가장 긴 경로는 dispatch → 워크플로 상태 조회 → enable → dispatch 네 번이다. 유닛이 그 전에 죽이면 마지막 재시도가 잘린다
    assert timeout > 4 * max_time


def test_the_service_is_hardened_like_the_other_sidecar_services_and_writes_nowhere():
    # Arrange
    service = _unit(SERVICE)["Service"]

    # Assert — 스크립트가 쓰는 곳은 mktemp 의 임시 파일뿐이라 개인 /tmp 로 충분하다(쓰기 허용 경로가 없다)
    assert {key: service.get(key) for key in HARDENING} == HARDENING
    assert "ReadWritePaths" not in service and "WorkingDirectory" not in service


def test_the_service_waits_for_the_network_and_names_the_repository_in_its_documentation():
    # Arrange
    unit = _unit(SERVICE)["Unit"]

    # Assert
    assert unit["After"] == unit["Wants"] == "network-online.target"
    assert unit["Documentation"] == "https://github.com/AirLens-cloud/airlens-data"


# ── 타이머 ───────────────────────────────────────────────────────────────────
def test_the_timer_fires_at_minute_50_of_hours_00_06_12_18_utc():
    # Arrange
    timer = _unit(TIMER)["Timer"]

    # Act
    match = re.fullmatch(r"\*-\*-\* ([\d*/,]+):(\d\d):00 UTC", timer["OnCalendar"])

    # Assert — `UTC` 접미사가 VM 의 시간대와 무관하게 워크플로의 6시간 칸에 못박는다
    assert timer["OnCalendar"] == "*-*-* 00/6:50:00 UTC"
    assert match and _hours(match.group(1)) == {0, 6, 12, 18} and match.group(2) == "50"


def test_the_timer_cadence_is_the_workflow_slot_the_guard_uses():
    # Arrange
    cron = re.search(r"- cron: '[\d*]+ \*/(\d+) \* \* \*'", WORKFLOW.read_text(encoding="utf-8"))
    assert cron, "워크플로 schedule 을 못 찾았다 — 추출이 깨졌다(vacuous pass 방지)"
    hours = sorted(_hours(re.fullmatch(r"\*-\*-\* ([\d*/,]+):\d\d:00 UTC", _unit(TIMER)["Timer"]["OnCalendar"]).group(1)))

    # Assert — 타이머 간격 = 백업 schedule 간격 = 가드의 칸. 어긋나면 칸마다 한 번씩 정시로 시작하는 약속이 깨진다
    assert [b - a for a, b in pairwise(hours)] == [int(cron.group(1))] * 3


def test_the_timer_minute_never_lands_on_an_hourly_collector_slot():
    # Arrange
    slots = {
        int(m.group(1))
        for p in HOURLY_TIMERS
        if (m := re.fullmatch(r"\*-\*-\* \*:(\d\d):00", _unit(p)["Timer"]["OnCalendar"]))
    }

    # Act
    mine = int(re.fullmatch(r"\*-\*-\* [\d*/,]+:(\d\d):00 UTC", _unit(TIMER)["Timer"]["OnCalendar"]).group(1))

    # Assert — 1GB 박스: 시간별 수집기의 메모리 피크와 겹치지 않는다
    assert slots == {5, 15, 35}, f"시간별 타이머 슬롯을 못 읽었다: {slots}"
    assert mine not in slots


def test_the_timer_runs_a_missed_slot_late_and_adds_no_jitter():
    # Arrange
    timer = _unit(TIMER)

    # Assert — VM 이 정시에 꺼져 있었으면 켜지는 즉시 한 번 돈다(가드가 칸으로 중복을 거른다). 지터가 있으면 "정시" 가 아니다
    assert timer["Timer"]["Persistent"] == "true"
    assert timer["Timer"]["RandomizedDelaySec"] == "0"
    assert timer["Timer"]["AccuracySec"] == "1min"
    assert timer["Install"]["WantedBy"] == "timers.target"


def test_the_timer_starts_the_service_of_the_same_name():
    # Arrange
    timer = _unit(TIMER)

    # Assert — `Unit=` 이 없으면 같은 이름의 .service 를 시작한다
    assert "Unit" not in timer["Timer"]
    assert SERVICE.name == TIMER.name.replace(".timer", ".service")


# ── deploy.sh ────────────────────────────────────────────────────────────────
def test_deploy_enables_the_dispatch_timer_only_when_its_token_file_is_readable():
    # Arrange
    block = _ssh_block()
    gate = re.compile(
        rf"if sudo test -r {re.escape(CREDENTIAL_PATH)}; then\n"
        rf"\s+sudo systemctl enable --now {re.escape(TIMER_UNIT)}\n"
        rf"\s+else\n"
        rf"\s+echo 'NOTE: {re.escape(CREDENTIAL_PATH)} absent — forecast-dispatch timer not enabled\.'\n"
        rf"\s+fi"
    )

    # Assert — 토큰 없이 켜면 6시간마다 실패 로그만 쌓인다(채팅 로그 타이머와 같은 방식). 켜는 줄은 이 if 안에만 있다
    assert gate.search(block), "토큰 확인 안의 enable 구문을 못 찾았다"
    assert block.count(f"enable --now {TIMER_UNIT}") == 1


def test_deploy_installs_the_units_and_reloads_before_it_enables_anything():
    # Arrange
    block = _ssh_block()

    # Act
    positions = [block.index(marker) for marker in ("/etc/systemd/system/", "systemctl daemon-reload", f"enable --now {TIMER_UNIT}")]

    # Assert — 새 유닛 파일이 깔리고 daemon-reload 가 끝난 뒤에야 enable 한다
    assert positions == sorted(positions)


def test_deploy_lists_the_dispatch_timer_with_the_other_timers():
    # Arrange
    block = _ssh_block()

    # Act
    listing = next(line for line in block.splitlines() if "systemctl list-timers" in line)

    # Assert
    assert TIMER_UNIT in listing and "--no-pager" in listing
    assert {"airlens-openaq-shadow.timer", "airlens-sc-shadow.timer", "airlens-chatlog-pull.timer"} <= set(listing.split())


def test_deploy_does_not_enable_the_dispatch_timer_in_the_unconditional_enable_line():
    # Arrange
    block = _ssh_block()

    # Act
    plain = next(line for line in block.splitlines() if "enable --now airlens-openaq-shadow.timer" in line)

    # Assert
    assert TIMER_UNIT not in plain


def test_the_deploy_header_names_the_token_file_among_the_secrets_it_never_touches():
    # Arrange
    header = DEPLOY.read_text(encoding="utf-8").split("set -euo pipefail", 1)[0]

    # Assert
    assert re.search(r"/etc/airlens/\{[^}]*\b" + CREDENTIAL_NAME + r"\b[^}]*\}", header)


def test_the_remote_command_string_has_nothing_the_local_shell_would_expand_but_the_stage_path():
    # Arrange
    block = _ssh_block()

    # Assert — 이 본문은 로컬 bash 의 큰따옴표 문자열이다: 새 `$`·백틱·큰따옴표가 끼면 로컬에서 먼저 풀리거나 문자열이 끊긴다
    assert "`" not in block and '"' not in block
    assert re.findall(r"\$(?!STAGE\b)", block) == []
    assert "$STAGE" in block, "본문을 못 읽었다 — 추출이 깨졌다(vacuous pass 방지)"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash 가 없다")
def test_deploy_script_parses():
    # Act
    done = subprocess.run(["bash", "-n", str(DEPLOY)], capture_output=True, text=True, timeout=60, check=False)

    # Assert
    assert done.returncode == 0, done.stderr
