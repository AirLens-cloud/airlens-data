#!/usr/bin/env python3
"""band 추론 venv 의 설치 결과 검사 (stdlib 전용).

설치는 세 단계다 — CPU torch(해시 고정) → 추론 wheel(`--no-deps`) → 나머지 폐포(해시 고정). wheel 의 기본 의존성
(xgboost 등)은 band 에 필요 없어서 일부러 깔지 않는다. 그래서 `pip check` 는 "requires X, which is not installed."
줄을 낸다. 이 줄을 통째로 무시하면(`|| true`) 진짜 문제도 같이 묻히므로 **허용 목록에 있는 정확한 줄만** 통과시킨다.
버전 충돌 · 다른 패키지의 미설치 · 예상 밖 의존성처럼 그 밖의 줄이 하나라도 있으면 실패다.

세 가지를 본다.
  1. pip check — 허용한 "설치 안 함" 줄 외의 줄이 없어야 한다. 줄에 wheel 버전이 들어 있어서 다른 버전이 깔려도 걸린다.
     출력이 비어 있어도 실패다 — 정상이어도 pip 는 `No broken requirements found.` 를 쓰므로 빈 출력은 pip 가 아무것도 쓰지 않은 것이다.
  2. 설치 목록(`pip list --format=freeze`) — wheel 이 핀 버전으로 깔려 있고, GPU 스택(nvidia-*, cuda-*, triton)이 없어야 한다.
  3. `torch.version.cuda` — CPU 빌드면 `None` 이다.

CLI:
    check_band_env.py --pip-check FILE --pip-list FILE --torch-cuda VALUE --project NAME --version VER --allow-missing a,b,c
"""
from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from pathlib import Path

NO_PROBLEMS_LINE = "No broken requirements found."
GPU_PREFIXES = ("nvidia-", "cuda-")  # nvidia-nccl-cu12 · cuda-bindings · cuda-toolkit …
GPU_NAMES = frozenset({"triton"})


def canonical(name: str) -> str:
    """PEP 503 정규화 — `pip check`/`pip list` 가 쓰는 이름 규칙과 맞춘다."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_allow_missing(raw: str) -> list[str]:
    """`a,b,c` → 정규화한 이름 목록. 빈 항목은 오타일 가능성이 높아 거부한다."""
    names = [canonical(item.strip()) for item in raw.split(",")]
    if not raw.strip() or any(not name for name in names):
        raise ValueError(f"허용 목록이 비었거나 빈 항목이 있다: {raw!r}")
    return names


def pip_check_lines(output: str) -> list[str]:
    """`pip check` 출력의 비어 있지 않은 줄(앞뒤 공백 제거)."""
    return [line for line in (raw.strip() for raw in output.splitlines()) if line]


def _allowed_lines(project: str, version: str, allowed_missing: Iterable[str]) -> set[str]:
    """허용하는 "설치 안 함" 줄 = `<project> <version> requires <dep>, which is not installed.`"""
    return {f"{project} {version} requires {canonical(dep)}, which is not installed." for dep in allowed_missing}


def unexpected_pip_check_lines(output: str, project: str, version: str, allowed_missing: Iterable[str]) -> list[str]:
    """`pip check` 출력에서 허용하지 않은 줄들. 허용 = `<project> <version> requires <dep>, which is not installed.` 정확히 한 줄.

    부분 일치가 아니다 — 줄 전체가 같아야 한다. `No broken requirements found.` 도 정상 줄이다.
    출력이 비었는지는 여기서 보지 않는다(`problems` 가 본다).
    """
    allowed = _allowed_lines(project, version, allowed_missing) | {NO_PROBLEMS_LINE}
    return [line for line in pip_check_lines(output) if line not in allowed]


def observed_allowed_lines(output: str, project: str, version: str, allowed_missing: Iterable[str]) -> int:
    """`pip check` 출력에 실제로 남은 허용 줄의 수(같은 줄은 한 번). `No broken requirements found.` 는 세지 않는다."""
    allowed = _allowed_lines(project, version, allowed_missing)
    return len(allowed.intersection(pip_check_lines(output)))


def _freeze_entries(freeze: str) -> list[tuple[str, str]]:
    """`pip list --format=freeze` 텍스트 → (정규화한 이름, 버전). `이름==버전` 이 아닌 줄은 건너뛴다."""
    entries: list[tuple[str, str]] = []
    for raw in freeze.splitlines():
        line = raw.strip()
        if "==" not in line:
            continue
        name, _, version = line.partition("==")
        entries.append((canonical(name.strip()), version.strip()))
    return entries


def gpu_packages(freeze: str) -> list[str]:
    """설치 목록에 있는 GPU 스택 패키지 이름들(CPU 빌드 torch 는 이 중 아무것도 끌어오지 않는다)."""
    return sorted(name for name, _ in _freeze_entries(freeze) if name.startswith(GPU_PREFIXES) or name in GPU_NAMES)


def installed_version(freeze: str, name: str) -> str | None:
    return next((version for entry, version in _freeze_entries(freeze) if entry == canonical(name)), None)


def problems(pip_check: str, pip_list: str, torch_cuda: str, project: str, version: str, allowed_missing: Iterable[str]) -> list[str]:
    """실패 사유 목록(비면 통과)."""
    found: list[str] = []
    if not pip_check_lines(pip_check):
        found.append("pip check 출력이 비어 있다 — pip 가 아무것도 쓰지 않았다")
    for line in unexpected_pip_check_lines(pip_check, project, version, allowed_missing):
        found.append(f"pip check 예상 밖 줄: {line}")
    installed = installed_version(pip_list, project)
    if installed != version:
        found.append(f"{project} 가 {version} 으로 설치돼 있지 않다 (설치 목록: {installed or '없음'})")
    for name in gpu_packages(pip_list):
        found.append(f"GPU 스택 패키지가 설치돼 있다: {name}")
    if torch_cuda.strip() != "None":
        found.append(f"torch.version.cuda 가 None 이 아니다: {torch_cuda.strip()!r:.40} (CPU 빌드가 아니다)")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pip-check", required=True, type=Path, help="`pip check` 출력 파일")
    parser.add_argument("--pip-list", required=True, type=Path, help="`pip list --format=freeze` 출력 파일")
    parser.add_argument("--torch-cuda", required=True, help="`torch.version.cuda` 를 출력한 값(CPU 빌드면 None)")
    parser.add_argument("--project", required=True, help="wheel 의 배포 이름")
    parser.add_argument("--version", required=True, help="wheel 의 핀 버전")
    parser.add_argument("--allow-missing", required=True, help="일부러 설치하지 않은 wheel 기본 의존성(쉼표로 구분)")
    args = parser.parse_args(argv)

    try:
        allowed = parse_allow_missing(args.allow_missing)
        pip_check = args.pip_check.read_text(encoding="utf-8")
        pip_list = args.pip_list.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"::error::check_band_env: 입력을 읽을 수 없다 — {type(exc).__name__}: {exc}")
        return 2

    found = problems(pip_check, pip_list, args.torch_cuda, args.project, args.version, allowed)
    if found:
        for reason in found:
            print(f"::error::band 설치 검사 실패 — {reason}")
        return 1
    observed = observed_allowed_lines(pip_check, args.project, args.version, allowed)
    print(f"band 설치 검사 통과 — {args.project}=={args.version}, pip check 에 남은 허용 줄 {observed}개, "
          f"GPU 패키지 0개, torch.version.cuda=None")
    return 0


if __name__ == "__main__":
    sys.exit(main())
