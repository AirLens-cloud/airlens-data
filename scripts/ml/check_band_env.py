#!/usr/bin/env python3
"""band 추론 venv 의 설치 결과 검사 (stdlib 전용).

설치는 세 단계다 — CPU torch(해시 고정) → 추론 wheel(`--no-deps`) → 나머지 폐포(해시 고정). wheel 의 기본 의존성
(xgboost 등)은 band 에 필요 없어서 일부러 깔지 않는다. 그래서 `pip check` 는 그 의존성마다 정확히 한 줄,
"requires X, which is not installed." 를 낸다. 워크플로가 `pip check` 의 종료 코드를 삼키므로(`|| true`) 출력이 유일한
신호다 — 그 줄의 집합이 **기대한 줄과 정확히 같을 때만** 통과시킨다.

세 가지를 본다.
  1. pip check — 낸 줄의 집합이 기대한 줄(`--expect-missing` 의존성마다 한 줄)과 정확히 같다. 기대한 줄이 하나라도 없거나
     (그 의존성이 깔렸거나 wheel 의 의존성이 바뀌었다 — 목록을 고친다) 다른 줄이 있으면(버전 충돌 · 다른 패키지의 미설치 ·
     예상 밖 의존성) 실패다. 줄에 wheel 버전이 들어 있어서 다른 버전이 깔려도 걸린다. 출력이 비어 있어도 실패고,
     `No broken requirements found.` 도 실패다(기대한 줄이 하나도 없다는 뜻이다).
  2. 설치 목록(`pip list --format=freeze`) — wheel 이 핀 버전으로 깔려 있고, GPU 스택(nvidia-*, cuda-*, triton)이 없어야 한다.
  3. `torch.version.cuda` — CPU 빌드면 `None` 이다.

CLI:
    check_band_env.py --pip-check FILE --pip-list FILE --torch-cuda VALUE --project NAME --version VER --expect-missing a,b,c
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


def parse_expect_missing(raw: str) -> list[str]:
    """`a,b,c` → 정규화한 이름 목록. 빈 항목은 오타일 가능성이 높아 거부한다."""
    names = [canonical(item.strip()) for item in raw.split(",")]
    if not raw.strip() or any(not name for name in names):
        raise ValueError(f"기대 목록이 비었거나 빈 항목이 있다: {raw!r}")
    return names


def pip_check_lines(output: str) -> list[str]:
    """`pip check` 출력의 비어 있지 않은 줄(앞뒤 공백 제거)."""
    return [line for line in (raw.strip() for raw in output.splitlines()) if line]


def expected_pip_check_lines(project: str, version: str, expected_missing: Iterable[str]) -> set[str]:
    """`pip check` 가 내야 하는 줄 전부 = 일부러 깔지 않은 의존성마다 `<project> <version> requires <dep>, which is not installed.`"""
    return {f"{project} {version} requires {canonical(dep)}, which is not installed." for dep in expected_missing}


def pip_check_mismatches(output: str, project: str, version: str, expected_missing: Iterable[str]) -> list[str]:
    """`pip check` 출력이 기대한 줄의 집합과 다른 점들(사유 문장). 정확히 같으면 빈 목록.

    줄은 전체가 같아야 한다(부분 일치 없음). 같은 줄이 되풀이돼도 한 줄로 본다 — wheel 이 같은 의존성을 조건만 달리해 두 번
    적으면 pip 도 같은 줄을 두 번 쓴다.
    """
    observed = set(pip_check_lines(output))
    expected = expected_pip_check_lines(project, version, expected_missing)
    if not observed:
        return ["pip check 출력이 비어 있다 — pip 가 아무것도 쓰지 않았다"]
    if observed == {NO_PROBLEMS_LINE}:
        return [f"pip check 가 `{NO_PROBLEMS_LINE}` 라고 했다 — 설치하지 않기로 한 의존성이 깔렸거나 wheel 의 의존성이 바뀌었다 "
                "(--expect-missing 목록과 설치 고정본을 다시 확인한다)"]
    return [f"pip check 예상 밖 줄: {line}" for line in sorted(observed - expected)] + \
           [f"pip check 에 기대한 줄이 없다: {line}" for line in sorted(expected - observed)]


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


def problems(pip_check: str, pip_list: str, torch_cuda: str, project: str, version: str, expected_missing: Iterable[str]) -> list[str]:
    """실패 사유 목록(비면 통과)."""
    found = pip_check_mismatches(pip_check, project, version, expected_missing)
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
    parser.add_argument("--expect-missing", required=True,
                        help="일부러 설치하지 않은 wheel 기본 의존성(쉼표로 구분) — pip check 가 정확히 이 의존성들의 '미설치' 줄만 내야 한다")
    args = parser.parse_args(argv)

    try:
        expected = parse_expect_missing(args.expect_missing)
        pip_check = args.pip_check.read_text(encoding="utf-8")
        pip_list = args.pip_list.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"::error::check_band_env: 입력을 읽을 수 없다 — {type(exc).__name__}: {exc}")
        return 2

    found = problems(pip_check, pip_list, args.torch_cuda, args.project, args.version, expected)
    if found:
        for reason in found:
            print(f"::error::band 설치 검사 실패 — {reason}")
        return 1
    observed = len(set(pip_check_lines(pip_check)))
    print(f"band 설치 검사 통과 — {args.project}=={args.version}, pip check 줄 {observed}개가 기대와 정확히 같다, "
          f"GPU 패키지 0개, torch.version.cuda=None")
    return 0


if __name__ == "__main__":
    sys.exit(main())
