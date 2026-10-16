#!/usr/bin/env python3
"""설치된 console script 를 traceback 없이 실행한다 (stdlib 전용).

왜 필요한가. 추론 wheel 은 비공개 코드인데 이 레포의 Actions 로그는 공개다. 실행 중 예외가 나면
Python 은 traceback 에 **소스 줄**을 그대로 찍는다 — 실패할 때마다 비공개 코드 조각이 공개 로그에 남는다.
그래서 console script 를 직접 부르지 않고 이 실행기로 부른다:

    python -I -u scripts/ml/run_entry.py airlens-predict-grid --source observed ...

- `sys.tracebacklimit = 0` — 예외는 `ValueError: 메시지` 한 줄만 남는다(프레임·소스 줄 없음). logging 의
  `exception()` 도 같은 값을 따른다. 예외 종류와 메시지는 남으므로 원인 파악은 된다. 줄 단위 재현은 같은
  wheel 버전을 로컬에 설치해서 한다.
- 경고(`warnings`)는 traceback 이 아니라도 **소스 줄을 같이 찍는다**. 형식을 `파일:줄: 종류: 메시지` 로 바꿔
  소스 줄만 뺀다.
- 종료 코드는 console script 래퍼와 같다: 반환값 None → 0, 정수 → 그 값, `SystemExit` → 그 코드.

한계: 이 값은 이 인터프리터 안에서만 유효하다. 자식 프로세스(`multiprocessing` spawn · joblib 프로세스 풀 등)의
traceback 은 덮지 못한다 — 워크플로가 `JOBLIB_MULTIPROCESSING=0` 으로 joblib 프로세스 풀을 막는다.
"""
from __future__ import annotations

import importlib.metadata
import sys
import warnings
from pathlib import Path

NOT_FOUND = 127  # 셸의 "command not found"


def _format_warning(message, category, filename, lineno, line=None) -> str:
    """기본 형식에서 소스 줄을 뺀다 (경로도 파일명만)."""
    return f"{Path(filename).name}:{lineno}: {category.__name__}: {message}\n"


def _exit_code(value: object) -> int:
    """`sys.exit(value)` 의 규약대로 정수 종료 코드로 바꾼다."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    print(value, file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("사용: run_entry.py <console-script 이름> [인자...]", file=sys.stderr)
        return 2
    name, rest = args[0], args[1:]

    sys.tracebacklimit = 0
    warnings.formatwarning = _format_warning

    matches = importlib.metadata.entry_points(group="console_scripts", name=name)
    if not matches:
        print(f"console script 를 찾을 수 없다: {name}", file=sys.stderr)
        return NOT_FOUND
    entry = next(iter(matches)).load()

    sys.argv = [name, *rest]  # argparse 의 prog·인자 해석이 래퍼를 직접 부른 것과 같게
    try:
        return _exit_code(entry())
    except SystemExit as exc:
        return _exit_code(exc.code)


if __name__ == "__main__":
    sys.exit(main())
