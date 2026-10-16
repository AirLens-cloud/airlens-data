"""run_entry.py 시험 (AAA).

핵심은 두 가지다.
1. 종료 코드가 console script 래퍼와 같다 (None → 0, 정수, SystemExit, 문자열 → 1, 미발견 127).
2. 예외·경고가 **소스 줄을 찍지 않는다** — 공개 로그에 비공개 코드 조각이 남지 않게 하는 것이 이 실행기의 존재 이유다.
   2번은 자식 프로세스로 확인한다(`sys.tracebacklimit` 은 인터프리터 전역이라 같은 프로세스에서 시험하면 pytest 자신을 건드린다).
   대조군(실행기 없이 직접 호출)이 소스 줄을 찍는 것도 같이 확인해, 이 시험이 소스 줄 노출을 실제로 잡아낼 수 있음을 보인다.
"""
from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import run_entry

RUN_ENTRY = Path(run_entry.__file__).resolve()

TRACE_MARKER = "TRACE_SOURCE_MARKER_zq81"
WARN_MARKER = "WARN_SOURCE_MARKER_zq81"

MODULE_SOURCE = f'''\
import sys
import warnings


def ok():
    return None


def code3():
    return 3


def exit5():
    raise SystemExit(5)


def exit_none():
    sys.exit()


def exit_text():
    raise SystemExit("exit-text-message")


def fail():
    raise ValueError("boom-message")  # {TRACE_MARKER}


def chained():
    try:
        raise KeyError("inner")  # {TRACE_MARKER}
    except KeyError as exc:
        raise RuntimeError("outer-message") from exc  # {TRACE_MARKER}


def warn():
    warnings.warn("careful-message", UserWarning)  # {WARN_MARKER}


def log_exception():
    import logging
    logging.basicConfig(format="%(levelname)s:%(message)s")
    try:
        raise ValueError("log-boom")  # {TRACE_MARKER}
    except ValueError:
        logging.exception("while-logging")


def thread_failure():
    import threading

    def target():
        raise ValueError("thread-boom")  # {TRACE_MARKER}

    worker = threading.Thread(target=target)
    worker.start()
    worker.join()


def show_argv():
    print("argv=" + repr(sys.argv))
'''

# 모듈 최상위 코드에서 예외가 나는 entry — 실행기가 entry 를 불러오는 도중의 실패 경로
BROKEN_MODULE_SOURCE = f'''\
raise ImportError("import-boom")  # {TRACE_MARKER}


def main():
    return None
'''

ENTRY_POINTS = """\
[console_scripts]
fake-ok = fakecli_mod:ok
fake-code3 = fakecli_mod:code3
fake-exit5 = fakecli_mod:exit5
fake-exit-none = fakecli_mod:exit_none
fake-exit-text = fakecli_mod:exit_text
fake-fail = fakecli_mod:fail
fake-chained = fakecli_mod:chained
fake-warn = fakecli_mod:warn
fake-log-exception = fakecli_mod:log_exception
fake-thread-failure = fakecli_mod:thread_failure
fake-argv = fakecli_mod:show_argv
fake-broken-import = fakecli_broken:main
"""


@pytest.fixture
def fake_dist(tmp_path: Path) -> Path:
    """`fakecli_mod`(+ import 중 실패하는 `fakecli_broken`)과 그 console script 를 가진 dist-info 를 둔 디렉터리."""
    (tmp_path / "fakecli_mod.py").write_text(MODULE_SOURCE, encoding="utf-8")
    (tmp_path / "fakecli_broken.py").write_text(BROKEN_MODULE_SOURCE, encoding="utf-8")
    info = tmp_path / "fakecli-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fakecli\nVersion: 1.0\n", encoding="utf-8")
    (info / "entry_points.txt").write_text(ENTRY_POINTS, encoding="utf-8")
    return tmp_path


@pytest.fixture
def isolated_interpreter_state(monkeypatch: pytest.MonkeyPatch):
    """run_entry.main 이 바꾸는 전역(tracebacklimit·formatwarning·argv)을 시험 뒤 되돌린다."""
    monkeypatch.setattr(warnings, "formatwarning", warnings.formatwarning)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    had = hasattr(sys, "tracebacklimit")
    old = getattr(sys, "tracebacklimit", None)
    yield
    if had:
        sys.tracebacklimit = old  # type: ignore[assignment]
    elif hasattr(sys, "tracebacklimit"):
        del sys.tracebacklimit
    sys.modules.pop("fakecli_mod", None)


def _run_in_process(name_and_args: list[str], fake_dist: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.syspath_prepend(str(fake_dist))
    return run_entry.main(name_and_args)


# ── 종료 코드 ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("script", "expected"),
    [("fake-ok", 0), ("fake-code3", 3), ("fake-exit5", 5), ("fake-exit-none", 0), ("fake-exit-text", 1)],
)
def test_exit_code_matches_console_script_wrapper_semantics(script, expected, fake_dist, monkeypatch, isolated_interpreter_state):
    # Arrange / Act
    code = _run_in_process([script], fake_dist, monkeypatch)

    # Assert
    assert code == expected


def test_system_exit_with_text_prints_it_to_stderr(fake_dist, monkeypatch, isolated_interpreter_state, capsys):
    # Act
    code = _run_in_process(["fake-exit-text"], fake_dist, monkeypatch)

    # Assert — `sys.exit("msg")` 는 msg 를 stderr 에 찍고 1 로 끝난다(래퍼와 같다)
    assert code == 1
    assert "exit-text-message" in capsys.readouterr().err


def test_unknown_script_returns_127_with_name_only(fake_dist, monkeypatch, isolated_interpreter_state, capsys):
    # Act
    code = _run_in_process(["no-such-script", "--x"], fake_dist, monkeypatch)

    # Assert
    assert code == run_entry.NOT_FOUND == 127
    assert "no-such-script" in capsys.readouterr().err


def test_no_arguments_prints_usage_and_returns_2(isolated_interpreter_state, capsys):
    # Act
    code = run_entry.main([])

    # Assert
    assert code == 2
    assert "사용" in capsys.readouterr().err


def test_arguments_after_the_script_name_reach_the_entry_point_as_argv(fake_dist, monkeypatch, isolated_interpreter_state, capsys):
    # Act
    code = _run_in_process(["fake-argv", "--source", "observed", "--dqss"], fake_dist, monkeypatch)

    # Assert — argparse 가 `prog` 와 인자를 래퍼를 직접 부른 것과 같게 보려면 argv 가 [name, *rest] 여야 한다
    assert code == 0
    assert "argv=['fake-argv', '--source', 'observed', '--dqss']" in capsys.readouterr().out


def test_main_sets_traceback_limit_zero_and_replaces_warning_format(fake_dist, monkeypatch, isolated_interpreter_state):
    # Act
    _run_in_process(["fake-ok"], fake_dist, monkeypatch)

    # Assert
    assert sys.tracebacklimit == 0
    assert warnings.formatwarning is run_entry._format_warning


def test_format_warning_drops_source_line_and_directory():
    # Act
    text = run_entry._format_warning("msg", UserWarning, "/some/private/dir/mod.py", 42, line="secret_source_line()")

    # Assert
    assert text == "mod.py:42: UserWarning: msg\n"


# ── 자식 프로세스: traceback·경고에 소스 줄이 없다 ───────────────────────────
def _spawn(args: list[str], fake_dist: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(fake_dist), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, "-u", *args], env=env, capture_output=True, text=True, timeout=60, cwd=fake_dist)


def test_uncaught_exception_prints_one_line_without_frames_or_source(fake_dist):
    # Act
    result = _spawn([str(RUN_ENTRY), "fake-fail"], fake_dist)

    # Assert — 종류와 메시지는 남고, 프레임·소스 줄·파일 경로는 없다
    assert result.returncode == 1
    assert "ValueError: boom-message" in result.stderr
    assert TRACE_MARKER not in result.stderr
    assert "File \"" not in result.stderr
    assert "fakecli_mod.py" not in result.stderr


def test_chained_exception_also_prints_no_source_lines(fake_dist):
    # Act
    result = _spawn([str(RUN_ENTRY), "fake-chained"], fake_dist)

    # Assert
    assert result.returncode == 1
    assert "RuntimeError: outer-message" in result.stderr
    assert TRACE_MARKER not in result.stderr
    assert "File \"" not in result.stderr


def test_control_direct_call_without_run_entry_does_print_source_lines(fake_dist):
    # Arrange — 대조군: 이 시험들이 소스 줄 노출을 실제로 잡아낼 수 있다는 증거
    code = "import fakecli_mod; fakecli_mod.fail()"

    # Act
    result = _spawn(["-c", code], fake_dist)

    # Assert
    assert result.returncode == 1
    assert TRACE_MARKER in result.stderr
    assert "Traceback" in result.stderr


def test_logging_exception_prints_the_message_and_error_without_frames_or_source(fake_dist):
    # Act
    result = _spawn([str(RUN_ENTRY), "fake-log-exception"], fake_dist)

    # Assert — `logging.exception` 도 traceback 을 찍는다. 같은 값(`sys.tracebacklimit`)을 따라 프레임·소스 줄이 없어야 한다.
    assert result.returncode == 0
    assert "while-logging" in result.stderr and "ValueError: log-boom" in result.stderr
    assert TRACE_MARKER not in result.stderr
    assert "File \"" not in result.stderr


def test_control_logging_exception_without_run_entry_does_print_source_lines(fake_dist):
    # Arrange
    code = "import fakecli_mod; fakecli_mod.log_exception()"

    # Act
    result = _spawn(["-c", code], fake_dist)

    # Assert
    assert TRACE_MARKER in result.stderr and "Traceback" in result.stderr


def test_exception_in_a_thread_prints_the_error_without_frames_or_source(fake_dist):
    # Act
    result = _spawn([str(RUN_ENTRY), "fake-thread-failure"], fake_dist)

    # Assert — 스레드의 미처리 예외는 `threading.excepthook` 이 찍는다. 이 경로도 소스 줄이 없어야 한다(종료 코드는 그대로 0).
    assert result.returncode == 0
    assert "ValueError: thread-boom" in result.stderr
    assert TRACE_MARKER not in result.stderr
    assert "File \"" not in result.stderr


def test_control_exception_in_a_thread_without_run_entry_does_print_source_lines(fake_dist):
    # Arrange
    code = "import fakecli_mod; fakecli_mod.thread_failure()"

    # Act
    result = _spawn(["-c", code], fake_dist)

    # Assert
    assert TRACE_MARKER in result.stderr and "Traceback" in result.stderr


def test_exception_while_importing_the_entry_module_prints_no_source_lines(fake_dist):
    # Act
    result = _spawn([str(RUN_ENTRY), "fake-broken-import"], fake_dist)

    # Assert — entry 를 불러오는 중(모듈 최상위 코드)의 예외도 소스 줄이 없어야 한다. 한도는 불러오기 전에 걸어야 한다.
    assert result.returncode == 1
    assert "ImportError: import-boom" in result.stderr
    assert TRACE_MARKER not in result.stderr
    assert "File \"" not in result.stderr


def test_control_import_time_failure_without_run_entry_does_print_source_lines(fake_dist):
    # Arrange
    code = "import fakecli_broken"

    # Act
    result = _spawn(["-c", code], fake_dist)

    # Assert
    assert TRACE_MARKER in result.stderr and "Traceback" in result.stderr


def test_warning_is_printed_without_its_source_line(fake_dist):
    # Act
    result = _spawn([str(RUN_ENTRY), "fake-warn"], fake_dist)

    # Assert — `warnings.warn` 은 tracebacklimit 과 무관하게 소스 줄을 찍는다. 그래서 형식을 따로 바꾼다.
    assert result.returncode == 0
    assert "UserWarning: careful-message" in result.stderr
    assert WARN_MARKER not in result.stderr


def test_control_warning_without_run_entry_does_print_its_source_line(fake_dist):
    # Arrange
    code = "import fakecli_mod; fakecli_mod.warn()"

    # Act
    result = _spawn(["-c", code], fake_dist)

    # Assert
    assert WARN_MARKER in result.stderr


def test_isolated_mode_ignores_pythonpath_and_cwd_so_planted_dists_are_not_found(fake_dist):
    # Arrange — 워크플로는 `python -I -u run_entry.py` 로 부른다. -I 는 PYTHONPATH 와 현재 디렉터리를 sys.path 에서 뺀다.
    #           작업 폴더에 놓인 dist-info·모듈이 console script 를 가로채지 못한다는 것을 호출 형태 그대로 확인한다.
    # Act
    result = _spawn(["-I", str(RUN_ENTRY), "fake-ok"], fake_dist)

    # Assert
    assert result.returncode == run_entry.NOT_FOUND
    assert "fake-ok" in result.stderr
