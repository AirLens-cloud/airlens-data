"""check_band_env 시험 — `pip check` 출력은 실제 pip 26.2 가 wheel 을 `--no-deps` 로 깐 환경에서 받은 줄 그대로다.

통과 조건은 줄 집합의 정확한 일치다: 기대한 줄이 모두 있고 다른 줄은 없다(빈 출력·`No broken requirements found.` 는 실패)."""
from __future__ import annotations

from pathlib import Path

import pytest

import check_band_env as cbe

PROJECT, VERSION = "airlens-inference", "0.3.0"
SKIPPED = ["joblib", "lightgbm", "python-dotenv", "scikit-learn", "scipy", "xgboost"]  # band 에 필요 없어 깔지 않는 기본 의존성
COVERED = ["numpy", "pandas", "pyyaml", "requests"]  # `[band]` 폐포가 이미 깔아 주는 기본 의존성


def line(dep: str, project: str = PROJECT, version: str = VERSION) -> str:
    return f"{project} {version} requires {dep}, which is not installed."


# `pip check` 실측 출력(wheel 만 --no-deps 로 설치한 빈 venv): 기본 의존성 10개가 전부 "설치 안 됨" 으로 나온다.
REAL_PIP_CHECK_ALL_TEN = "\n".join(line(dep) for dep in sorted(SKIPPED + COVERED, key=lambda d: d.replace("-", ""))) + "\n"
# `[band]` 폐포까지 깐 뒤에 남아야 하는 줄 — 일부러 깔지 않은 6개만.
EXPECTED_PIP_CHECK = "\n".join(line(dep) for dep in SKIPPED) + "\n"

FREEZE_OK = "\n".join(
    [f"{PROJECT}=={VERSION}", "numpy==2.5.1", "pandas==3.0.5", "pyarrow==23.0.1", "timesfm==2.0.2", "torch==2.13.0+cpu", "pip==26.2.1"]
) + "\n"


def problems(pip_check: str = EXPECTED_PIP_CHECK, pip_list: str = FREEZE_OK, cuda: str = "None", expected=SKIPPED) -> list[str]:
    return cbe.problems(pip_check, pip_list, cuda, PROJECT, VERSION, expected)


def mismatches(output: str, project: str = PROJECT, version: str = VERSION, expected=SKIPPED) -> list[str]:
    return cbe.pip_check_mismatches(output, project, version, expected)


# ── pip check ────────────────────────────────────────────────────────────────
def test_exactly_the_expected_lines_pass():
    # Act / Assert
    assert mismatches(EXPECTED_PIP_CHECK) == []
    assert cbe.expected_pip_check_lines(PROJECT, VERSION, SKIPPED) == {line(dep) for dep in SKIPPED}


def test_order_blank_lines_whitespace_and_repeated_lines_do_not_matter():
    # Arrange — 줄의 집합을 본다. wheel 이 같은 의존성을 조건만 달리해 두 번 적으면 pip 은 같은 줄을 두 번 쓴다
    reordered = "\n\n  " + "  \n".join(line(dep) for dep in reversed(SKIPPED)) + "\n\n"

    # Act / Assert
    assert mismatches(reordered) == []
    assert mismatches(EXPECTED_PIP_CHECK + EXPECTED_PIP_CHECK) == []
    assert cbe.pip_check_lines("  a \n\n\tb\n") == ["a", "b"]
    assert cbe.pip_check_lines("") == cbe.pip_check_lines("  \n\n") == []


def test_real_pip_check_output_of_a_bare_wheel_install_flags_exactly_the_dependencies_the_closure_must_cover():
    # Act — 폐포를 아직 깔지 않았다면 기대 목록 밖의 4개(numpy·pandas·pyyaml·requests)가 걸린다
    found = mismatches(REAL_PIP_CHECK_ALL_TEN)

    # Assert — 기대한 6줄은 다 있으니 "없는 줄" 사유는 없다
    assert found == sorted(f"pip check 예상 밖 줄: {line(dep)}" for dep in COVERED)


@pytest.mark.parametrize("installed", SKIPPED)
def test_an_expected_line_that_is_missing_is_a_mismatch_because_the_dependency_got_installed(installed):
    # Arrange — 일부러 깔지 않기로 한 의존성이 깔리면(wheel 의 의존성이 바뀌었거나 고정본이 넓어졌다) 그 줄이 사라진다
    output = "\n".join(line(dep) for dep in SKIPPED if dep != installed) + "\n"

    # Act / Assert — 통과시키지 않는다: 기대 목록이 현실과 어긋났다는 신호다
    assert mismatches(output) == [f"pip check 에 기대한 줄이 없다: {line(installed)}"]


def test_every_expected_line_missing_but_other_lines_present_reports_both_kinds():
    # Arrange
    output = line("numpy") + "\n" + line("joblib") + "\n"

    # Act
    found = mismatches(output)

    # Assert
    assert found == [f"pip check 예상 밖 줄: {line('numpy')}"] + sorted(f"pip check 에 기대한 줄이 없다: {line(dep)}" for dep in SKIPPED if dep != "joblib")


@pytest.mark.parametrize("output", ["", "  ", "\n\n", " \t\n \n"])
def test_an_empty_pip_check_output_is_one_mismatch(output):
    # Act / Assert — 정상이어도 pip 는 줄을 쓴다. 빈 출력은 pip 가 아무것도 쓰지 않은 것이다(여섯 줄이 없다고 따로 나열하지 않는다)
    found = mismatches(output)
    assert len(found) == 1 and "비어 있다" in found[0]


@pytest.mark.parametrize("output", ["No broken requirements found.\n", "\n  No broken requirements found.  \n\n"])
def test_the_no_broken_requirements_message_alone_is_a_mismatch(output):
    # Act
    found = mismatches(output)

    # Assert — 기대한 줄이 하나도 없다는 뜻이다. 원인과 고칠 곳을 알려 준다
    assert len(found) == 1
    assert "No broken requirements found." in found[0] and "--expect-missing" in found[0]


def test_the_no_broken_requirements_message_next_to_other_lines_is_an_unexpected_line():
    # Act — pip 은 두 가지를 같이 쓰지 않지만, 섞여 나오면 예상 밖 줄이다
    found = mismatches("No broken requirements found.\n" + EXPECTED_PIP_CHECK)

    # Assert
    assert found == ["pip check 예상 밖 줄: No broken requirements found."]


@pytest.mark.parametrize(
    "bad_line",
    [
        f"{PROJECT} {VERSION} has requirement numpy<3.0,>=1.26, but you have numpy 3.0.0.",  # 버전 충돌
        "timesfm 2.0.2 requires torch, which is not installed.",  # 다른 패키지의 미설치
        "torch 2.13.0+cpu requires sympy, which is not installed.",
        line("numpy"),  # 기대 목록에 없는 기본 의존성
        line("some-new-dependency"),  # wheel 이 새 의존성을 얻은 경우
        line("joblib", version="0.2.0"),  # 다른 버전의 wheel 이 깔린 경우
        line("joblib", project="other-project"),
        line("joblib") + " extra",  # 줄 전체가 같아야 한다 — 부분 일치는 안 된다
        "prefix " + line("joblib"),
        "Traceback (most recent call last):",  # pip 자체가 죽은 경우
        "ERROR: something went wrong",
    ],
)
def test_any_other_line_is_flagged(bad_line):
    # Act
    found = mismatches(EXPECTED_PIP_CHECK + bad_line + "\n")

    # Assert — 기대한 6줄은 그대로 있으니 예상 밖 줄 하나뿐이다
    assert found == [f"pip check 예상 밖 줄: {bad_line}"]


def test_dependency_names_are_compared_in_canonical_form():
    # Arrange — pip 은 `python-dotenv`·`scikit-learn` 처럼 정규화한 이름을 낸다. 기대 목록은 어떤 표기든 받는다.
    expected = ["Python_Dotenv", "scikit.learn", "JOBLIB"]

    # Act / Assert
    assert mismatches("\n".join([line("python-dotenv"), line("scikit-learn"), line("joblib")]), expected=expected) == []


def test_expected_lines_follow_the_project_and_version_arguments():
    # Arrange — 기대 줄은 인자로 받은 wheel 이름·버전으로 만든다(다른 wheel 이 깔리면 같은 의존성 줄이라도 걸린다)
    output = line("joblib", project="other-project", version="9.9.9") + "\n"

    # Act / Assert
    assert mismatches(output, "other-project", "9.9.9", ["joblib"]) == []
    assert mismatches(output) == [f"pip check 예상 밖 줄: {output.strip()}"] + sorted(f"pip check 에 기대한 줄이 없다: {line(dep)}" for dep in SKIPPED)


@pytest.mark.parametrize(("raw", "expected"), [("a,b", ["a", "b"]), (" Joblib , scikit_learn ", ["joblib", "scikit-learn"]), ("x", ["x"])])
def test_parse_expect_missing_normalizes_names(raw, expected):
    # Act / Assert
    assert cbe.parse_expect_missing(raw) == expected


@pytest.mark.parametrize("raw", ["", "  ", ",", "a,,b", "a,", ",a"])
def test_parse_expect_missing_rejects_empty_lists_and_empty_items(raw):
    # Act / Assert
    with pytest.raises(ValueError):
        cbe.parse_expect_missing(raw)


# ── 설치 목록 ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "package",
    ["nvidia-nccl-cu12", "nvidia-cublas-cu12", "nvidia_cudnn_cu12", "NVIDIA-CUDA-RUNTIME-CU12", "cuda-bindings", "cuda-toolkit", "cuda-pathfinder", "triton"],
)
def test_gpu_stack_packages_are_detected_whatever_their_spelling(package):
    # Act
    found = cbe.gpu_packages(FREEZE_OK + f"{package}==1.0.0\n")

    # Assert
    assert found == [cbe.canonical(package)]


def test_cpu_stack_has_no_gpu_packages_and_lookalike_names_are_not_flagged():
    # Arrange — `nvidia` 가 이름 중간에 있거나 `cuda` 로 시작하지 않는 이름은 GPU 스택이 아니다
    freeze = FREEZE_OK + "not-nvidia-thing==1\nmycuda-tools==1\ntriton-helpers==1\n"

    # Act / Assert
    assert cbe.gpu_packages(freeze) == []
    assert cbe.gpu_packages("") == []


def test_installed_version_reads_the_freeze_text_by_canonical_name():
    # Act / Assert
    assert cbe.installed_version(FREEZE_OK, "Airlens_Inference") == VERSION
    assert cbe.installed_version(FREEZE_OK, "torch") == "2.13.0+cpu"
    assert cbe.installed_version(FREEZE_OK, "xgboost") is None
    assert cbe.installed_version("airlens-inference @ file:///x.whl\n", PROJECT) is None


# ── 종합 ─────────────────────────────────────────────────────────────────────
def test_a_clean_cpu_install_has_no_problems():
    # Act / Assert
    assert problems() == []


@pytest.mark.parametrize("output", ["", "  ", "\n\n", " \t\n \n"])
def test_an_empty_pip_check_output_is_a_problem_even_when_the_install_is_otherwise_clean(output):
    # Act
    found = problems(pip_check=output)

    # Assert — 정상이어도 pip 는 줄을 쓴다. 빈 출력은 pip 가 아무것도 쓰지 않은 것이다
    assert len(found) == 1 and "비어 있다" in found[0]


def test_the_no_broken_requirements_message_alone_is_a_problem_even_when_the_install_is_otherwise_clean():
    # Act
    found = problems(pip_check="No broken requirements found.\n")

    # Assert — 설치하지 않기로 한 의존성이 실제로는 깔렸다는 뜻이다. 막지 않으면 기대 목록이 현실과 조용히 어긋난다
    assert len(found) == 1 and "No broken requirements found." in found[0]


def test_each_failure_is_reported_on_its_own():
    # Act
    pip_check = problems(pip_check=EXPECTED_PIP_CHECK + line("numpy") + "\n")
    lost_line = problems(pip_check="\n".join(line(dep) for dep in SKIPPED[1:]) + "\n")
    wrong_wheel = problems(pip_list=FREEZE_OK.replace(f"{PROJECT}=={VERSION}", f"{PROJECT}==0.2.0"))
    no_wheel = problems(pip_list=FREEZE_OK.replace(f"{PROJECT}=={VERSION}\n", ""))
    gpu = problems(pip_list=FREEZE_OK + "nvidia-nccl-cu12==2.30.7\n")
    cuda = problems(cuda="12.8")
    cuda_import_failed = problems(cuda="")

    # Assert
    assert [len(p) for p in (pip_check, lost_line, wrong_wheel, no_wheel, gpu, cuda, cuda_import_failed)] == [1, 1, 1, 1, 1, 1, 1]
    assert "numpy" in pip_check[0] and "기대한 줄이 없다" in lost_line[0] and SKIPPED[0] in lost_line[0]
    assert "0.2.0" in wrong_wheel[0] and "없음" in no_wheel[0]
    assert "nvidia-nccl-cu12" in gpu[0]
    assert "12.8" in cuda[0] and "CPU" in cuda[0]


@pytest.mark.parametrize("value", ["None", "None\n", "  None  "])
def test_torch_cuda_value_none_with_surrounding_whitespace_is_a_cpu_build(value):
    # Act / Assert
    assert problems(cuda=value) == []


@pytest.mark.parametrize("value", ["none", "NONE", "'None'", "null", "False", "0", "12.8 None"])
def test_torch_cuda_value_must_be_exactly_none(value):
    # Act / Assert
    assert len(problems(cuda=value)) == 1


# ── CLI ──────────────────────────────────────────────────────────────────────
def write_inputs(tmp_path: Path, pip_check: str = EXPECTED_PIP_CHECK, pip_list: str = FREEZE_OK) -> list[str]:
    (tmp_path / "pip-check.txt").write_text(pip_check, encoding="utf-8")
    (tmp_path / "pip-list.txt").write_text(pip_list, encoding="utf-8")
    return ["--pip-check", str(tmp_path / "pip-check.txt"), "--pip-list", str(tmp_path / "pip-list.txt")]


def cli(tmp_path: Path, capsys, *, pip_check: str = EXPECTED_PIP_CHECK, pip_list: str = FREEZE_OK, cuda: str = "None", expect: str = ",".join(SKIPPED)):
    argv = write_inputs(tmp_path, pip_check, pip_list) + ["--torch-cuda", cuda, "--project", PROJECT, "--version", VERSION, "--expect-missing", expect]
    code = cbe.main(argv)
    return code, capsys.readouterr().out


def test_cli_passes_a_clean_install_and_says_what_was_checked(tmp_path, capsys):
    # Act
    code, out = cli(tmp_path, capsys)

    # Assert
    assert code == 0
    assert "통과" in out and "::error::" not in out
    assert f"{PROJECT}=={VERSION}" in out and "pip check 줄 6개가 기대와 정확히 같다" in out


def test_cli_success_message_counts_the_distinct_lines_pip_check_wrote(tmp_path, capsys):
    # Act — 같은 줄이 되풀이돼 12줄이 나와도 줄 집합은 6개다
    code, out = cli(tmp_path, capsys, pip_check=EXPECTED_PIP_CHECK + EXPECTED_PIP_CHECK)

    # Assert
    assert code == 0
    assert "pip check 줄 6개" in out and "12개" not in out


@pytest.mark.parametrize(
    ("pip_check", "needle"),
    [
        ("", "비어 있다"),
        ("  \n\n", "비어 있다"),
        ("No broken requirements found.\n", "No broken requirements found."),
        ("\n".join(line(dep) for dep in SKIPPED[:5]) + "\n", "기대한 줄이 없다"),  # 하나는 깔려서 그 줄이 없다
    ],
    ids=["empty", "blank-only", "no-broken-message", "one-expected-line-missing"],
)
def test_cli_fails_with_one_error_unless_pip_check_wrote_exactly_the_expected_lines(tmp_path, capsys, pip_check, needle):
    # Act
    code, out = cli(tmp_path, capsys, pip_check=pip_check)

    # Assert
    assert code == 1
    assert out.count("::error::band 설치 검사 실패") == 1 and needle in out
    assert "통과" not in out


def test_cli_reports_every_problem_as_an_error_annotation_and_exits_one(tmp_path, capsys):
    # Act
    code, out = cli(tmp_path, capsys, pip_check=EXPECTED_PIP_CHECK + line("numpy") + "\n", pip_list=FREEZE_OK + "triton==3.0\n", cuda="12.8")

    # Assert
    assert code == 1
    assert out.count("::error::band 설치 검사 실패") == 3


def test_cli_with_a_missing_input_file_exits_two_without_a_traceback(tmp_path, capsys):
    # Act
    code = cbe.main(["--pip-check", str(tmp_path / "nope.txt"), "--pip-list", str(tmp_path / "nope2.txt"), "--torch-cuda", "None",
                     "--project", PROJECT, "--version", VERSION, "--expect-missing", "joblib"])

    # Assert
    out = capsys.readouterr().out
    assert code == 2 and "FileNotFoundError" in out and "Traceback" not in out


def test_cli_with_an_empty_expected_list_exits_two(tmp_path, capsys):
    # Act
    code, out = cli(tmp_path, capsys, expect=" ")

    # Assert
    assert code == 2 and "기대 목록" in out


def test_cli_requires_every_argument():
    # Act / Assert
    with pytest.raises(SystemExit) as caught:
        cbe.main(["--pip-check", "a"])
    assert caught.value.code == 2
