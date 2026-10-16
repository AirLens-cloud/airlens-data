"""check_band_env 시험 — `pip check` 출력은 실제 pip 26.2 가 wheel 을 `--no-deps` 로 깐 환경에서 받은 줄 그대로다."""
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


def problems(pip_check: str = EXPECTED_PIP_CHECK, pip_list: str = FREEZE_OK, cuda: str = "None", allowed=SKIPPED) -> list[str]:
    return cbe.problems(pip_check, pip_list, cuda, PROJECT, VERSION, allowed)


# ── pip check ────────────────────────────────────────────────────────────────
def test_the_expected_residue_of_the_skipped_dependencies_passes():
    # Act / Assert
    assert cbe.unexpected_pip_check_lines(EXPECTED_PIP_CHECK, PROJECT, VERSION, SKIPPED) == []


def test_real_pip_check_output_of_a_bare_wheel_install_flags_exactly_the_dependencies_the_closure_must_cover():
    # Act — 폐포를 아직 깔지 않았다면 허용 목록 밖의 4개(numpy·pandas·pyyaml·requests)가 걸린다
    unexpected = cbe.unexpected_pip_check_lines(REAL_PIP_CHECK_ALL_TEN, PROJECT, VERSION, SKIPPED)

    # Assert
    assert sorted(unexpected) == sorted(line(dep) for dep in COVERED)


def test_no_broken_requirements_message_and_blank_lines_are_fine():
    # Act / Assert
    assert cbe.unexpected_pip_check_lines("No broken requirements found.\n\n", PROJECT, VERSION, SKIPPED) == []
    assert cbe.unexpected_pip_check_lines("", PROJECT, VERSION, SKIPPED) == []
    assert cbe.unexpected_pip_check_lines("  \n\n", PROJECT, VERSION, SKIPPED) == []


@pytest.mark.parametrize(
    "bad_line",
    [
        f"{PROJECT} {VERSION} has requirement numpy<3.0,>=1.26, but you have numpy 3.0.0.",  # 버전 충돌
        "timesfm 2.0.2 requires torch, which is not installed.",  # 다른 패키지의 미설치
        "torch 2.13.0+cpu requires sympy, which is not installed.",
        line("numpy"),  # 허용 목록에 없는 기본 의존성
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
    unexpected = cbe.unexpected_pip_check_lines(EXPECTED_PIP_CHECK + bad_line + "\n", PROJECT, VERSION, SKIPPED)

    # Assert
    assert unexpected == [bad_line]


def test_dependency_names_are_compared_in_canonical_form():
    # Arrange — pip 은 `python-dotenv`·`scikit-learn` 처럼 정규화한 이름을 낸다. 허용 목록은 어떤 표기든 받는다.
    allowed = ["Python_Dotenv", "scikit.learn", "JOBLIB"]

    # Act
    unexpected = cbe.unexpected_pip_check_lines("\n".join([line("python-dotenv"), line("scikit-learn"), line("joblib")]), PROJECT, VERSION, allowed)

    # Assert
    assert unexpected == []


def test_allowed_lines_follow_the_project_and_version_arguments():
    # Arrange — 허용 줄은 인자로 받은 wheel 이름·버전으로 만든다(다른 wheel 이 깔리면 같은 의존성 줄이라도 걸린다)
    output = line("joblib", project="other-project", version="9.9.9") + "\n"

    # Act / Assert
    assert cbe.unexpected_pip_check_lines(output, "other-project", "9.9.9", ["joblib"]) == []
    assert cbe.unexpected_pip_check_lines(output, PROJECT, VERSION, ["joblib"]) == [output.strip()]


@pytest.mark.parametrize(("raw", "expected"), [("a,b", ["a", "b"]), (" Joblib , scikit_learn ", ["joblib", "scikit-learn"]), ("x", ["x"])])
def test_parse_allow_missing_normalizes_names(raw, expected):
    # Act / Assert
    assert cbe.parse_allow_missing(raw) == expected


@pytest.mark.parametrize("raw", ["", "  ", ",", "a,,b", "a,", ",a"])
def test_parse_allow_missing_rejects_empty_lists_and_empty_items(raw):
    # Act / Assert
    with pytest.raises(ValueError):
        cbe.parse_allow_missing(raw)


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


def test_each_failure_is_reported_on_its_own():
    # Act
    pip_check = problems(pip_check=EXPECTED_PIP_CHECK + line("numpy") + "\n")
    wrong_wheel = problems(pip_list=FREEZE_OK.replace(f"{PROJECT}=={VERSION}", f"{PROJECT}==0.2.0"))
    no_wheel = problems(pip_list=FREEZE_OK.replace(f"{PROJECT}=={VERSION}\n", ""))
    gpu = problems(pip_list=FREEZE_OK + "nvidia-nccl-cu12==2.30.7\n")
    cuda = problems(cuda="12.8")
    cuda_import_failed = problems(cuda="")

    # Assert
    assert [len(p) for p in (pip_check, wrong_wheel, no_wheel, gpu, cuda, cuda_import_failed)] == [1, 1, 1, 1, 1, 1]
    assert "numpy" in pip_check[0]
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


def cli(tmp_path: Path, capsys, *, pip_check: str = EXPECTED_PIP_CHECK, pip_list: str = FREEZE_OK, cuda: str = "None", allow: str = ",".join(SKIPPED)):
    argv = write_inputs(tmp_path, pip_check, pip_list) + ["--torch-cuda", cuda, "--project", PROJECT, "--version", VERSION, "--allow-missing", allow]
    code = cbe.main(argv)
    return code, capsys.readouterr().out


def test_cli_passes_a_clean_install_and_says_what_was_checked(tmp_path, capsys):
    # Act
    code, out = cli(tmp_path, capsys)

    # Assert
    assert code == 0
    assert "통과" in out and "::error::" not in out
    assert f"{PROJECT}=={VERSION}" in out and "6개" in out


def test_cli_reports_every_problem_as_an_error_annotation_and_exits_one(tmp_path, capsys):
    # Act
    code, out = cli(tmp_path, capsys, pip_check=EXPECTED_PIP_CHECK + line("numpy") + "\n", pip_list=FREEZE_OK + "triton==3.0\n", cuda="12.8")

    # Assert
    assert code == 1
    assert out.count("::error::band 설치 검사 실패") == 3


def test_cli_with_a_missing_input_file_exits_two_without_a_traceback(tmp_path, capsys):
    # Act
    code = cbe.main(["--pip-check", str(tmp_path / "nope.txt"), "--pip-list", str(tmp_path / "nope2.txt"), "--torch-cuda", "None",
                     "--project", PROJECT, "--version", VERSION, "--allow-missing", "joblib"])

    # Assert
    out = capsys.readouterr().out
    assert code == 2 and "FileNotFoundError" in out and "Traceback" not in out


def test_cli_with_an_empty_allow_list_exits_two(tmp_path, capsys):
    # Act
    code, out = cli(tmp_path, capsys, allow=" ")

    # Assert
    assert code == 2 and "허용 목록" in out


def test_cli_requires_every_argument():
    # Act / Assert
    with pytest.raises(SystemExit) as caught:
        cbe.main(["--pip-check", "a"])
    assert caught.value.code == 2
