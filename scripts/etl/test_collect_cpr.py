"""collect_cpr — 0건이면 cpr.json 을 덮어쓰지 않고 실패로 끝난다 (AAA, 네트워크 0).

Actions 의 수동 실행(`run_cpr`)은 `datasets` 를 깔지 않고 쓰기 토큰에 gated 읽기 권한도 없어서 수집이 늘 0건이다.
예전에는 그 0건을 cpr.json 에 써서 정책 발행이 CPR 없이 성공으로 끝났다.
"""
import json
import sys

import pytest

from collect_policies import collect_cpr as m

POLICY = {"id": "cpr-1", "countryCode": "KR", "type": "law"}


def run_main(monkeypatch, tmp_path, policies):
    monkeypatch.setattr(m, "BY_SOURCE_DIR", tmp_path)
    monkeypatch.setattr(m, "collect_from_huggingface", lambda skip_download=False: policies)
    monkeypatch.setattr(sys, "argv", ["collect_cpr.py"])
    m.main()


def test_zero_policies_fail_and_keep_the_previous_file(monkeypatch, tmp_path):
    # Arrange
    (tmp_path / "cpr.json").write_text(json.dumps([POLICY]), encoding="utf-8")

    # Act
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, tmp_path, [])

    # Assert — run_pipeline.sh 의 set -e 가 이 실패로 멈추고, 발행 스텝은 돌지 않는다
    assert exit_info.value.code == 1
    assert json.loads((tmp_path / "cpr.json").read_text(encoding="utf-8")) == [POLICY]


def test_missing_datasets_library_fails_without_writing(monkeypatch, tmp_path):
    # Arrange — Actions 와 같은 상태: `datasets` 가 없다. None 이면 import 가 ImportError 를 낸다(다운로드 0)
    fallback_calls = []
    monkeypatch.setitem(sys.modules, "datasets", None)
    monkeypatch.setattr(m, "collect_from_api_fallback", lambda: fallback_calls.append(1) or [])
    monkeypatch.setattr(m, "BY_SOURCE_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["collect_cpr.py"])

    # Act
    with pytest.raises(SystemExit) as exit_info:
        m.main()

    # Assert — pyarrow 를 지나 datasets import 에서 예비 경로로 갔다(pyarrow 가 없어 일찍 끝난 것이 아니다)
    assert fallback_calls == [1]
    assert exit_info.value.code == 1
    assert not (tmp_path / "cpr.json").exists()


def test_empty_cached_file_fails_with_skip_download(monkeypatch, tmp_path):
    # Arrange — 예전 코드가 써 둔 빈 캐시
    (tmp_path / "cpr.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(m, "BY_SOURCE_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["collect_cpr.py", "--skip-download"])

    # Act
    with pytest.raises(SystemExit) as exit_info:
        m.main()

    # Assert
    assert exit_info.value.code == 1


def test_collected_policies_are_written(monkeypatch, tmp_path):
    # Arrange / Act
    run_main(monkeypatch, tmp_path, [POLICY])

    # Assert
    assert json.loads((tmp_path / "cpr.json").read_text(encoding="utf-8")) == [POLICY]
