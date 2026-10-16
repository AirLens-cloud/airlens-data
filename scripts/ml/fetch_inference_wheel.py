#!/usr/bin/env python3
"""비공개 모델 저장소에서 추론 wheel 을 받아 핀과 대조한다 (stdlib 전용).

받는 것은 `inference/<version>/` 아래 3개뿐이다: wheel · SHA256SUMS · manifest.json.
`latest.json` 은 읽지 않는다 — 어떤 wheel 을 돌릴지는 워크플로의 핀(버전 + sha256)이 정하고,
배포 = 핀을 바꾸는 PR 이다.

아래가 전부 같을 때만 성공한다(하나라도 다르면 wheel 을 남기지 않고 죽는다):

    받은 wheel 의 sha256 = SHA256SUMS = manifest 의 wheel.sha256 = 핀
    manifest 의 version = 핀 버전

**토큰은 같은 origin(scheme · host · port)으로 가는 리다이렉트에만 따라간다.** 허브는 파일 요청을 같은
호스트의 다른 경로(`/api/resolve-cache/...`)로 넘기고, 비공개 저장소에서는 그 경로도 인증을 요구한다.
반면 LFS 파일은 다른 호스트의 서명 URL(CDN)로 넘기는데 거기에는 토큰이 가면 안 된다. urllib 의 기본
동작은 일반 헤더를 전부 복사하거나(`add_header`) unredirected 헤더를 전부 버려서(`add_unredirected_header`)
둘 다 맞지 않는다 — 첫 요청에는 `add_unredirected_header` 로 달고, 리다이렉트 hop 마다 origin 을 비교해
같을 때만 다시 붙인다(공식 클라이언트와 같은 정책).

**출력에 URL·저장소 id·응답 본문·토큰을 쓰지 않는다.** 이 레포는 공개라 로그도 공개다. 네트워크 실패는
예외 클래스 이름과 HTTP 상태만 남긴다(메시지·URL 은 호스트·경로를 담을 수 있어 찍지 않는다).
예상 밖 예외도 클래스 이름만 찍고 traceback 은 남기지 않는다. 저장소 id 는 환경변수로만 받는다.

환경변수:
  HF_MODELS_READ   읽기 전용 토큰 (필수)
  HF_MODELS_REPO   `<소유자>/<이름>` (필수)
  GITHUB_OUTPUT    있으면 `wheel=<경로>` 를 덧붙인다

사용:
    python3 scripts/ml/fetch_inference_wheel.py --version 0.1.0 --sha256 <64자리> --dest DIR
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

HF_HOST = "https://huggingface.co"
TOKEN_ENV = "HF_MODELS_READ"
REPO_ENV = "HF_MODELS_REPO"

TIMEOUT_S = 60
CHUNK = 1 << 20
MAX_META_BYTES = 1 << 20  # manifest·SHA256SUMS 는 수 KB — 1MB 가 넘으면 응답이 이상한 것이다
MAX_WHEEL_BYTES = 300 << 20  # 실제 wheel 은 수백 KB. 디스크를 채우는 응답을 막는 상한일 뿐이다
RETRIES = 3
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_SLEEP_S = (2.0, 5.0)

SHA256_RE = re.compile(r"[0-9a-f]{64}")
VERSION_RE = re.compile(r"[0-9][0-9A-Za-z.+_-]{0,31}")
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
TOKEN_RE = re.compile(r"[A-Za-z0-9_.=-]{8,256}")
WHEEL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\.whl")

# 일시 장애 후보. OSError 전체를 잡지 않는다 — 로컬 디스크 오류를 네트워크 오류로 오인해 되풀이하지 않게.
_NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, ssl.SSLError)


class FetchError(Exception):
    """사용자에게 그대로 보여 줘도 안전한 메시지만 담는다."""


class _Memory:
    """본문을 메모리에 모으는 sink."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def reset(self) -> None:
        self.buf.clear()

    def write(self, chunk: bytes) -> None:
        self.buf += chunk

    def data(self) -> bytes:
        return bytes(self.buf)


class _HashedFile:
    """파일에 쓰면서 sha256 을 같이 계산하는 sink."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.reset()

    def reset(self) -> None:
        self.hash = hashlib.sha256()
        self.path.write_bytes(b"")

    def write(self, chunk: bytes) -> None:
        with self.path.open("ab") as f:
            f.write(chunk)
        self.hash.update(chunk)


def _origin(url: str) -> tuple[str, str, int] | None:
    """(scheme, host, port). 기본 포트를 스킴에서 채워 같은 origin 이 같은 값이 되게 한다. 읽을 수 없으면 None."""
    parts = urllib.parse.urlsplit(url)
    scheme, host = parts.scheme, parts.hostname or ""  # urlsplit 이 스킴 · 호스트를 소문자로 준다
    try:
        port = parts.port
    except ValueError:  # 숫자가 아니거나 범위 밖인 포트
        return None
    if port is None:
        port = {"http": 80, "https": 443}.get(scheme)
    if not (scheme and host and port):  # 포트 0 · 모르는 스킴 · 상대 URL 도 비교할 수 없다
        return None
    return scheme, host, port


class _KeepAuthOnSameOrigin(urllib.request.HTTPRedirectHandler):
    """리다이렉트를 따라가되 Authorization 은 **같은 origin 으로 가는 hop** 에만 붙인다.

    첫 요청의 헤더는 unredirected 라 urllib 이 새 요청에 복사하지 않는다. 같은 origin 일 때만 여기서 다시 단다.
    다른 origin(CDN·스토리지)이나 스킴 다운그레이드, 다른 포트에는 가지 않고, 한 번 나간 뒤에는 돌아와도 붙지 않는다
    (hop 마다 바로 앞 요청과 비교하고, 앞 요청에 토큰이 없으면 붙일 것이 없다).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        auth = req.get_header("Authorization")
        if new is not None and auth:
            before, after = _origin(req.full_url), _origin(new.full_url)
            if before is not None and before == after:
                new.add_unredirected_header("Authorization", auth)
        return new


_OPENER = urllib.request.build_opener(_KeepAuthOnSameOrigin)


def _urlopen(req: urllib.request.Request):
    # 테스트가 바꿔 끼우는 지점.
    return _OPENER.open(req, timeout=TIMEOUT_S)  # noqa: S310 — https 고정 호스트


def _describe(exc: BaseException) -> str:
    """예외를 '클래스 + HTTP 상태' 로만 줄인다. str(exc) 는 URL·호스트·본문을 담을 수 있다."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTPError status={exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return f"URLError({type(exc.reason).__name__})"
    return type(exc).__name__


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in RETRY_STATUS
    return True  # _NETWORK_ERRORS 의 나머지(연결·타임아웃·끊김)는 일시 장애로 본다


def _request(url: str, token: str) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"User-Agent": "airlens-data-ml-predict"})
    # unredirected 헤더 — urllib 이 리다이렉트 요청에 복사하지 않는다. 같은 origin 일 때만 핸들러가 다시 단다.
    req.add_unredirected_header("Authorization", f"Bearer {token}")
    return req


def _get(
    what: str,
    url: str,
    token: str,
    *,
    limit: int,
    sink: _Memory | _HashedFile,
    sleep: Callable[[float], None],
) -> None:
    """url 의 본문을 sink 에 쓴다. 일시 장애는 RETRIES 회까지 처음부터 다시 받는다."""
    for attempt in range(1, RETRIES + 1):
        sink.reset()
        try:
            with _urlopen(_request(url, token)) as resp:
                total = 0
                while chunk := resp.read(CHUNK):
                    total += len(chunk)
                    if total > limit:
                        raise FetchError(f"{what}: 응답이 상한({limit} bytes)을 넘는다")
                    sink.write(chunk)
                return
        except _NETWORK_ERRORS as exc:
            if not _retryable(exc) or attempt == RETRIES:
                raise FetchError(f"{what}: {_describe(exc)}") from None
            print(f"retry {attempt}/{RETRIES - 1}: {what}: {_describe(exc)}", file=sys.stderr)
            sleep(RETRY_SLEEP_S[min(attempt - 1, len(RETRY_SLEEP_S) - 1)])


def _wheel_name_problem(name: object) -> str | None:
    if not isinstance(name, str) or not name:
        return "파일명이 문자열이 아니거나 비어 있다"
    if Path(name).name != name or not WHEEL_NAME_RE.fullmatch(name):
        return "평범한 .whl 파일명이 아니다"
    return None


def _manifest_fields(raw: bytes) -> tuple[str, str, str]:
    """(version, wheel 파일명, wheel sha256). 모양이 틀리면 FetchError."""
    try:
        manifest = json.loads(raw.decode("utf-8"))
        version = manifest["version"]
        name, sha = manifest["wheel"]["filename"], manifest["wheel"]["sha256"]
    except (ValueError, KeyError, TypeError):
        raise FetchError("manifest.json 을 읽을 수 없다(JSON 이 아니거나 필드가 없다)") from None
    if not isinstance(version, str) or not isinstance(sha, str):
        raise FetchError("manifest.json 의 version/sha256 이 문자열이 아니다")
    problem = _wheel_name_problem(name)
    if problem:
        raise FetchError(f"manifest.json 의 wheel 파일명이 올바르지 않다: {problem}")
    return version, name, sha


def fetch(
    version: str,
    sha256_pin: str,
    dest: Path,
    *,
    repo: str,
    token: str,
    base_url: str = HF_HOST,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """wheel + SHA256SUMS + manifest.json 을 dest 에 받고 wheel 경로를 돌려준다. 실패하면 FetchError."""
    # 입력 검증 — 잘못된 값은 되울리지 않는다(저장소 id·토큰이 로그에 남지 않게).
    if not SHA256_RE.fullmatch(sha256_pin):
        raise FetchError("--sha256 은 소문자 16진수 64자리여야 한다")
    if not VERSION_RE.fullmatch(version) or ".." in version:
        raise FetchError("--version 형식이 올바르지 않다")
    if not REPO_RE.fullmatch(repo):
        raise FetchError(f"{REPO_ENV} 가 없거나 형식이 올바르지 않다(<소유자>/<이름>)")
    if not TOKEN_RE.fullmatch(token):
        raise FetchError(f"{TOKEN_ENV} 가 없거나 형식이 올바르지 않다")

    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        # 받은 3개만 있어야 워크플로가 *.whl 글롭으로 wheel 하나를 집을 수 있다.
        raise FetchError("--dest 가 비어 있지 않다")

    def url_of(name: str) -> str:
        return f"{base_url}/{repo}/resolve/main/inference/{version}/{urllib.parse.quote(name)}"

    # 1) manifest — 작은 파일부터 받아 핀과 어긋나면 wheel 을 받기 전에 멈춘다.
    manifest = _Memory()
    _get("manifest.json", url_of("manifest.json"), token, limit=MAX_META_BYTES, sink=manifest, sleep=sleep)
    m_version, wheel_name, m_sha = _manifest_fields(manifest.data())
    if m_version != version:
        raise FetchError(f"manifest 의 version({m_version}) != 핀 버전({version})")
    if m_sha != sha256_pin:
        raise FetchError("manifest 의 wheel.sha256 != 핀")

    # 2) SHA256SUMS — `<sha256>  <파일명>` 한 줄.
    sums = _Memory()
    _get("SHA256SUMS", url_of("SHA256SUMS"), token, limit=MAX_META_BYTES, sink=sums, sleep=sleep)
    if sums.data().decode("utf-8", "replace").split() != [sha256_pin, wheel_name]:
        raise FetchError("SHA256SUMS 가 핀·manifest 와 다르다")

    # 3) wheel — 임시 이름으로 받아 해시가 맞을 때만 제 이름을 준다.
    part = dest / f".{wheel_name}.part"
    sink = _HashedFile(part)
    try:
        _get(wheel_name, url_of(wheel_name), token, limit=MAX_WHEEL_BYTES, sink=sink, sleep=sleep)
        actual = sink.hash.hexdigest()
        if actual != sha256_pin:
            raise FetchError(f"wheel sha256 불일치: 실제 {actual} != 핀 {sha256_pin}")
        wheel_path = dest / wheel_name
        part.replace(wheel_path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    (dest / "manifest.json").write_bytes(manifest.data())
    (dest / "SHA256SUMS").write_bytes(sums.data())
    return wheel_path


def main(argv: list[str] | None = None, environ: dict[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="핀 버전 (예: 0.1.0)")
    parser.add_argument("--sha256", required=True, help="핀 wheel sha256 (소문자 16진수 64자리)")
    parser.add_argument("--dest", required=True, type=Path, help="받을 빈 디렉터리")
    args = parser.parse_args(argv)

    try:
        wheel = fetch(
            args.version,
            args.sha256,
            args.dest,
            repo=env.get(REPO_ENV, "").strip(),
            token=env.get(TOKEN_ENV, "").strip(),
        )
    except FetchError as exc:
        print(f"fetch_inference_wheel: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — 예상 밖 예외의 메시지·traceback 에 URL 이 실릴 수 있어 클래스만 남긴다
        print(f"fetch_inference_wheel: 예상 밖 실패 {type(exc).__name__}", file=sys.stderr)
        return 1

    print(f"fetched {wheel.name} (sha256 {args.sha256}) + SHA256SUMS + manifest.json — 핀과 일치")
    out = env.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"wheel={wheel}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
