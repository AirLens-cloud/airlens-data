"""fetch_inference_wheel.py 시험 (AAA, 비공개 저장소 접근 0).

두 갈래로 시험한다.
1. 가짜 `_urlopen` — 핀 대조·재시도·오류 출력의 세부를 빠르고 결정적으로 본다.
2. 로컬 `http.server` 두 개(허브 + CDN) — **실제 urllib** 이 리다이렉트를 따라갈 때 토큰이 어디에 가는지 본다:
   허브 안(같은 origin)의 리다이렉트에는 따라가고(비공개 저장소는 그 경로도 인증을 요구한다), CDN(다른 origin)
   에는 가지 않는다. 가짜 `_urlopen` 으로는 확인할 수 없다(리다이렉트는 urllib 핸들러가 한다).

가짜 토큰·저장소 id 는 시험용 문자열이다. 실제 값이 아니다.
"""
from __future__ import annotations

import hashlib
import http.client
import http.server
import io
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import fetch_inference_wheel as fiw

VERSION = "0.1.0"
WHEEL_NAME = "pkg_under_test-0.1.0-py3-none-any.whl"
WHEEL_BYTES = b"PK\x03\x04 not a real wheel, only bytes whose hash is pinned"
PIN = hashlib.sha256(WHEEL_BYTES).hexdigest()
REPO = "example-org/example-models"
TOKEN = "unit-test-token-value"
SECRET_BODY = "secret-body-must-not-be-printed"


def _manifest(version: str = VERSION, sha: str = PIN, name: str = WHEEL_NAME) -> bytes:
    return json.dumps({"name": "pkg-under-test", "version": version,
                       "wheel": {"filename": name, "sha256": sha, "size_bytes": len(WHEEL_BYTES)}}).encode()


def _sums(sha: str = PIN, name: str = WHEEL_NAME) -> bytes:
    return f"{sha}  {name}\n".encode()


class _Resp:
    def __init__(self, body: bytes) -> None:
        self._buf = io.BytesIO(body)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class _CutResp(_Resp):
    """첫 read 에서 일부만 주고 두 번째 read 에서 연결이 끊긴다."""

    def __init__(self, body: bytes) -> None:
        super().__init__(body)
        self._calls = 0

    def read(self, n: int = -1) -> bytes:
        self._calls += 1
        if self._calls == 1:
            return super().read(5)
        raise ConnectionResetError("cut")


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(f"https://huggingface.co/{REPO}/resolve/main/x", code, SECRET_BODY, {}, io.BytesIO(SECRET_BODY.encode()))


class _FakeNet:
    """파일명 → 결과 목록(bytes · 예외 · 응답 객체). 같은 파일을 다시 부르면 다음 결과, 마지막은 반복."""

    def __init__(self, routes: dict[str, list[object]]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, req: urllib.request.Request):
        name = req.full_url.rsplit("/", 1)[-1]
        self.calls.append((name, dict(req.header_items())))
        seen = sum(1 for n, _ in self.calls if n == name)
        outcomes = self.routes[name]
        outcome = outcomes[min(seen, len(outcomes)) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, bytes):
            return _Resp(outcome)
        return outcome

    def names(self) -> list[str]:
        return [n for n, _ in self.calls]


def _good_routes() -> dict[str, list[object]]:
    return {"manifest.json": [_manifest()], "SHA256SUMS": [_sums()], WHEEL_NAME: [WHEEL_BYTES]}


def _fetch(dest: Path, net: _FakeNet, monkeypatch: pytest.MonkeyPatch, *, sleeps: list[float] | None = None) -> Path:
    monkeypatch.setattr(fiw, "_urlopen", net)
    return fiw.fetch(VERSION, PIN, dest, repo=REPO, token=TOKEN, sleep=(sleeps.append if sleeps is not None else (lambda _s: None)))


def _dest_names(dest: Path) -> list[str]:
    return sorted(p.name for p in dest.iterdir())


# ── 성공 경로 ────────────────────────────────────────────────────────────────
def test_fetch_writes_exactly_wheel_sums_and_manifest(tmp_path, monkeypatch):
    # Arrange
    dest = tmp_path / "wheel"

    # Act
    wheel = _fetch(dest, _FakeNet(_good_routes()), monkeypatch)

    # Assert — 3개뿐이고 임시 파일(.part)은 남지 않는다. wheel 내용이 핀과 같다.
    assert wheel == dest / WHEEL_NAME
    assert _dest_names(dest) == sorted([WHEEL_NAME, "SHA256SUMS", "manifest.json"])
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == PIN


def test_fetch_downloads_small_metadata_before_the_wheel(tmp_path, monkeypatch):
    # Arrange
    net = _FakeNet(_good_routes())

    # Act
    _fetch(tmp_path / "w", net, monkeypatch)

    # Assert — 핀과 어긋나면 wheel 을 받기 전에 멈추려면 작은 파일이 먼저여야 한다. latest.json 은 읽지 않는다.
    assert net.names() == ["manifest.json", "SHA256SUMS", WHEEL_NAME]


def test_fetch_sends_bearer_token_and_asks_the_pinned_version_path(tmp_path, monkeypatch):
    # Arrange
    urls: list[str] = []
    net = _FakeNet(_good_routes())
    orig = net.__call__

    def spy(req):
        urls.append(req.full_url)
        return orig(req)

    # Act
    monkeypatch.setattr(fiw, "_urlopen", spy)
    fiw.fetch(VERSION, PIN, tmp_path / "w", repo=REPO, token=TOKEN, sleep=lambda _s: None)

    # Assert
    assert urls[0] == f"https://huggingface.co/{REPO}/resolve/main/inference/{VERSION}/manifest.json"
    assert all(headers.get("Authorization") == f"Bearer {TOKEN}" for _, headers in net.calls)


# ── 핀 대조 실패 ─────────────────────────────────────────────────────────────
def test_wheel_with_different_bytes_is_rejected_and_not_left_behind(tmp_path, monkeypatch):
    # Arrange — 메타데이터는 핀과 같지만 실제 받은 wheel 이 다르다(변조된 저장소)
    routes = _good_routes()
    routes[WHEEL_NAME] = [WHEEL_BYTES + b"tampered"]
    dest = tmp_path / "w"

    # Act / Assert
    with pytest.raises(fiw.FetchError, match="sha256 불일치"):
        _fetch(dest, _FakeNet(routes), monkeypatch)
    assert _dest_names(dest) == []


def test_manifest_version_different_from_pin_stops_before_the_wheel(tmp_path, monkeypatch):
    # Arrange
    routes = _good_routes()
    routes["manifest.json"] = [_manifest(version="0.2.0")]
    net = _FakeNet(routes)

    # Act / Assert
    with pytest.raises(fiw.FetchError, match="version"):
        _fetch(tmp_path / "w", net, monkeypatch)
    assert WHEEL_NAME not in net.names()


def test_manifest_wheel_sha_different_from_pin_stops_before_the_wheel(tmp_path, monkeypatch):
    # Arrange
    routes = _good_routes()
    routes["manifest.json"] = [_manifest(sha="0" * 64)]
    net = _FakeNet(routes)

    # Act / Assert
    with pytest.raises(fiw.FetchError, match="wheel.sha256 != 핀"):
        _fetch(tmp_path / "w", net, monkeypatch)
    assert WHEEL_NAME not in net.names()


@pytest.mark.parametrize(
    "sums",
    [
        _sums(sha="1" * 64),  # 해시가 다르다
        _sums(name="other-0.1.0-py3-none-any.whl"),  # 파일명이 다르다
        b"",  # 비었다
        _sums() + _sums(),  # 줄이 둘이다
    ],
    ids=["other-hash", "other-name", "empty", "two-lines"],
)
def test_sha256sums_that_disagrees_with_pin_and_manifest_is_rejected(sums, tmp_path, monkeypatch):
    # Arrange
    routes = _good_routes()
    routes["SHA256SUMS"] = [sums]
    net = _FakeNet(routes)

    # Act / Assert
    with pytest.raises(fiw.FetchError, match="SHA256SUMS"):
        _fetch(tmp_path / "w", net, monkeypatch)
    assert WHEEL_NAME not in net.names()


@pytest.mark.parametrize("name", ["../evil.whl", "sub/dir.whl", "/abs.whl", "plain.txt", "", ".hidden", "a b.whl"])
def test_wheel_filename_that_escapes_dest_or_is_not_a_wheel_is_rejected(name, tmp_path, monkeypatch):
    # Arrange
    routes = {"manifest.json": [_manifest(name=name)]}
    dest = tmp_path / "w"

    # Act / Assert
    with pytest.raises(fiw.FetchError, match="파일명"):
        _fetch(dest, _FakeNet(routes), monkeypatch)
    assert _dest_names(dest) == []
    assert not (tmp_path / "evil.whl").exists()


@pytest.mark.parametrize("raw", [b"not json", b"[]", b'{"version": "0.1.0"}', b'{"version": 1, "wheel": {"filename": "a.whl", "sha256": "x"}}'])
def test_unreadable_manifest_is_rejected(raw, tmp_path, monkeypatch):
    # Arrange
    routes = {"manifest.json": [raw]}

    # Act / Assert
    with pytest.raises(fiw.FetchError, match="manifest"):
        _fetch(tmp_path / "w", _FakeNet(routes), monkeypatch)


def test_dest_that_is_not_empty_is_rejected(tmp_path, monkeypatch):
    # Arrange — 워크플로가 *.whl 글롭으로 wheel 하나를 집으려면 받은 3개만 있어야 한다
    dest = tmp_path / "w"
    dest.mkdir()
    (dest / "stale.whl").write_bytes(b"old")
    net = _FakeNet(_good_routes())

    # Act / Assert
    with pytest.raises(fiw.FetchError, match="비어 있지 않다"):
        _fetch(dest, net, monkeypatch)
    assert net.calls == []


def test_response_over_the_limit_is_rejected_and_part_file_removed(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.setattr(fiw, "MAX_WHEEL_BYTES", 10)
    dest = tmp_path / "w"

    # Act / Assert
    with pytest.raises(fiw.FetchError, match="상한"):
        _fetch(dest, _FakeNet(_good_routes()), monkeypatch)
    assert _dest_names(dest) == []


# ── 입력 검증: 잘못된 값은 되울리지 않는다 ───────────────────────────────────
@pytest.mark.parametrize("pin", ["", "A" * 64, "a" * 63, "g" * 64, PIN + "0"])
def test_malformed_sha256_pin_is_rejected_without_network(pin, tmp_path, monkeypatch):
    # Arrange
    net = _FakeNet(_good_routes())
    monkeypatch.setattr(fiw, "_urlopen", net)

    # Act / Assert
    with pytest.raises(fiw.FetchError, match="sha256"):
        fiw.fetch(VERSION, pin, tmp_path / "w", repo=REPO, token=TOKEN)
    assert net.calls == []


@pytest.mark.parametrize("version", ["", "latest", "../x", "0.1.0/../../y", "1..2", "0.1 0", "0.1.0\n"])
def test_malformed_version_is_rejected_without_network(version, tmp_path, monkeypatch):
    # Arrange
    net = _FakeNet(_good_routes())
    monkeypatch.setattr(fiw, "_urlopen", net)

    # Act / Assert
    with pytest.raises(fiw.FetchError, match="version"):
        fiw.fetch(version, PIN, tmp_path / "w", repo=REPO, token=TOKEN)
    assert net.calls == []


@pytest.mark.parametrize("repo", ["", "no-slash", "a/b/c", "/lead/x", "owner/", "ow ner/name", "owner/na me"])
def test_malformed_repo_is_rejected_and_value_is_not_echoed(repo, tmp_path):
    # Act / Assert
    with pytest.raises(fiw.FetchError) as exc:
        fiw.fetch(VERSION, PIN, tmp_path / "w", repo=repo, token=TOKEN)
    assert "HF_MODELS_REPO" in str(exc.value)
    if repo:
        assert repo not in str(exc.value)


@pytest.mark.parametrize("token", ["", "short", "has space in it", "new\nline-token", "a" * 257])
def test_malformed_token_is_rejected_and_value_is_not_echoed(token, tmp_path):
    # Act / Assert
    with pytest.raises(fiw.FetchError) as exc:
        fiw.fetch(VERSION, PIN, tmp_path / "w", repo=REPO, token=token)
    assert "HF_MODELS_READ" in str(exc.value)
    if token.strip():
        assert token not in str(exc.value)


# ── 재시도 ───────────────────────────────────────────────────────────────────
def test_transient_503_is_retried_with_backoff_then_succeeds(tmp_path, monkeypatch):
    # Arrange — manifest 가 503 두 번 뒤 정상
    routes = _good_routes()
    routes["manifest.json"] = [_http_error(503), _http_error(503), _manifest()]
    net, sleeps = _FakeNet(routes), []

    # Act
    wheel = _fetch(tmp_path / "w", net, monkeypatch, sleeps=sleeps)

    # Assert
    assert wheel.read_bytes() == WHEEL_BYTES
    assert net.names().count("manifest.json") == 3
    assert sleeps == [2.0, 5.0]


def test_persistent_503_gives_up_after_three_attempts_with_status_only(tmp_path, monkeypatch):
    # Arrange
    routes = _good_routes()
    routes["manifest.json"] = [_http_error(503)]
    net = _FakeNet(routes)

    # Act / Assert
    with pytest.raises(fiw.FetchError, match="HTTPError status=503") as exc:
        _fetch(tmp_path / "w", net, monkeypatch)
    assert net.names().count("manifest.json") == fiw.RETRIES
    assert SECRET_BODY not in str(exc.value)


@pytest.mark.parametrize("code", [401, 403, 404])
def test_client_errors_are_not_retried(code, tmp_path, monkeypatch):
    # Arrange
    routes = _good_routes()
    routes["manifest.json"] = [_http_error(code)]
    net = _FakeNet(routes)

    # Act / Assert
    with pytest.raises(fiw.FetchError, match=f"HTTPError status={code}"):
        _fetch(tmp_path / "w", net, monkeypatch)
    assert net.names() == ["manifest.json"]


def test_connection_cut_midway_restarts_the_download_from_scratch(tmp_path, monkeypatch):
    # Arrange — 첫 시도는 5바이트만 받고 끊긴다. 이어붙이면 해시가 깨진다.
    routes = _good_routes()
    routes[WHEEL_NAME] = [_CutResp(WHEEL_BYTES), WHEEL_BYTES]
    net = _FakeNet(routes)

    # Act
    wheel = _fetch(tmp_path / "w", net, monkeypatch)

    # Assert
    assert net.names().count(WHEEL_NAME) == 2
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == PIN


# ── 출력: URL·저장소·본문·토큰이 나가지 않는다 ───────────────────────────────
ARGV = ["--version", VERSION, "--sha256", PIN]
ENV = {"HF_MODELS_READ": TOKEN, "HF_MODELS_REPO": REPO}


def _assert_no_secrets(text: str) -> None:
    for needle in (TOKEN, REPO, "example-org", "huggingface.co", SECRET_BODY, "https://"):
        assert needle not in text, f"출력에 {needle!r} 가 남았다"


def test_main_http_error_prints_status_only(tmp_path, monkeypatch, capsys):
    # Arrange
    routes = _good_routes()
    routes["manifest.json"] = [_http_error(404)]
    monkeypatch.setattr(fiw, "_urlopen", _FakeNet(routes))

    # Act
    rc = fiw.main([*ARGV, "--dest", str(tmp_path / "w")], dict(ENV))

    # Assert
    captured = capsys.readouterr()
    assert rc == 1
    assert "HTTPError status=404" in captured.err
    _assert_no_secrets(captured.out + captured.err)


def test_main_url_error_prints_reason_class_only(tmp_path, monkeypatch, capsys):
    # Arrange — URLError 의 reason 은 호스트 이름을 문자열에 담을 수 있다
    routes = _good_routes()
    routes["manifest.json"] = [urllib.error.URLError(OSError(f"cannot reach huggingface.co for {REPO} with {TOKEN}"))]
    monkeypatch.setattr(fiw, "_urlopen", _FakeNet(routes))
    monkeypatch.setattr(fiw, "RETRY_SLEEP_S", (0.0, 0.0))  # main() 은 실제 sleep 을 쓴다 — 대기 시간만 0 으로

    # Act
    rc = fiw.main([*ARGV, "--dest", str(tmp_path / "w")], dict(ENV))

    # Assert
    captured = capsys.readouterr()
    assert rc == 1
    assert "URLError(OSError)" in captured.err
    _assert_no_secrets(captured.out + captured.err)


def test_main_unexpected_exception_prints_class_name_only(tmp_path, monkeypatch, capsys):
    # Arrange — 예상 밖 예외의 메시지·traceback 에 URL 이 실릴 수 있다
    def boom(_req):
        raise RuntimeError(f"https://huggingface.co/{REPO} {TOKEN} {SECRET_BODY}")

    monkeypatch.setattr(fiw, "_urlopen", boom)

    # Act
    rc = fiw.main([*ARGV, "--dest", str(tmp_path / "w")], dict(ENV))

    # Assert
    captured = capsys.readouterr()
    assert rc == 1
    assert "RuntimeError" in captured.err
    assert "Traceback" not in captured.err
    _assert_no_secrets(captured.out + captured.err)


@pytest.mark.parametrize("env", [{}, {"HF_MODELS_READ": TOKEN}, {"HF_MODELS_REPO": REPO}, {"HF_MODELS_READ": "  ", "HF_MODELS_REPO": REPO}])
def test_main_missing_secret_env_fails_with_variable_name_only(env, tmp_path, capsys):
    # Act
    rc = fiw.main([*ARGV, "--dest", str(tmp_path / "w")], dict(env))

    # Assert
    captured = capsys.readouterr()
    assert rc == 1
    assert "HF_MODELS_" in captured.err
    _assert_no_secrets(captured.out + captured.err)


def test_main_success_reports_pin_match_and_appends_github_output(tmp_path, monkeypatch, capsys):
    # Arrange
    monkeypatch.setattr(fiw, "_urlopen", _FakeNet(_good_routes()))
    out_file = tmp_path / "github_output"
    dest = tmp_path / "w"

    # Act
    rc = fiw.main([*ARGV, "--dest", str(dest)], {**ENV, "GITHUB_OUTPUT": str(out_file)})

    # Assert
    captured = capsys.readouterr()
    assert rc == 0
    assert "핀과 일치" in captured.out
    assert out_file.read_text(encoding="utf-8") == f"wheel={dest / WHEEL_NAME}\n"
    _assert_no_secrets(captured.out + captured.err)


# ── 실제 urllib 리다이렉트: 같은 origin 에는 토큰이 따라가고 CDN 에는 가지 않는다 ──────────
class _Hub:
    """허브 + CDN. 요청 헤더를 기록한다.

    `resolve_cache_hop=True` 면 허브가 모든 파일 요청을 **같은 호스트의 상대 경로**(`/api/resolve-cache/...`)로
    307 한다 — 실제 허브가 하는 일이다. 그 경로는 비공개 저장소처럼 Authorization 이 없으면 401 이다.
    wheel 은 두 경우 모두 마지막에 다른 origin(CDN)으로 302 된다.
    """

    def __init__(self, *, resolve_cache_hop: bool = False) -> None:
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.cdn = self._serve("cdn", {"/blob/wheel": (200, {}, WHEEL_BYTES)}, set())
        prefix = f"/{REPO}/resolve/main/inference/{VERSION}"
        cache = f"/api/resolve-cache/models/{REPO}/0123abcd/inference%2F{VERSION}%2F"
        cdn_target = f"{self.cdn_url}/blob/wheel"
        routes: dict[str, tuple[int, dict[str, str], bytes]] = {}
        protected: set[str] = set()
        for name, body in (("manifest.json", _manifest()), ("SHA256SUMS", _sums())):
            if resolve_cache_hop:
                routes[f"{prefix}/{name}"] = (307, {"Location": f"{cache}{name}?etag=x"}, b"")
                routes[f"{cache}{name}?etag=x"] = (200, {}, body)
                protected.add(f"{cache}{name}?etag=x")
            else:
                routes[f"{prefix}/{name}"] = (200, {}, body)
        if resolve_cache_hop:
            routes[f"{prefix}/{WHEEL_NAME}"] = (307, {"Location": f"{cache}{WHEEL_NAME}?etag=x"}, b"")
            routes[f"{cache}{WHEEL_NAME}?etag=x"] = (302, {"Location": cdn_target}, b"")
            protected.add(f"{cache}{WHEEL_NAME}?etag=x")
        else:
            routes[f"{prefix}/{WHEEL_NAME}"] = (302, {"Location": cdn_target}, b"")
        self.hub = self._serve("hub", routes, protected)

    @property
    def cdn_url(self) -> str:
        return f"http://127.0.0.1:{self.cdn.server_port}"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.hub.server_port}"

    def _serve(self, role: str, routes: dict[str, tuple[int, dict[str, str], bytes]], protected: set[str]) -> http.server.ThreadingHTTPServer:
        recorder = self.requests

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 — http.server 규약
                recorder.append((role, self.path, {k.lower(): v for k, v in self.headers.items()}))
                if self.path in protected and "Authorization" not in self.headers:
                    status, headers, body = 401, {}, b""
                else:
                    status, headers, body = routes.get(self.path, (404, {}, SECRET_BODY.encode()))
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def close(self) -> None:
        for server in (self.hub, self.cdn):
            server.shutdown()
            server.server_close()


def _no_proxy(monkeypatch) -> None:
    # 개발 머신의 프록시 설정이 루프백 요청을 가로채지 않게 한다(NO_PROXY 가 있으면 환경 프록시만 쓰고 시스템 프록시는 안 본다).
    for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")


@pytest.fixture
def hub(monkeypatch):
    _no_proxy(monkeypatch)
    server = _Hub()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture
def cache_hub(monkeypatch):
    _no_proxy(monkeypatch)
    server = _Hub(resolve_cache_hop=True)
    try:
        yield server
    finally:
        server.close()


def test_token_goes_to_the_hub_only_not_to_the_redirect_target(hub, tmp_path):
    # Arrange — 실제 urllib 이 허브의 302 를 따라 CDN 으로 간다
    dest = tmp_path / "w"

    # Act
    wheel = fiw.fetch(VERSION, PIN, dest, repo=REPO, token=TOKEN, base_url=hub.base_url, sleep=lambda _s: None)

    # Assert
    hub_calls = [(path, h) for role, path, h in hub.requests if role == "hub"]
    cdn_calls = [(path, h) for role, path, h in hub.requests if role == "cdn"]
    assert wheel.read_bytes() == WHEEL_BYTES
    assert len(hub_calls) == 3
    assert all(h.get("authorization") == f"Bearer {TOKEN}" for _, h in hub_calls), "허브 요청에는 토큰이 있어야 한다"
    assert len(cdn_calls) == 1, "wheel 은 리다이렉트된 CDN 에서 받았어야 한다"
    assert "authorization" not in cdn_calls[0][1], "리다이렉트된 요청에 토큰이 갔다"


def test_token_follows_same_origin_redirects_inside_the_hub_but_still_not_to_the_cdn(cache_hub, tmp_path):
    # Arrange — 실제 허브처럼 모든 파일이 같은 호스트의 `/api/resolve-cache/...` 로 307 되고, 그 경로는 인증이 없으면 401 이다
    dest = tmp_path / "w"

    # Act
    wheel = fiw.fetch(VERSION, PIN, dest, repo=REPO, token=TOKEN, base_url=cache_hub.base_url, sleep=lambda _s: None)

    # Assert
    hub_calls = [(path, h) for role, path, h in cache_hub.requests if role == "hub"]
    cdn_calls = [(path, h) for role, path, h in cache_hub.requests if role == "cdn"]
    assert wheel.read_bytes() == WHEEL_BYTES
    assert len(hub_calls) == 6  # 파일 3개 × (resolve + resolve-cache)
    assert sum("/api/resolve-cache/" in path for path, _ in hub_calls) == 3
    assert all(h.get("authorization") == f"Bearer {TOKEN}" for _, h in hub_calls), "허브 안의 리다이렉트에도 토큰이 따라가야 한다"
    assert len(cdn_calls) == 1
    assert "authorization" not in cdn_calls[0][1], "다른 origin(CDN)에는 토큰이 가면 안 된다"


def test_a_protected_hop_without_the_token_is_a_401_not_a_silent_success(cache_hub):
    # Arrange — 대조군: 토큰 없이 같은 경로를 부르면 401 이다(= 위 시험의 서버가 인증을 실제로 요구한다)
    prefix = f"/{REPO}/resolve/main/inference/{VERSION}/manifest.json"

    # Act / Assert
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(cache_hub.base_url + prefix, timeout=10)
    assert exc.value.code == 401


ORIGIN_URL = f"https://huggingface.co/{REPO}/resolve/main/inference/{VERSION}/manifest.json"


def _redirected(new_url: str, *, with_token: bool = True) -> urllib.request.Request | None:
    req = urllib.request.Request(ORIGIN_URL)
    if with_token:
        req.add_unredirected_header("Authorization", f"Bearer {TOKEN}")
    handler = fiw._KeepAuthOnSameOrigin()
    return handler.redirect_request(req, io.BytesIO(b""), 307, "Temporary Redirect", {}, new_url)


@pytest.mark.parametrize(
    ("new_url", "forwarded"),
    [
        ("https://huggingface.co/api/resolve-cache/models/x/abc/f?y=1", True),
        ("https://HuggingFace.co/api/x", True),  # 호스트는 대소문자를 구분하지 않는다
        ("https://huggingface.co:443/api/x", True),  # 명시한 기본 포트
        ("https://cdn-lfs.hf.co/repos/aa/bb/blob?sig=1", False),  # 다른 호스트(서명 URL)
        ("https://hf.co/x", False),  # 같은 허브 계열이어도 호스트가 다르면 보수적으로 붙이지 않는다
        ("https://huggingface.co.evil.example/x", False),
        ("https://huggingface.co./x", False),  # 끝 점이 붙은 FQDN 은 다른 호스트 문자열이다 — 보수적으로 붙이지 않는다
        ("https://huggingface.co@evil.example/x", False),  # userinfo 로 호스트를 속이는 형태
        ("http://huggingface.co/x", False),  # 스킴 다운그레이드
        ("https://huggingface.co:8443/x", False),  # 다른 포트
    ],
)
def test_authorization_is_forwarded_only_when_the_redirect_stays_on_the_same_origin(new_url, forwarded):
    # Arrange / Act
    new = _redirected(new_url)

    # Assert
    assert new is not None
    assert new.get_header("Authorization") == (f"Bearer {TOKEN}" if forwarded else None)


class _Parent:
    """`http_error_302` 가 새 요청을 넘기는 opener 자리 — 요청을 받아 두기만 한다."""

    def __init__(self) -> None:
        self.opened: urllib.request.Request | None = None

    def open(self, req, timeout=None):
        self.opened = req


def _follow(location: str) -> urllib.request.Request | None:
    """Location 헤더가 `location` 인 302 응답을 urllib 의 실제 리다이렉트 처리(상대 URL 해석 포함)로 따라가 새 요청을 돌려준다."""
    req = urllib.request.Request(ORIGIN_URL)
    req.add_unredirected_header("Authorization", f"Bearer {TOKEN}")
    req.timeout = 1.0  # OpenerDirector.open 이 요청마다 다는 값 — 핸들러를 직접 부르므로 여기서 단다
    handler = fiw._KeepAuthOnSameOrigin()
    parent = _Parent()
    handler.add_parent(parent)
    headers = http.client.HTTPMessage()
    headers["Location"] = location
    handler.http_error_302(req, io.BytesIO(b""), 302, "Found", headers)
    return parent.opened


@pytest.mark.parametrize(
    ("location", "target", "forwarded"),
    [
        ("//cdn.example/x", "https://cdn.example/x", False),  # 스킴 상대: 현재 스킴으로 풀리지만 다른 호스트다
        ("//huggingface.co/api/x", "https://huggingface.co/api/x", True),  # 같은 호스트의 스킴 상대는 허브 안이다(대조군)
        ("/api/resolve-cache/models/x/abc/f", "https://huggingface.co/api/resolve-cache/models/x/abc/f", True),  # 허브 안 상대 경로(대조군)
        ("https://huggingface.co./x", "https://huggingface.co./x", False),  # 끝 점이 붙은 FQDN
    ],
)
def test_a_real_redirect_response_is_resolved_before_deciding_whether_the_token_follows(location, target, forwarded):
    # Arrange / Act
    new = _follow(location)

    # Assert — 상대 Location 은 urllib 이 요청 URL 로 풀어 준 뒤 origin 을 비교한다
    assert new is not None and new.full_url == target
    assert new.get_header("Authorization") == (f"Bearer {TOKEN}" if forwarded else None)


def test_forwarded_authorization_is_still_an_unredirected_header():
    # Arrange / Act — 다음 hop 에서도 같은 판정을 받으려면 일반 헤더가 아니라 unredirected 로 남아야 한다
    new = _redirected("https://huggingface.co/api/x")

    # Assert
    assert new is not None
    assert "Authorization" not in new.headers and new.unredirected_hdrs.get("Authorization") == f"Bearer {TOKEN}"


def test_a_request_without_a_token_is_never_given_one_on_redirect():
    # Arrange / Act
    new = _redirected("https://huggingface.co/api/x", with_token=False)

    # Assert
    assert new is not None and new.get_header("Authorization") is None


def test_token_does_not_return_after_leaving_the_origin():
    # Arrange — 허브 -> CDN(토큰 제거) -> 다시 허브: 두 번째 hop 의 요청에 토큰이 없으므로 붙일 것이 없다
    handler = fiw._KeepAuthOnSameOrigin()
    first = urllib.request.Request(ORIGIN_URL)
    first.add_unredirected_header("Authorization", f"Bearer {TOKEN}")

    # Act
    on_cdn = handler.redirect_request(first, io.BytesIO(b""), 302, "Found", {}, "https://cdn.example/blob")
    back = handler.redirect_request(on_cdn, io.BytesIO(b""), 302, "Found", {}, "https://huggingface.co/x")

    # Assert
    assert on_cdn.get_header("Authorization") is None and back.get_header("Authorization") is None


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co:notaport/x",  # 숫자가 아닌 포트
        "https://huggingface.co:99999/x",  # 범위 밖
        "https://huggingface.co:0/x",  # 0 은 기본 포트로 취급하면 안 된다
        "/relative",
        "",
        "ftp://huggingface.co/x",  # 기본 포트를 모르는 스킴
    ],
)
def test_origin_is_none_for_urls_that_cannot_be_compared(url):
    # Arrange / Act / Assert — 비교할 수 없으면 None 이고, 그러면 토큰을 붙이지 않는다
    assert fiw._origin(url) is None


def test_origin_fills_default_ports_and_lowercases():
    # Arrange / Act / Assert
    assert fiw._origin("https://HuggingFace.CO/a") == ("https", "huggingface.co", 443)
    assert fiw._origin("http://example.test/a") == ("http", "example.test", 80)
    assert fiw._origin("https://huggingface.co:443/a") == fiw._origin("https://huggingface.co/b")
    assert fiw._origin("https://huggingface.co:8443/a") == ("https", "huggingface.co", 8443)


def test_an_unreadable_redirect_target_does_not_get_the_token():
    # Arrange / Act
    new = _redirected("https://huggingface.co:notaport/x")

    # Assert
    assert new is not None and new.get_header("Authorization") is None


def test_two_unreadable_origins_are_not_treated_as_the_same_origin():
    # Arrange — 비교할 수 없는 둘(None, None)이 "같다"로 읽히면 토큰이 간다
    req = urllib.request.Request("https://huggingface.co:notaport/x")
    req.add_unredirected_header("Authorization", f"Bearer {TOKEN}")

    # Act
    new = fiw._KeepAuthOnSameOrigin().redirect_request(
        req, io.BytesIO(b""), 307, "Temporary Redirect", {}, "https://huggingface.co:notaport/y")

    # Assert
    assert new is not None and new.get_header("Authorization") is None


def test_harness_would_detect_a_leak_if_authorization_were_an_ordinary_header(hub):
    # Arrange — 대조군: 일반 헤더로 달면 같은 서버 구성에서 토큰이 CDN 에 간다(= 위 시험이 의미 있다는 증거)
    prefix = f"/{REPO}/resolve/main/inference/{VERSION}/{WHEEL_NAME}"
    req = urllib.request.Request(hub.base_url + prefix, headers={"Authorization": f"Bearer {TOKEN}"})

    # Act
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()

    # Assert
    cdn_calls = [h for role, _path, h in hub.requests if role == "cdn"]
    assert len(cdn_calls) == 1
    assert cdn_calls[0].get("authorization") == f"Bearer {TOKEN}"


def test_missing_file_on_the_hub_reports_404_without_the_response_body(hub, tmp_path):
    # Arrange — 존재하지 않는 버전
    # Act / Assert
    with pytest.raises(fiw.FetchError) as exc:
        fiw.fetch("9.9.9", PIN, tmp_path / "w", repo=REPO, token=TOKEN, base_url=hub.base_url, sleep=lambda _s: None)
    assert "HTTPError status=404" in str(exc.value)
    assert SECRET_BODY not in str(exc.value)
    assert hub.base_url not in str(exc.value)
