#!/usr/bin/env python3
"""hf_publish — AirLens 핫 데이터의 Hugging Face Hub 발행 공용 유틸.

Supabase Free Storage 용량 402 차단(2026-08-20)을 계기로, 파이프라인 산출물의
1차 저장소를 Supabase Storage → HF 공개 dataset repo 로 이전한다
(plan: supabase-polymorphic-abelson). repo 내 경로는 기존 버킷 레이아웃을
그대로 미러링한다 (aq-data/... , wind-data/... , app-models/...) — 읽기 URL 이
`/storage/v1/object/public/<bucket>/<path>` → `resolve/main/<bucket>/<path>` 로
기계적으로 치환되도록.

이 파일이 유일본이다 (2026-10 조직 재편). 모노레포 `AirLens-cloud/airlens-ml`(2026-10-21까지 `AirLens`) 에
있던 사본은 추론 job 이 이 레포로 옮겨온 뒤 삭제됐다(AirLens#1481, 2026-10-21) — 고칠 곳은 여기 하나다.

Subcommands:
  ensure-repo                          repo 생성(존재 시 no-op) + dataset card
  upload --src PATH --dest REPO_PATH   파일 또는 디렉터리 업로드 (upsert)
         [--schema NAME|PATH|auto]     업로드 *직전* 계약 검증. auto = 페이로드의
                                       schema_version(계약 이름)으로 해석
  batch  --map SRC=DEST [--map ...]    여러 파일을 **커밋 1개**로 업로드 — 실행당
         [--schema NAME|PATH|auto]     커밋 수를 줄여 HF 커밋 rate limit 회피.
                                       계약 검증은 원격 호출 전에 전부 끝낸다
  prune  --prefix P --keep-days N      P 아래 <name>-YYYYMMDDHH.json[.gz] 형식
                                       파일 중 N일 이전分 삭제 (timeline +
                                       shadow retention 공용). `--dry-run` 이면
                                       삭제 대상만 출력하고 실제 삭제는 하지
                                       않는다. `openaq-backfill-kr-*` 등
                                       backfill 산출물은 파일명에 타임스탬프가
                                       있어도 항상 보존한다 (일회성 백필 —
                                       재수집 불가, DQSS W5 사고 방지).
  squash                               git 히스토리를 1커밋으로 압축 — 시간별
                                       upsert 커밋 누적 방지 (HF 공식 권장,
                                       비가역이나 현재 트리는 보존됨)

Environment:
  HF_TOKEN      write token (필수 — ensure-repo/upload/prune 공통)
  HF_LIVE_REPO  대상 repo (기본 Robeedau/airlens-live, 공개 dataset)

원격 호출 재시도: 429·5xx·연결 끊김·타임아웃은 최대 5회, `min(60, 2^n)` 초 +
jitter 로 재시도한다 (`Retry-After` 헤더가 있으면 그 값을 따른다). 끝내 실패하면
exit 75 (EX_TEMPFAIL) — "일시 장애" 와 계약 위반 등 영구 실패(exit 1)를 구분한다.
다음 실행이 최신본을 다시 올리므로 따로 큐를 두지 않는다.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

REPO_ID = os.environ.get("HF_LIVE_REPO", "Robeedau/airlens-live")
REPO_TYPE = "dataset"

# timeline 프레임(pm25-YYYYMMDDHH.json) + shadow 슬롯(openaq-YYYYMMDDHH.json.gz,
# sensor-community-YYYYMMDDHH.json.gz) 공용 타임스탬프 규칙. 구분자만 다르고
# (하이픈/언더스코어) 접미 10자리 시각 + 압축 유무만 다른 동일 규약이라
# 접두사·확장자 무관 단일 정규식으로 모두 매칭. 예측 히스토리 스냅샷
# (`grid_YYYYMMDDHH.json`)도 같은 규약이다. group(1) 은 변경 없음.
TS_NAME_RE = re.compile(r"[-_](\d{10})\.json(?:\.gz)?$")

# `openaq-backfill-kr-YYYYMMDD-YYYYMMDD.json.gz` 형 백필 파일(원본 =
# e2-sidecar/sidecar/openaq_backfill.ts) — 날짜 구간이 8자리라 TS_NAME_RE 의
# 10자리 요건과 우연히도 안 겹치지만, 명명 규칙이 바뀌어도 오삭제가 나지
# 않도록 파일명에 "backfill" 이 있으면 prune 대상에서 항상 제외한다
# (방어적 이중 안전장치).
BACKFILL_MARKER = "backfill"

RETRY_ATTEMPTS = 5
RETRY_MAX_DELAY = 60.0  # Retry-After 도 이 값으로 자른다 — VM 유닛 TimeoutStartSec=10min
RETRY_STATUS = {429, 500, 502, 503, 504}
# requests(huggingface_hub 0.x)와 httpx(1.x)의 네트워크 예외 이름. 클래스를 직접
# import 하면 어느 한쪽 버전에서 ImportError 가 나므로 이름으로 판정한다. MRO 를
# 훑으므로 httpx 의 기반 클래스(TimeoutException/NetworkError/ProtocolError)가 하위
# 예외(WriteTimeout, CloseError 등)를 함께 덮는다.
TRANSIENT_EXC_NAMES = {
    # requests
    "ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout",
    # httpx
    "TimeoutException", "NetworkError", "ProtocolError", "TransportError",
}
EX_TEMPFAIL = 75
_sleep = time.sleep  # 테스트에서 교체


def _is_transient(exc: BaseException) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return status in RETRY_STATUS
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return any(c.__name__ in TRANSIENT_EXC_NAMES for c in type(exc).__mro__)


def _retry_after(exc: BaseException) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        value = float(headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None  # 없음, 또는 HTTP-date 형식 — 지수 백오프로 대체
    if not math.isfinite(value) or value < 0:
        return None
    return min(value, RETRY_MAX_DELAY)


def _call(what: str, fn: Callable[[], object]) -> object:
    """원격 호출 1건을 일시 장애에 한해 재시도한다. 영구 오류는 그대로 올린다."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:  # 판정 후 영구 오류는 재-raise
            if not _is_transient(exc):
                raise
            if attempt == RETRY_ATTEMPTS:
                print(
                    f"::error title=HF transient::{what} — {RETRY_ATTEMPTS}회 시도 실패: {exc}",
                    file=sys.stderr,
                )
                sys.exit(EX_TEMPFAIL)
            delay = _retry_after(exc)
            if delay is None:
                delay = min(RETRY_MAX_DELAY, 2.0 ** attempt) + random.uniform(0, 1)
            print(f"retry {attempt}/{RETRY_ATTEMPTS - 1}: {what} — {exc} (sleep {delay:.1f}s)",
                  file=sys.stderr)
            _sleep(delay)
    raise AssertionError("unreachable")

DATASET_CARD = """\
---
license: other
license_name: airlens-live-mixed-sources
license_link: https://github.com/AirLens-cloud/airlens-data/blob/main/DATA_SOURCES.md
pretty_name: AirLens Live Air-Quality Data
tags:
  - air-quality
  - pm25
  - environment
  - south-korea
---

# AirLens Live Data

Live data layer for [AirLens](https://github.com/AirLens-cloud/airlens-data), an open
air-quality monitoring platform. Updated by scheduled GitHub Actions pipelines.

Layout mirrors the former Supabase Storage buckets:

| Path | Content | Cadence |
|---|---|---|
| `aq-data/current-*-grid.json` | Global pollutant grids (PM2.5/PM10/O3/NO2/CO) | hourly |
| `aq-data/timeline/` | GEFS-Aerosols PM2.5 frames, -24h..+24h, 3h step | every 3h |
| `aq-data/predictions/grid_latest.json` | AOD→PM2.5 model predictions (p10-p90 + DQSS) | every 3h |
| `aq-data/predictions/history/` | Timestamped prediction snapshots (30-day retention) | every 3h |
| `aq-data/forecast-archive/` | CAMS/AIFS forecast snapshots (30-day retention) | 6-hourly |
| `wind-data/` | Weather / marine grids | hourly |
| `news-data/articles.json` | RSS-collected environment/air-quality news (merged, article_url-deduped) | every 6h |
| `app-models/aod/` | AOD→PM2.5 model artifacts (sha256-attested) | per release |

Sources: NOAA GEFS-Aerosols, Open-Meteo, AirKorea, CAMS and others. Each source's
license, required attribution and terms link are listed in
[DATA_SOURCES.md](https://github.com/AirLens-cloud/airlens-data/blob/main/DATA_SOURCES.md).
ML outputs always carry
p10-p90 uncertainty bands and a DQSS quality grade.
"""


def _api() -> HfApi:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        sys.exit(1)
    return HfApi(token=token)


def ensure_repo(api: HfApi) -> None:
    _call("create_repo", lambda: api.create_repo(
        repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True, private=False))
    existing = set(_call("list_repo_files", lambda: api.list_repo_files(
        repo_id=REPO_ID, repo_type=REPO_TYPE)))
    if "README.md" not in existing:
        _call("upload dataset card", lambda: api.upload_file(
            path_or_fileobj=DATASET_CARD.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message="Add dataset card",
        ))
        print(f"dataset card created -> {REPO_ID}")
    print(f"repo ready: hf.co/datasets/{REPO_ID}")


@functools.cache
def _validator():
    """contracts/validate.py 를 파일 경로로 한 번만 적재한다 (batch 에서 파일마다 재실행 방지)."""
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_airlens_contract_validate", root / "contracts" / "validate.py"
    )
    if spec is None or spec.loader is None:
        print(f"::error title=Contract::검증기를 찾을 수 없다: {root}/contracts/validate.py", file=sys.stderr)
        sys.exit(1)
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    return v


def check_contract(src: str, schema: str, targets_glob: str = "*.json") -> None:
    """발행 **직전** 계약 검증. 위반이면 업로드하지 않고 죽는다.

    `targets_glob` 은 한 디렉터리가 **여러 제품**을 담을 때만 쓴다 (web/v1 이
    격자 2개 + `health.json` 을 함께 낸다). 좁히면 나머지가 조용히 무검증이 되므로
    건너뛴 파일을 로그에 이름으로 남긴다 — 부분 검증을 전체 검증으로 위장하지 않는다.

    검증 지점이 여기인 이유: 생성 직후가 아니라 업로드 직전이어야 "만들 때는
    맞았는데 올라간 건 다른 것" 이 불가능해진다. 검증기(`contracts/validate.py`)
    는 미지원 키워드를 통과가 아니라 실패로 다루므로, 스키마에 오타를 내면
    조용히 무검증이 되는 대신 여기서 죽는다.
    """
    # sys.path 를 건드리지 않고 파일 경로로 직접 적재한다. `sys.path.insert` 로
    # contracts/ 를 전역 경로에 얹으면 이후 어디서든 `import validate` 가 이
    # 파일로 해석되어, 같은 이름의 서드파티 모듈과 조용히 충돌할 수 있다
    # (council 리뷰 지적). 검증기는 여기서만 쓰므로 전역에 남길 이유가 없다.
    #
    # `--schema auto` 는 파일마다 페이로드의 `schema_version`(계약 이름, ML 산출물
    # 관례)으로 계약을 고른다. 수집 산출물의 `schemaVersion` 은 "1.0" 같은 숫자
    # 버전이라 계약 이름이 아니다 — 그쪽은 호출부가 이름을 명시한다.
    v = _validator()
    ContractError, load_schema, validate_payload = v.ContractError, v.load_schema, v.validate_payload
    resolve_schema = v.resolve_schema

    path = Path(src)
    if path.is_dir():
        targets = sorted(path.rglob(targets_glob))
        skipped = sorted(set(path.rglob("*.json")) - set(targets))
    else:
        targets, skipped = [path], []
    if not targets:
        # 글롭이 아무것도 못 맞히면 "위반 0건" 이 아니라 검증이 안 돌아간 것이다.
        print(
            f"::error title=Contract::{src} — '{targets_glob}' 에 맞는 JSON 이 없다", file=sys.stderr
        )
        sys.exit(1)
    if skipped:
        print(f"contract skip ({len(skipped)}): {', '.join(t.name for t in skipped)} — {schema} 대상 아님")
    auto = schema == "auto"
    sch: dict | None = None
    if not auto:
        try:
            sch = load_schema(schema)
        except ContractError as e:
            print(f"::error title=Contract schema::{schema} — {e}", file=sys.stderr)
            sys.exit(1)
    for t in targets:
        try:
            payload = json.loads(t.read_text(encoding="utf-8"))
            validate_payload(payload, resolve_schema(payload) if auto else sch)
        except ContractError as e:
            print(
                f"::error title=Contract violation::{t.name} vs {schema} — {e}. "
                f"발행 중단 (옛 버전이 그대로 남는다)",
                file=sys.stderr,
            )
            sys.exit(1)
        except (OSError, json.JSONDecodeError) as e:
            print(f"::error title=Contract::{t.name} 읽기 실패 — {e}", file=sys.stderr)
            sys.exit(1)
    print(f"contract ok: {len(targets)} file(s) vs {schema}")


def upload(api: HfApi, src: str, dest: str, message: str | None) -> None:
    path = Path(src)
    if not path.exists():
        print(f"ERROR: source not found: {src}", file=sys.stderr)
        sys.exit(1)
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
        if not files:
            print(f"ERROR: empty source dir: {src}", file=sys.stderr)
            sys.exit(1)
        _call(f"upload_folder {dest}", lambda: api.upload_folder(
            folder_path=str(path),
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            path_in_repo=dest,
            commit_message=message or f"Update {dest} ({len(files)} files)",
        ))
        print(f"uploaded dir {src} ({len(files)} files) -> {REPO_ID}/{dest}")
    else:
        if path.stat().st_size == 0:
            # 빈 파일이 정상본을 덮어쓰는 사고 방지 (guard_no_empty_replace 정신)
            print(f"ERROR: refusing to upload empty file: {src}", file=sys.stderr)
            sys.exit(1)
        _call(f"upload_file {dest}", lambda: api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=dest,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=message or f"Update {dest}",
        ))
        print(f"uploaded {src} -> {REPO_ID}/{dest}")


def _parse_map(items: list[str]) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for item in items:
        src, sep, dest = item.partition("=")
        if not sep or not src or not dest:
            print(f"ERROR: --map 은 SRC=DEST 형식이어야 한다: {item!r}", file=sys.stderr)
            sys.exit(1)
        path = Path(src)
        if not path.is_file():
            print(f"ERROR: batch 는 파일만 받는다 (없거나 디렉터리): {src}", file=sys.stderr)
            sys.exit(1)
        if path.stat().st_size == 0:
            print(f"ERROR: refusing to upload empty file: {src}", file=sys.stderr)
            sys.exit(1)
        pairs.append((path, dest))
    dests = [d for _, d in pairs]
    if len(set(dests)) != len(dests):
        print("ERROR: batch 안에 같은 DEST 가 두 번 이상 있다", file=sys.stderr)
        sys.exit(1)
    return pairs


def _batch_ops(pairs: list[tuple[Path, str]]) -> list[CommitOperationAdd]:
    """원격 호출 전에 커밋 연산을 만든다 — 잘못된 DEST(`..` 등)는 여기서 거부된다."""
    try:
        return [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(src)) for src, dest in pairs]
    except ValueError as e:
        print(f"ERROR: batch DEST 가 올바르지 않다 — {e}", file=sys.stderr)
        sys.exit(1)


def batch(api: HfApi, ops: list[CommitOperationAdd], message: str | None) -> None:
    """여러 파일을 커밋 1개로 올린다. 입력·계약 검증은 호출 전에 끝나 있어야 한다."""
    _call(f"batch commit ({len(ops)} files)", lambda: api.create_commit(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        operations=ops,
        commit_message=message or f"Update {len(ops)} files",
    ))
    for op in ops:
        print(f"uploaded {op.path_or_fileobj} -> {REPO_ID}/{op.path_in_repo}")


def _stale_files(files: list[str], prefix: str, cutoff: datetime) -> tuple[list[str], int, int]:
    stale: list[str] = []
    skipped_bad_ts = 0
    skipped_backfill = 0
    for f in files:
        if not f.startswith(prefix):
            continue
        if BACKFILL_MARKER in Path(f).name.lower():
            skipped_backfill += 1
            continue  # openaq-backfill-kr-*.json.gz 등 — 절대 자동 삭제하지 않는다
        m = TS_NAME_RE.search(f)
        if not m:
            continue  # manifest.json 등 타임스탬프 없는 파일은 건드리지 않는다
        try:
            ts = datetime.strptime(m.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)
        except ValueError:
            # 일반화된 정규식([-_]\d{10}\.json(\.gz)?$)이 10자리 숫자를 매치했지만
            # 유효한 YYYYMMDDHH 가 아닌 경우(월/일/시 범위 밖) — 삭제 대상 오판 방지.
            skipped_bad_ts += 1
            continue
        if ts < cutoff:
            stale.append(f)
    return stale, skipped_bad_ts, skipped_backfill


def prune(api: HfApi, prefix: str, keep_days: int, dry_run: bool = False) -> None:
    if not prefix.endswith("/"):
        prefix += "/"
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    files = list(_call("list_repo_files", lambda: api.list_repo_files(
        repo_id=REPO_ID, repo_type=REPO_TYPE)))
    stale, skipped_bad_ts, skipped_backfill = _stale_files(files, prefix, cutoff)
    if skipped_bad_ts:
        print(f"prune: {skipped_bad_ts}개 파일이 10자리 숫자를 포함하나 유효 시각이 아님 — skip")
    if skipped_backfill:
        print(f"prune: backfill 파일 {skipped_backfill}개 — 항상 보존, skip")
    if not stale:
        print(f"prune: nothing older than {keep_days}d under {prefix}")
        return
    if dry_run:
        print(f"prune --dry-run: would delete {len(stale)} files under {prefix} (kept <= {keep_days}d)")
        for f in stale:
            print(f"  - {f}")
        return
    # 재시도 주의: 첫 시도가 커밋됐는데 응답만 잃으면(504/timeout) 재시도는 이미 지운
    # 경로를 다시 지우려다 4xx 로 끝난다 — exit 1 이지만 데이터 손상은 없고, 다음
    # 실행이 목록을 새로 읽는다.
    _call(f"prune {prefix}", lambda: api.create_commit(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        operations=[CommitOperationDelete(path_in_repo=f) for f in stale],
        commit_message=f"Prune {len(stale)} files >{keep_days}d under {prefix}",
    ))
    print(f"pruned {len(stale)} files under {prefix} (kept <= {keep_days}d)")


def squash(api: HfApi) -> None:
    _call("super_squash_history", lambda: api.super_squash_history(
        repo_id=REPO_ID, repo_type=REPO_TYPE))
    print(f"history squashed to 1 commit: {REPO_ID} (quota 반영은 최대 36h 소요)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ensure-repo")
    p_up = sub.add_parser("upload")
    p_up.add_argument("--src", required=True)
    p_up.add_argument("--dest", required=True)
    p_up.add_argument("--message", default=None)
    p_up.add_argument(
        "--schema",
        default=None,
        help="계약 이름(예: current-aq-grid.v1), 스키마 파일 경로, 또는 auto(페이로드의 "
             "schema_version 으로 해석 — ML 산출물용). 주면 업로드 직전에 검증하고, "
             "위반이면 올리지 않고 죽는다. 계약 대상이 아닌 산출물에는 붙이지 않는다.",
    )
    p_up.add_argument(
        "--schema-targets",
        default="*.json",
        help="디렉터리 안에서 이 계약의 대상만 좁히는 글롭(기본 *.json). 한 디렉터리가 "
             "여러 제품을 담을 때만 쓴다. 건너뛴 파일은 로그에 이름으로 남는다.",
    )
    p_b = sub.add_parser("batch")
    p_b.add_argument("--map", action="append", required=True, metavar="SRC=DEST",
                     help="올릴 파일과 repo 경로. 여러 번 줄 수 있다")
    p_b.add_argument("--message", default=None)
    p_b.add_argument("--schema", default=None,
                     help="모든 파일에 적용할 계약 이름/경로, 또는 auto(파일별 자기신고)")
    p_pr = sub.add_parser("prune")
    p_pr.add_argument("--prefix", required=True)
    p_pr.add_argument("--keep-days", type=int, default=30)
    p_pr.add_argument(
        "--dry-run",
        action="store_true",
        help="삭제 대상만 출력하고 실제로는 삭제하지 않는다",
    )
    sub.add_parser("squash")
    args = parser.parse_args()

    api = _api()
    if args.cmd == "ensure-repo":
        ensure_repo(api)
    elif args.cmd == "upload":
        # 계약 검증이 먼저다 — ensure_repo/업로드보다 앞. 위반이면 원격을 건드리기
        # 전에 죽어야 옛 버전이 온전히 남는다.
        if args.schema:
            check_contract(args.src, args.schema, args.schema_targets)
        ensure_repo(api)
        upload(api, args.src, args.dest, args.message)
    elif args.cmd == "batch":
        # 원격을 건드리기 전에 입력과 계약을 전부 검증한다 — 하나라도 위반이면
        # 아무것도 올리지 않는다 (커밋 1개 = 전부 아니면 전무).
        pairs = _parse_map(args.map)
        if args.schema:
            for src, _ in pairs:
                check_contract(str(src), args.schema)
        ops = _batch_ops(pairs)
        ensure_repo(api)
        batch(api, ops, args.message)
    elif args.cmd == "prune":
        prune(api, args.prefix, args.keep_days, dry_run=args.dry_run)
    elif args.cmd == "squash":
        squash(api)


if __name__ == "__main__":
    main()
