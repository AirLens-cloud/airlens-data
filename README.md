# airlens-data

AirLens 데이터 수집 파이프라인 — 스케줄 수집기가 외부 공개 API에서 데이터를 모아
Hugging Face 데이터셋 [`Robeedau/airlens-live`](https://huggingface.co/datasets/Robeedau/airlens-live)로 발행한다.
(2026-08-26 비공개 ML 모노레포에서 분리. 2026-10-10 public 전환 — 모든 스케줄 잡이 GitHub hosted 러너(`ubuntu-latest`)에서 돈다. 이전 이력은 비공개 보관 레포에 보존.)

## 워크플로

| 워크플로 | 주기 | 수집원 → HF 발행 경로 |
|---|---|---|
| `data-collect-hourly` | 3h | Open-Meteo 날씨·해양·가스(O3/NO2/CO) + NOAA GEFS PM2.5/PM10 + GFS 바람 → `aq-data/`, `wind-data/` |
| `timeline-collect` | 3h | NOAA GEFS PM2.5 ±24h 타임라인 → `aq-data/timeline/` |
| `forecast-collect` | 6h | Open-Meteo CAMS 도시별 PM2.5 예보 → `aq-data/forecast.json` (도시 시드 = `seed/cams_forecast.json`) |
| `firms-collect` | 6h | NASA FIRMS 화재 핫스팟 → `wind-data/active-fires.json` |
| `news-collect` | 6h | RSS 뉴스 → `news-data/articles.json` |
| `mac-data-publish` | 1h | CAMS + GEFS-chem + AirKorea + EEA → `mac-data/data/{mac,web}/v1/` |
| `hf-live-squash` | 주간(일 19:20 UTC) | HF repo 히스토리 squash (용량 관리) |
| `openaq-bulk-collect` | 일간(01:45 UTC) | OpenAQ 파라미터별 벌크 수집 → `insights-data/openaq-bulk/{YYYY-MM}/` (+ `manifest.json`) |
| `policy-collect` | 주간(일 04:00 UTC) | 대기질 정책 레지스트리 → `insights-data/policy/` |
| `contracts-publish` | 3h | 산출물 계약 manifest/registry/health 생성 → `meta/` (`product_manifest.json`·`source_registry.json`·`product_health.json`) |
| `ml-predict` | 3h(:45) | 비공개 추론 wheel 로 PM2.5 격자 예측 + DQSS → `aq-data/predictions/` (`grid_latest.json`·`history/`), `aq-data/data_quality.json`, `aq-data/quality-history/`. 후보 모드면 `aq-data/_candidate/` (`PUBLISH_MODE`) |
| `band-forecast` | 예보 수집 직후 + 6h(:50) 백업 | 비공개 추론 wheel 로 도시별 PM2.5 예측 밴드(p10·p50·p90) → `aq-data/forecast-band/latest.json`. 후보 모드면 `aq-data/_candidate/forecast-band/` (`PUBLISH_MODE`) |
| `shadow-retention` | 일간(02:40 UTC) | E2 사이드카 shadow 슬롯 14일 초과분 정리 → `openaq-shadow/`, `sensor-community-shadow/` |
| `candidate-prune` | 수동 | 컷오버 뒤 후보 파일 정리(기본 dry-run, 그 제품 워크플로가 production 일 때만) → `aq-data/_candidate/` |
| `mac-snapshot-healthcheck` | 1h(:35) | 발행된 mac 스냅샷 신선도 프로브 (발행 없음, advisory) |
| `predictions-freshness-healthcheck` | 2h(:17) | `grid_latest.json` 나이(8h)·history 슬롯 결손·예측 밴드 나이(12h)·`ml-predict`·`band-forecast` 활성 상태 프로브, 경보 이슈를 열고 닫는다 (발행 없음) |
| `e2-shadow-freshness` | 6h(:25) | `openaq-shadow/`·`sensor-community-shadow/` 최신 슬롯 프로브 (발행 없음) |
| `keepalive` | 월 2회(1·15일) | public 레포 60일 비활동 자동 비활성화 방지 (워크플로 재활성화, 발행 없음) |
| `pr-check` | PR/push | 테스트·린트 (발행 없음) |

ETL 스크립트는 `scripts/etl/` (모노레포 경로 보존 — 워크플로 수정 최소화).
HF 초기 시드(hf-live-seed)는 모노레포 정적 데이터 의존이라 이관하지 않음(이미 시드 완료).

## 문서

| 문서 | 내용 |
|---|---|
| [`DATA_SOURCES.md`](DATA_SOURCES.md) | 외부 데이터 출처·라이선스·표기 |
| [`contracts/README.md`](contracts/README.md) | 발행물 계약(스키마) 색인, `validate.py` 사용법, DQSS 정의 |
| [`contracts/EVIDENCE_CONTRACT.md`](contracts/EVIDENCE_CONTRACT.md) | 사용자에게 보이는 값이 지켜야 할 증거 규칙 |
| [`e2-sidecar/README.md`](e2-sidecar/README.md) | E2 사이드카(OpenAQ·Sensor.Community 시간별 수집, 채팅 로그 pull, 예보 수집 정시 dispatch) |
| [`scripts/etl/collect_policies/SOURCES.md`](scripts/etl/collect_policies/SOURCES.md) | 정책 레지스트리 수집원 |

## Secrets

### environment `hf-write`

| 이름 | 용도 |
|---|---|
| `HF_TOKEN` | HF `Robeedau/airlens-live` write (HF에 쓰는 워크플로 전부) |

- HF에 쓰는 잡은 `environment: {name: hf-write, deployment: false}` 를 선언하고, `scripts/etl/test_publish_install_lock.py` 가 그 선언을 확인한다. 배포 브랜치 규칙과 secret 위치는 GitHub 설정이라 이 레포의 시험이 확인하지 못한다.
- 이 environment 는 쓰기 토큰이 있는 잡만 쓴다 — 토큰 없는 잡을 넣지 않는다(시험이 확인한다).
- 보호는 두 설정이 함께 있을 때만 성립한다: (가) 이 environment 의 배포 브랜치 규칙이 `main` 하나, (나) 레포 secret 에 `HF_TOKEN` 이 없음. 그러면 main 이 아닌 ref 에서 돈 잡은 쓰기 토큰을 받지 못한다. 레포 secret 에 같은 이름을 다시 두면 environment 를 선언하지 않은 잡과 다른 브랜치가 그 값을 받는다.
- (가)(나)가 갖춰지면 이 잡들은 main 에서만 수동 실행된다. 다른 브랜치에서 dispatch 하면 environment 규칙에서 거부된다.
- `deployment: false` 라 실행마다 배포 기록을 만들지 않는다.
- 없는 environment 를 참조하면 GitHub 가 규칙 없는 environment 를 새로 만든다. 이름을 바꿀 때는 새 environment(규칙 포함)를 먼저 만든 뒤 워크플로를 고치고, 지울 때는 워크플로에서 참조를 먼저 뺀다.
- GitHub Free 에서는 environment secret 이 공개 레포에서만 동작한다. 레포를 비공개로 바꾸면 이 잡들이 토큰을 받지 못한다.
- 등록: Settings → Environments → `hf-write` → Environment secrets

### environment `hf-read`

| 이름 | 용도 |
|---|---|
| `HF_MODELS_READ` | 비공개 HF 모델 저장소 read (ml-predict·band-forecast 의 `infer` 잡, wheel 받는 스텝) |
| `HF_MODELS_REPO` | 그 모델 저장소 id |

- 이 두 값을 쓰는 잡은 `environment: {name: hf-read, deployment: false}` 를 선언하고, 토큰 없는 잡은 넣지 않는다. `scripts/ml/test_ml_workflow_safety.py` 가 둘 다 확인한다.
- 보호 조건은 `hf-write` 와 같다: (가) 배포 브랜치 규칙이 `main` 하나, (나) 레포 secret 에 같은 이름이 없음.
- 등록: Settings → Environments → `hf-read` → Environment secrets

### repo-level

등록: Settings → Secrets and variables → Actions → New repository secret

| 이름 | 용도 |
|---|---|
| `NASA_FIRMS_MAP_KEY` | firms-collect |
| `CDS_API_KEY` | mac-data-publish (Copernicus ADS — CAMS. `CDS_API_URL`은 워크플로에 하드코딩) |
| `AIRKOREA_API_KEY` | mac-data-publish (공공데이터포털 에어코리아) |
| `OPENAQ_API_KEY` | openaq-bulk-collect |
| `OPENAI_API_KEY` | policy-collect — 선택, 수동 실행에서 `run_llm_extract`를 켤 때만 |

## 로컬 훅

```bash
brew install gitleaks
git config core.hooksPath scripts/git-hooks
```
