# e2-sidecar — OpenAQ / Sensor.Community 시간별 수집기 + 예보 수집 정시 dispatch (airlens-e2 VM)

GitHub Actions 밖에서 도는 유일한 수집 레인. Oracle E2 VM(`airlens-e2`,
Tailscale 접속, 주소는 private 런북)의 systemd 타이머가 매시 HF dataset
`Robeedau/airlens-live`의 `openaq-shadow/`·`sensor-community-shadow/`로 직접
발행한다. 2026-09-02까지 VM에만 존재하던(소스컨트롤 밖) 실물을 sha256 대조로
회수해 편입했다 — **이 디렉터리가 정본, VM은 배포 대상이다.** 수정은 여기서
하고 `./deploy.sh`로 내보낸다. VM에서 직접 고치지 않는다.

같은 VM의 타이머가 6시간마다 예보 수집 워크플로(`forecast-collect.yml`)를
`workflow_dispatch` API로 정시에 시작시킨다. 수집은 hosted 러너에서 돌고 이 VM은
방아쇠만 당긴다 — 아래 「forecast dispatch 설계 요지」.

## 구성

| 경로 (repo) | 배포 위치 (VM) | 역할 |
|---|---|---|
| `bin/run-global-shadow.sh` | `/usr/local/bin/` | 엔트리 — `<sc\|openaq>` 인자로 deno 수집 → `hf_publish.py` 업로드 → 7일 로컬 trim |
| `bin/run-shadow.sh` | `/usr/local/bin/` | **legacy** (AirKorea shadow, Wave 1 pivot 이전) — 타이머 없음, 참조용 보존 |
| `sidecar/openaq_shadow.ts` | `/opt/airlens/sidecar/` | OpenAQ pm25 수집 (deno, `--allow-net=api.openaq.org`) |
| `sidecar/sensor_community_shadow.ts` | `/opt/airlens/sidecar/` | Sensor.Community 수집 (`--allow-net=data.sensor.community`) |
| `sidecar/airkorea_shadow.ts` · `sidecar/openaq_backfill.ts` | `/opt/airlens/sidecar/` | legacy / one-off 백필 — 타이머 없음 |
| `systemd/airlens-sc-shadow.{service,timer}` | `/etc/systemd/system/` | 매시 **:05** |
| `systemd/airlens-openaq-shadow.{service,timer}` | `/etc/systemd/system/` | 매시 **:15** (1GB 박스라 메모리 피크 겹침 방지 스태거) |
| `sidecar/chatlog_pull.py` | `/opt/airlens/sidecar/` | Field Assistant 대화 로그 수신 — R2 `airlens-chatlog` → 로컬 SQLite (stdlib만, SigV4 자체 서명) |
| `systemd/airlens-chatlog-pull.{service,timer}` | `/etc/systemd/system/` | 매시 **:35** (세 번째 스태거 슬롯) |
| `bin/dispatch-forecast-collect.sh` | `/usr/local/bin/` | `forecast-collect.yml`을 `workflow_dispatch` API로 시작 (bash + curl). 거절되면 워크플로의 `state`를 읽어, 60일 무활동으로 꺼진 것(`disabled_inactivity`)만 켜고 한 번 재시도, 사람이 끈 것은 그대로 둔다 |
| `systemd/airlens-forecast-dispatch.{service,timer}` | `/etc/systemd/system/` | 6시간마다 **:50** (UTC 00:50·06:50·12:50·18:50). 시간별 슬롯(:05/:15/:35)과 겹치지 않는다 |

### chatlog_pull 설계 요지

생산자는 다른 레포다 — airlens-web의 `workers/assistant/src/persist.ts`가 엣지에서
개인정보를 마스킹한 턴 1건을 JSON 객체 1개로 R2에 떨어뜨린다. 이 스크립트가 그걸
가져와 VM의 SQLite 파일(0600)에 넣는다. 생산자 쪽 흐름은
[airlens-web `workers/assistant/`](https://github.com/AirLens-cloud/airlens-web/tree/main/workers/assistant)를 본다.

- **전송은 pull 전용** — 이 VM도 워커 오리진도 인바운드를 열지 않는다.
- **삭제는 행 확인 뒤에만.** 객체가 유일본이라, insert가 트랜잭션 밖에서 실패했는데
  delete만 성공하면 조용히 유실된다. insert → commit → 행 재조회 → 그 다음 delete.
- **성공 판정은 종료코드가 아니라 처리 건수.** 객체를 나열해놓고 하나도 저장하지
  못한 채 0으로 끝나는 실행은 `systemctl status`에서 성공처럼 보인다.
- **`PRAGMA secure_delete`는 파일 속성이 아니라 커넥션 설정**이라 매 실행 켠다.
  생성 시 한 번만 켜면 이후 실행엔 적용되지 않고, 그러면 "90일 뒤 파기"는 NULL
  컬럼에 대한 서술일 뿐 옛 텍스트는 해제된 페이지에 남는다. WAL도 쓰지 않는다
  (파기한 텍스트가 체크포인트 전까지 `-wal`에 남는다).
- 보존: 원문 90일 후 NULL + `VACUUM`, 메타데이터는 존치. R2 버퍼는 버킷
  라이프사이클 `expire-7d`(prefix `turn/`)로 자동 소멸 — 타이머가 며칠 죽어도 그
  안에 복구되면 무손실이고, 7일 초과분은 유실을 허용한다(로그이지 원장이 아니다).

### forecast dispatch 설계 요지

- **왜 VM이 부르나.** GitHub의 `schedule`은 몇 시간씩 늦게 오고, 예보를 쓰는
  `band-forecast.yml`은 `forecast-collect.yml`이 끝난 뒤 `workflow_run`으로 돌아서
  수집이 늦으면 밴드도 같이 늦는다. 그래서 VM 타이머가 정시에 시작시키고, 워크플로의
  `schedule`은 백업으로 남겼다.
- **중복 방지는 워크플로 쪽이다.** `forecast-collect.yml`의 첫 잡 `guard`
  (`scripts/etl/forecast_slot_guard.py`)가 같은 6시간 칸(UTC 00/06/12/18시 경계)에
  이미 수집 결과가 있으면 건너뛴다. 정시 dispatch가 먼저 수집했으면 뒤늦게 온 백업이
  건너뛰고, 판정을 못 하면 수집한다(fail-open). VM은 입력 없이 호출하므로 가드가
  적용된다. 같은 칸을 일부러 다시 수집하려면 수동 실행에서 `force`를 켠다.
- **왜 :50인가.** 시간별 수집기가 :05/:15/:35를 쓰니 겹치지 않는다. :50은 칸 경계
  50분 뒤이고 같은 칸의 :15 백업보다 35분 뒤다. 그 칸에 먼저 수집이 없으면 정시
  dispatch가 새 칸을 열고, 예보가 H:5x에 나오면 band가 다음 관측 슬롯
  (H+1시 16~19분)까지 약 25분만 기다린다. 백업(:15)이 그 칸의 :50 전에 먼저 수집하면
  (같은 칸 백업이 35분 미만 늦었거나, 앞 칸 백업이 5시간 45분~6시간 35분 늦어 이 칸
  앞부분에 떨어진 경우) 정시 dispatch는 건너뛴다. 이미 수집된 칸이라 규칙상 맞지만
  수집이 :50보다 이르니 band가 그만큼 더 기다린다(H+1시 마감 안).
- **이 타이머가 죽으면** 예보 수집은 `schedule`로만 돈다. 늦을 뿐 멈추지 않는다.
- **실패 처리.** 토큰이 없거나 모양이 틀리면 curl을 부르기 전에 실패한다. dispatch가
  거절되면 상태 코드와 문구에 상관없이 워크플로의 `state`를 읽는다(GET) — 꺼진 워크플로에
  dispatch했을 때의 응답 형식은 GitHub 문서에 없어서 코드나 문구에 기대지 않는다.
  GitHub가 60일 무활동으로 끈 것(`disabled_inactivity`)은 `schedule` 백업도 함께
  멈추므로 켜서 살리고 한 번 재시도한다. 사람이 일부러 끈 것(`disabled_manually`, 사고
  대응·롤백 절차)은 존중해서 켜지 않는다. 그 밖의 상태, 읽을 수 없는 값, 조회 실패도
  켜지 않고, 처음 거절의 HTTP 코드와 본문의 `message`에 읽은 상태(조회가 실패했으면 그
  코드)를 붙여 한 줄로 남기고 실패한다(유닛이 한 번 더 재시도한다).
- **로그.** 접수되면 journal에 `dispatched forecast-collect.yml on main (HTTP 200, run <id>)`
  한 줄이 남는다. dispatch API는 200과 함께 새 run의 ID를 돌려주므로(예전의 204는 본문이
  없다) 스크립트는 본문에서 그 숫자만 읽어 적고, 본문 원문은 싣지 않는다. 숫자를 못 읽어도
  접수된 것이면 ID 없이 성공으로 남긴다. 켠 뒤 재시도한 경우도 같다. 켜기 직전 줄에는
  처음 거절의 HTTP 코드와 `message`가 함께 남아, GitHub가 꺼진 워크플로에 실제로 주는
  응답을 journal에서 볼 수 있다.

## 운영 원칙

- **시크릿은 레포에 두지 않는다.** VM의 파일(root 소유, 0600)을 systemd `LoadCredential`로 주입하고,
  교체는 VM에서 파일만 바꾼다(oneshot 유닛이 슬롯마다 다시 읽으므로 재시작 불요).
- 채팅 로그용 R2 토큰은 **`airlens-chatlog` 버킷 한정 Object Read & Write** 스코프여야
  한다. 계정 스코프 토큰이면 이 VM이 같은 계정의 다른 버킷까지 갖게 된다.
- dispatch용 GitHub 토큰(`/etc/airlens/gh_dispatch_pat`, root 소유 0600, 토큰 한 줄)은
  **fine-grained PAT, 이 레포(`airlens-data`) 하나, Actions 읽기·쓰기 + Metadata 읽기**,
  만료 1년 이하여야 한다. 클래식 PAT나 조직 전체 범위면 이 VM이 다른 레포까지 갖게 된다.
  Actions 권한은 읽기/쓰기 두 단계뿐이라, 이 토큰을 가진 쪽은 이 레포의 모든 워크플로를
  시작·끄기·취소할 수 있다(시크릿 값은 읽지 못한다). 그래서 토큰 파일 권한을 지키고,
  사람 계정이 아닌 이 용도 전용 계정으로 발급하기를 권한다.
  만료되면 dispatch가 401로 실패하고 `schedule` 백업만 남으므로 만료일 전에 교체한다
  (VM의 파일만 바꾸면 된다 — oneshot 유닛이 실행마다 다시 읽는다).
- 토큰 파일이 없으면 `deploy.sh`는 dispatch 타이머를 켜지 않는다. 토큰을 둔 뒤 `deploy.sh`를
  다시 돌리거나, VM에서 `sudo systemctl enable --now airlens-forecast-dispatch.timer`를 실행한다.
- dispatch 스크립트는 토큰을 명령줄·환경 변수·로그에 두지 않는다. curl 설정을 stdin(`curl -q -K -`)으로
  넘기고(`-q`는 `~/.curlrc`를 읽지 않게 하는 첫 인자다) `set -x`로 돌리지 않는다.
- dispatch 유닛은 `DynamicUser=yes`로 돈다(`ubuntu` 계정이 아니다). 실행하는 동안 토큰은 이 유닛의
  credential 디렉터리에 있고 그 디렉터리는 유닛의 UID(와 root)만 읽을 수 있어서, 같은 `ubuntu`로
  도는 시간별 수집기가 그 창에 읽지 못하게 한 것이다. 코어 파일도 남기지 않는다(`LimitCORE=0`).
- `deploy.sh`가 이 레포의 `scripts/etl/hf_publish.py`와 `contracts/`를 VM에 같은 상대
  구조로 설치한다. 운영 타이머(`run-global-shadow.sh`)는 레포 클론이 필요 없다.
- 시크릿 파일 형식, VM 쪽 준비 사항과 레거시 정리 대상, 접속 값은 비공개 런북에 있다.

## 접속 / 배포 / 검증

```bash
# 접속 (Tailscale SSH) — E2_HOST·E2_SSH_KEY 값은 비공개 런북에 있다
ssh -i "$E2_SSH_KEY" "$E2_HOST"

# 배포 (멱등)
./deploy.sh

# 검증 1 — 타이머 살아있나
ssh -i "$E2_SSH_KEY" "$E2_HOST" 'systemctl list-timers airlens-*'

# 검증 2 — 실제 발행됐나 (green≠working: 산출물로 본다)
# 다음 정시 슬롯 후 HF openaq-shadow/ 최신 파일 타임스탬프 확인.
# 상시 감시는 .github/workflows/e2-shadow-freshness.yml (hosted, 6h) 담당.

# 검증 3 — forecast dispatch 타이머와 마지막 실행 로그 (VM)
# 성공 줄 `dispatched … (HTTP 200, run <id>)` 의 id 로 그 run 을 바로 열 수 있다:
#   gh run view <id> -R AirLens-cloud/airlens-data
ssh -i "$E2_SSH_KEY" "$E2_HOST" \
  'systemctl list-timers airlens-forecast-dispatch.timer --no-pager; journalctl -u airlens-forecast-dispatch.service -n 20 --no-pager'

# 검증 4 — forecast-collect 가 workflow_dispatch 로 정시에 시작됐나 (hosted 쪽에서 본다)
# event 가 workflow_dispatch 인 run 의 createdAt 이 UTC 00:50·06:50·12:50·18:50 근처여야 한다.
gh run list -R AirLens-cloud/airlens-data -w forecast-collect.yml --limit 8 \
  --json event,createdAt,status,conclusion --jq '.[] | [.event, .createdAt, .status, .conclusion] | @tsv'
# guard 가 건너뛴 run 은 success 로 끝나지만 collect-forecast 잡이 skipped 로 보인다(같은 칸에 이미 수집됨).
# 예보가 실제로 갱신됐는지는 run 이 아니라 산출물로 본다: aq-data/forecast.json 의 generated_at.
curl -sL https://huggingface.co/datasets/Robeedau/airlens-live/resolve/main/aq-data/forecast.json \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["generated_at"])'
```

## 감시 독립성

freshness probe(`e2-shadow-freshness.yml`)는 **hosted 러너에 남긴다** — E2가
죽으면 E2 위의 무엇도 그 사실을 보고할 수 없기 때문. 이 원칙은 이 VM으로
잡을 더 옮길 때도 유지한다. dispatch도 같다 — 이 VM이 죽으면 dispatch는 멈추지만
워크플로의 `schedule` 백업은 hosted에 남아 있어 예보 수집이 늦게나마 돈다.
예보 수집이 정시로 시작됐는지 보는 쪽(위 검증 4)도 VM 밖에서 한다.
