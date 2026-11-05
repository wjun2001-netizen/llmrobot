# llmrobot

현재 연구 실행 backend는 **로컬 Hugging Face causal LLM**입니다.
[로컬 NVIDIA 설치·실행 및 entropy 정의](docs/LOCAL_LLM.md)를 먼저 참고하세요.
PDDL 본문은 `collect_pddl.py`에 원본 그대로 보존했고, `collect_normal.py`는 기본 `pddl`을 유지하는 backend 라우터입니다.
OpenAI/Gemini 코드와 기존 로그는 변경하지 않았습니다. 아래 기존 API/baseline 결과는 이전 단계의 기록입니다.

```bash
# 로컬 NVIDIA PC: CUDA PyTorch 설치 후
python -m pip install -r requirements-local.txt
python scripts/download_data.py
python collect_normal.py --agent local --check-imports
python collect_normal.py --agent local --model Qwen/Qwen3-4B --episodes 5
```

`configs/local_llm.json`으로 모델·4-bit·생성 길이를 변경할 수 있습니다. 기본값은 NF4 4-bit, thinking 비활성화, raw logits 저장 비활성화입니다.
현재 local agent는 `PLAN: ...` / `ACTION_ID: 정수`를 출력하고 현재 원본 `admissible_actions[action_id]`를 실행합니다. 원문·ID·매핑된 action을 함께 기록하며 이전 local 버전은 `archive/local_action_string_v1/`에 보존했습니다.
전체 vocabulary token entropy와 step 평균을 nats로 저장하며, `--save-logits`에서만 raw logits를 추가 저장합니다.
Cloud에서는 import와 CPU unit test만 수행했으며 모델 다운로드·GPU 실행·실제 episode는 수행하지 않았습니다.

추가된 LLM 단계: [OpenAI Responses agent 및 실행 지침](docs/LLM_AGENT.md).
기존 PDDL baseline 코드는 그대로 보존했습니다. 현재 키 전달과 실제 모델 목록 조회는 성공했지만 Responses 생성 요청이 `HTTP 429 / insufficient_quota`로 차단되어 실제 LLM episode는 미실행입니다.
최신 결과: [OpenAI 실제 연결 검증 보고서](logs/llm_live_validation/20261103T054915.982490Z_fb8993e1/REPORT.md).
아래 실행 결과는 기존 PDDL baseline의 결과입니다.

ALFWorld 기반 LLM 로봇의 장시간 작업 실패 사전 탐지 연구를 위한 첫 단계입니다.
현재 구현 범위는 **text-only 환경에서 실제 성공 episode를 실행하고 정상 trajectory를 재현 가능하게 저장·검증하는 파이프라인**입니다.
행동은 ALFWorld가 제공하는 privileged PDDL expert가 선택합니다. LLM planning 성능이나 장시간 작업 분포를 검증한 결과는 아닙니다.
fault injection, LSTM, intervention은 구현하지 않았습니다.

## 검증한 환경

- Debian GNU/Linux 13.6 (x86_64)
- Python 3.12.14, 프로젝트 전용 `.venv` (시스템 Python 변경 없음)
- 시스템 pip 26.2.1, 가상환경 pip 25.0.1
- ALFWorld 0.4.2 / TextWorld 1.6.2 / fast-downward-textworld 20.6.4
- 모든 Python 의존성 버전은 `requirements.txt`에 고정
- `ai2thor`, `torch`, `torchvision`, `opencv-python` 및 시각 모델은 설치하지 않음
- Python 3.12, `venv`, C/C++ 빌드 도구와 `make`가 있는 Linux 환경을 기준으로 함

## 설치와 실행

```bash
cd /workspace/llmrobot
bash scripts/setup.sh

# 먼저 실제 episode 하나 실행하고 JSONL 재검증
.venv/bin/python collect_normal.py --episodes 1 --seed 42

# 서로 다른 게임 5개 실행
.venv/bin/python collect_normal.py --episodes 5 --seed 42
```

`setup.sh`는 기존 가상환경을 재사용하며 고정 의존성 설치, `pip check`, 데이터 준비를 수행합니다.
Python 실행 파일을 바꾸려면 `PYTHON_BIN=/path/to/python3.12 bash scripts/setup.sh`를 사용합니다.
collector는 설치된 `.venv/bin/python`으로 직접 실행하므로 shell activation이 필요 없습니다.
외부 API, API key, GPU, 실행 중인 서버는 필요하지 않습니다.
설치 시 `pypi.org`, `files.pythonhosted.org`, `github.com`, `release-assets.githubusercontent.com`에 HTTPS 접근이 필요합니다.

기본 데이터 경로는 `data/alfworld`, 기본 출력 경로는 `runs/`입니다.
`--data-dir`, `--output-dir`, `--split`, `--task-type`, `--max-steps`로 변경할 수 있습니다.
기본값은 `train`, `pick_and_place_simple`, 최대 50 step입니다.
공식 데이터 중 `solvable=true`이고 지원되는 서로 다른 게임을 선택하며, 저장된 walkthrough 길이가 짧은 순서로 정렬하고 동률은 seed로 결정합니다.
따라서 간단한 정상 수집 검증용 선택이며 무작위 전체 벤치마크 평가가 아닙니다.

### text-only 데이터

`scripts/download_data.py`는 공식 release의 다음 두 파일만 받습니다.

- `0.2.2/json_2.1.1_json.zip`: expert 초기화에 필요한 `traj_data.json`
- `0.4.0/json_2.1.2_tw-pddl.zip`: 실행 가능한 `game.tw-pddl`

두 archive 모두 `json_2.1.1/`에 풀립니다. 버전명 차이는 공식 배포 이름 그대로입니다.
`game.tw-pddl`에 도메인·문법·문제 정의가 포함되므로 별도 visual/PDDL 생성 파이프라인은 필요 없습니다.
최초 공식 HTTPS 다운로드에서 기록한 SHA256과 ZIP CRC를 검사합니다. 이 해시는 upstream 서명으로 주장하지 않습니다.
기존 파일은 내용이 같으면 재사용하고, 다르면 덮어쓰지 않고 오류를 냅니다.
archive와 출처 manifest는 `data/alfworld/downloads/`, `dataset_manifest.json`에 보존됩니다.

## 파일 구조

```text
llmrobot/
├── README.md
├── .python-version
├── .gitignore
├── requirements.txt
├── collect_normal.py
├── src/
│   ├── __init__.py
│   └── validate.py
├── scripts/
│   ├── setup.sh
│   └── download_data.py
├── tests/test_validation.py
├── logs/                         # 환경 정보, 오류/설치/검증 결과
├── runs/<UTC timestamp>_<uuid>/   # 실행마다 새로운 디렉터리
│   ├── config.json               # seed, 버전, 설정, 게임/코드 해시, 출처
│   ├── packages.txt
│   ├── trajectories.jsonl        # 실제 raw transition, 실패/부분 실행도 보존
│   ├── episodes.json             # episode별 결과와 오류
│   ├── validation.json           # 저장 파일을 다시 읽은 결과
│   └── run.log
├── data/alfworld/                 # Git 제외, 설치 환경에는 유지
└── .venv/                        # Git 제외
```

raw JSONL은 파일 생성 모드 `x`로 열고 각 step을 flush/fsync합니다.
기존 run 파일을 삭제하거나 덮어쓰는 동작은 없습니다. 실패한 episode를 몰래 제거하거나 다른 episode로 교체하지 않습니다.
모든 요청 episode가 성공하고 검증을 통과해야 collector가 종료 코드 0을 반환합니다.
오류 또는 step 제한 도달 시 원본을 남기고 종료 코드 1을 반환합니다. 부분 실행은 정상 corpus로 취급하지 않습니다.
run 디렉터리의 시각은 명시적으로 UTC이며 사용자 표시 시간대와 독립적인 식별자입니다.

## JSONL 필드와 시점

한 줄은 한 번의 `env.step(action)`에 대응합니다. step은 1부터 시작합니다.

| 필드 | 출처 및 의미 |
| --- | --- |
| `episode_id` | 실제 게임의 split/task/trial 상대 경로로 만든 식별자. run 내에서 고유 |
| `step` | 실제 실행한 행동의 순번 |
| `goal` | reset feedback의 `Your task is to:` 뒤 원문; 파싱 불가 시 `null` |
| `observation` | 행동 **직전** 환경 feedback |
| `plan` | 행동 직전 환경의 `extra.expert_plan` 목록. LLM 사고 과정이 아님 |
| `action` | 위 expert plan의 첫 번째 실제 실행 행동 |
| `admissible_actions` | 행동 직전 환경의 `admissible_commands` |
| `reward` | `env.step`이 반환한 값을 그대로 저장. 이 PDDL 환경에서는 score 0/1 |
| `done` | 행동 **직후** 환경이 반환한 종료 여부 |
| `success` | 행동 직후 환경의 `won`; reward로 추정하거나 마지막 성공값을 앞 step에 복사하지 않음 |
| `next_observation` | 행동 직후 feedback. 마지막 행동 결과도 보존하기 위한 추가 필드 |
| `truncated` | 수집기의 최대 step 도달 여부. 환경 `done`과 별개 |

환경이나 agent가 제공하지 않는 값은 `null`로 남기며 의미를 추정해 채우지 않습니다.
expert가 행동을 제공하지 않으면 임의 행동을 만들지 않고 해당 episode 오류를 기록합니다.
이번 정상 5개 episode에서는 요청한 11개 필드 모두 수집 가능했고 `null`이 없었습니다.
LLM 내부 reasoning/장기 자연어 계획은 현재 agent에서 제공하지 않으므로 수집하지 않습니다. `plan`은 실제 PDDL expert 출력입니다.
`success=false`인 중간 step은 정상입니다. 마지막 step의 `done=true, success=true`로 성공 episode를 판정합니다.

## 저장 파일 재검증

```bash
.venv/bin/python -m src.validate \
  runs/20261103T024323.126935Z_6dd6830c/trajectories.jsonl \
  --expected-episodes 5

# 실제 수집 로그의 임시 복사본에 누락/빈 값/실패 등을 만들어 검증기 테스트
.venv/bin/python -m unittest discover -s tests -v
```

검증은 episode 수, 각 episode step 수, 연속 step 번호, 필수 필드, null 개수,
빈 observation/action, action의 admissibility, expert plan 일치, 관찰 연속성,
각 step의 success 기록 및 최종 성공 여부를 확인합니다.
명시적 report 출력이 필요하면 `--report <새 파일 경로>`를 사용합니다. 기존 report는 덮어쓰지 않습니다.

## 실제 실행 결과

최종 의존성 조합에서 서로 다른 **5개 episode, 성공 5 / 실패 0, 총 17 step**을 실행했습니다.
각 step 수는 **3, 3, 3, 4, 4**입니다. 필수 필드 누락 0, null 0, 빈 observation 0, 빈 action 0입니다.

- 기준 raw: `runs/20261103T024323.126935Z_6dd6830c/trajectories.jsonl`
- 동일 seed 재실행: `runs/20261103T024438.177478Z_9d2bd689/trajectories.jsonl`
- 두 실행은 SHA256 `37ee64ecc269065298181dbd33837d213eedf5fd97475288f56206bc07de598b`로 **바이트 단위 동일**
- 재현성 확인: `logs/reproducibility.json`
- 독립 JSONL 재검증: `logs/independent_validation.json`
- 검증기 테스트: `logs/validation_tests.log` (8개 통과)
- 최종 설치/의존성 검사: `logs/setup_final.log`, `logs/pip_check_final.log`

초기 smoke episode 1개도 4 step으로 성공했습니다 (`logs/smoke_metadata_added.log`).
수집기 단일 episode 검증 1개와 TextWorld 1.7.0에서의 최초 5개 로그도 삭제하지 않고 보존했습니다.
최종 TextWorld 1.6.2에서는 기준 5개 + 재현성 확인 5개 = 10회 성공했으며, 서로 다른 게임은 5개입니다.

### 발생 오류와 해결

1. 게임 archive만 받은 첫 smoke 실행은 `traj_data.json` 누락으로 실패했습니다.
   `AlfredExpert`가 planner 모드에서도 metadata를 읽으므로 공식 JSON archive를 추가했고, 이후 episode가 성공했습니다.
   원본 오류는 `logs/smoke_initial.log`에 있습니다.
2. TextWorld 1.7.0은 episode 실행에는 성공했으나 배포 wheel의 `WHEEL` 파일에서 빈 줄 뒤 `Tag`가 위치해 `pip check`가 플랫폼 미지원으로 판정했습니다.
   동일 버전 소스 빌드도 시도했지만 upstream `setup.sh`의 오래된 HTTP Inform7 다운로드가 정상 archive를 반환하지 않아 실패했습니다.
   이 실패한 빌드 산출물은 설치하지 않았고 검증을 우회하지 않았습니다.
   ALFWorld의 `textworld>=1.6.1` 조건을 만족하는 공식 Python 3.12용 TextWorld 1.6.2 wheel로 고정해 설치·episode·`pip check`를 다시 통과했습니다.
   상세 로그는 `logs/pip_check_initial.log`, `logs/textworld_source_install.log`, `logs/textworld_162_install.log`에 있습니다.

현재 정상 trajectory 생성·저장·검증 기반은 다음 연구 단계를 준비할 수 있는 상태입니다.
이번 범위에서 후속 모델이나 실패 유도 기능은 추가하지 않았습니다.
