# OpenAI Responses 기반 ALFWorld agent

현재 상태: **연동 코드와 오프라인/실제 ALFWorld 통합 테스트 완료, 실제 모델 목록 조회 성공, Responses 생성과 LLM episode 실행은 quota로 차단**.
`OPENAI_API_KEY` 등록 후 실제 모델 목록은 조회했습니다. 목록의 `gpt-4.1-mini-2025-04-14`에 대한 Responses/logprobs probe는 `HTTP 429 / insufficient_quota`로 거부됐습니다.
최신 연결·quota 진단은 [실제 연결 검증 보고서](../logs/llm_live_validation/20261103T054915.982490Z_fb8993e1/REPORT.md)에 있습니다. 아래 키 누락 로그와 미실행 결과는 이전 단계의 기록입니다.
모델을 임의로 설치하거나 고정하지 않았습니다. 테스트 fixture 결과는 LLM 연구 trajectory에 포함하지 않았습니다.

## 보존한 baseline과 추가한 구조

기존 `collect_normal.py`, `src/validate.py`, `requirements.txt`, 설치 스크립트, baseline 테스트와 `runs/` 파일은 수정하지 않았습니다.
31개 보호 파일의 SHA256을 작업 전후 비교했습니다. 결과는 `logs/llm_setup_20261103T025242556726Z/baseline_preservation.json`입니다.

```text
check_openai.py             # 실제 모델 목록 및 명시한 모델의 API/logprobs probe
collect_llm.py              # 별도 LLM collector
src/llm_agent.py            # OpenAI Responses client, prompt, 출력 검사
src/llm_io.py               # 실행별 고유 경로, 배타적 파일 생성, durable JSONL
src/validate_llm.py         # 실패 episode도 유효 연구 데이터로 인정하는 별도 검증기
tests/test_llm_pipeline.py  # 오프라인 fixture + 실제 ALFWorld 통합 테스트
runs_llm/<UTC>_<uuid>/      # LLM 전용 출력, baseline runs/와 분리
logs/openai_checks/         # 모델 조회/probe 기록
```

새 모듈은 Python 표준 라이브러리를 사용합니다. baseline의 의존성 파일을 변경하거나 SDK/모델을 설치할 필요가 없습니다.

## Agent 구조

각 step마다 다음 네 입력만 JSON으로 전달합니다.

1. `initial_goal`: reset observation에서 추출한 목표 원문
2. `current_observation`: 현재 환경 feedback
3. `previous_actions`: 이전에 실제로 실행된 행동 전체, 시간순
4. `admissible_actions`: 현재 환경이 제공한 행동 목록

반환 형식은 `{"plan": "현재의 간단한 작업 계획", "action": "현재 admissible 목록의 정확한 문자열"}`입니다.
`plan`이 제공되지 않으면 `null`을 기록합니다. 내부 chain-of-thought를 요구하거나 기록하는 필드가 아닙니다.

Responses Structured Outputs의 `text.format`에 action enum을 전달해 **생성 단계에서** 선택지를 제한합니다.
수신 후에도 action 문자열이 목록과 정확히 일치하는지 검사합니다.
공백 보정, 대소문자 변경, fuzzy matching, 정답 action 대체, 계획 수정, 오류 출력 재작성 또는 재질의는 수행하지 않습니다.
형식 오류·refusal·incomplete 출력·유효하지 않은 action은 원문을 저장하고 해당 episode를 중단합니다.

환경에는 `AlfredDemangler`만 적용합니다. `AlfredExpert`를 설치하지 않고 `policy_commands`, facts, walkthrough를 요청하지 않습니다.
게임 선택은 기존 baseline과 동일한 seed와 선택 함수로 비교 가능하게 하되, 게임의 PDDL·walkthrough·expert plan은 모델 입력에 전달하지 않습니다.
별도 memory retrieval, fault injection, LSTM, intervention은 없습니다.

## 인증과 모델 선택

- Endpoint: `https://api.openai.com/v1/responses` (고정 공식 endpoint)
- 인증: **`OPENAI_API_KEY` 환경 변수**에서만 읽음
- 모델: 명시적 `--model` 또는 선택 후 설정한 `ALFWORLD_LLM_MODEL`; 기본 모델 없음
- API 키는 코드·저장소·JSONL·설정 JSON·로그에 저장하지 않음
- 키가 없으면 네트워크 호출이나 ALFWorld reset 전에 종료 코드 **2**로 중단
- HTTPS 인증서 검증 유지, 인증 헤더의 redirect 전달 차단, HTTP 오류 본문은 credential 노출 방지를 위해 저장하지 않음
- 네트워크 정책에서 `api.openai.com`의 HTTPS 접근이 필요함

키는 채팅이나 저장소에 붙여 넣지 말고 실행 환경에 안전하게 주입해야 합니다.
`OPENAI_`는 이 클라우드의 프록시 secret target 예약 접두사이므로 `OPENAI_API_KEY`를 그 target으로 선언할 수 없습니다.
변수명을 임의로 바꾸거나 Codex 내부 인증을 재사용하지 않습니다. 이 코드는 직접 주입된 `OPENAI_API_KEY`를 요구합니다.

키 설정 후, 먼저 실제 접근 가능한 모델을 확인합니다.

```bash
cd /workspace/llmrobot
.venv/bin/python check_openai.py models
```

`GET /v1/models`는 접근 가능한 모델 ID를 반환하지만, logprobs 지원 flag를 제공하지 않습니다.
목록에 있다는 이유만으로 지원 모델이라고 판정하지 않습니다.
실제 목록에서 후보 ID를 지정해 probe를 실행합니다. 아래 `<...>`는 실제 확인한 값으로 바꾸는 자리이며 지정된 모델명이 아닙니다.

```bash
.venv/bin/python check_openai.py probe --model '<실제 목록에서 확인한 모델 ID>'
```

probe는 실제 모델 접근, Responses 호출, Structured Outputs와 `message.output_text.logprobs` 반환을 확인합니다.
이 호출은 간단한 프로토콜 검사이며 ALFWorld episode 수에 포함하지 않습니다. 실제 토큰 사용 및 요금이 발생할 수 있습니다.
반환 logprobs가 없거나 출력이 완료되지 않으면 지원 여부를 미확인으로 남깁니다.
`probe.json`에 요청 모델, 실제 응답 모델, 검사 결과와 원본 응답을 보존합니다.

검증한 모델로 5개 episode를 실행합니다. collector도 시작 시 모델 목록과 probe를 다시 확인한 뒤 환경을 엽니다.

```bash
.venv/bin/python collect_llm.py \
  --model '<검증한 모델 ID>' --episodes 5 --seed 42 --max-steps 50
```

`--top-logprobs` 기본값은 5 (범위 1~20), `--max-output-tokens` 기본값은 512입니다.
모델별 지원 여부가 다른 seed/temperature 인자를 임의로 추가하지 않습니다.
저장된 seed는 게임 선택과 환경 RNG를 위한 값이며 원격 모델 출력의 결정성을 보장하지 않습니다.

## 로그와 실패 보존

실행마다 새 `runs_llm/<UTC timestamp>_<uuid>/`를 만듭니다.

| 파일 | 내용 |
| --- | --- |
| `status.json` | complete/incomplete/blocked, 실제 실행 수와 통계 |
| `api_config.json` | endpoint, 요청 모델, key 존재 여부, 출력 설정; key 값 제외 |
| `models.json`, `capabilities.json` | 실제 모델 목록과 probe 응답, 가능할 때만 생성 |
| `config.json` | 환경 버전, seed, 게임 파일/수집 코드 해시, 고정 prompt와 field 의미 |
| `trajectories.jsonl` | 실제 실행된 `env.step`별 데이터만 저장 |
| `decisions.jsonl` | 각 API 의사결정 입력, plan/action 및 원본 Responses 응답. 잘못된 출력과 API 오류도 보존 |
| `episode_outcomes.jsonl` | episode 종료 때마다 fsync한 결과 checkpoint |
| `episodes.json` | episode별 최종 outcome, step 수, done/success/error |
| `validation.json`, `run.log` | 재검증 결과와 진행 상태 |

실행된 step의 필수 필드:

`episode_id, step, goal, observation, plan, action, admissible_actions, previous_actions, reward, done, success, agent_type, model_name`.

`agent_type`은 `llm`이고, `model_name`은 **실제 응답의 model**입니다. 응답에 없으면 `null`이고 요청 모델은 `requested_model`로 별도 저장합니다.
observation/plan/action/admissible_actions/previous_actions는 행동 전, reward/done/success는 행동 후 값입니다.
`success`는 환경의 `won`이며 reward나 모델 주장으로 추정하지 않습니다. `next_observation`과 수집기의 `truncated`도 저장합니다.

API 출력 오류로 행동을 실행하지 못했다면 가짜 transition을 만들지 않습니다.
그 시도의 원문은 `decisions.jsonl`, zero-step/partial 결과는 episode outcome에 남습니다.
API 오류가 발생하면 동일 오류를 반복 호출하지 않고 나머지 episode 실행을 중단합니다.
성공, 환경 실패, step 제한, invalid output, API error, environment error는 서로 다른 outcome으로 기록됩니다.
실패 episode의 raw 파일은 삭제하거나 정상 데이터로 대체하지 않습니다.

키가 없는 현재 실행에서는 `status.json`과 `run.log`만 생성했습니다. trajectory나 성공/실패 결과를 꾸며 만들지 않았습니다.

## Output token logprobs

공식 SDK 타입 정의에서 확인한 요청 형식:

```json
{"include": ["message.output_text.logprobs"], "top_logprobs": 5}
```

각 transition의 `output_logprobs`는 output/content 위치, 원본 output text와 token별 `logprobs`를 보존합니다.
API가 반환한 `token`, `bytes`, `logprob`, `top_logprobs`를 수정하지 않습니다. 제공되지 않으면 해당 logprobs는 `null`입니다.
전체 Responses 응답도 `decisions.jsonl`에 남기므로 text/token 정렬 정보를 재확인할 수 있습니다.
Shannon entropy 계산은 아직 구현하지 않았습니다. top-k는 전체 vocabulary 분포가 아니므로 이를 그대로 전체 entropy라고 부를 수 없습니다.
또한 JSON schema/enum으로 제한된 출력 토큰의 불확실성과 행동 선택 자체의 불확실성은 구분해야 합니다.

API 필드 출처는 OpenAI 공식 Python SDK입니다.
[create params](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_create_params.py),
[output text](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_output_text.py).
확인한 소스의 SHA256은 `logs/llm_setup_20261103T025242556726Z/api_reference_evidence.json`에 기록했습니다.
문서 사이트는 네트워크 proxy 403으로 읽지 못해 공식 GitHub 소스를 확인했습니다.
이 자료는 API 형식의 근거이며 특정 모델의 현재 지원이나 계정 접근 권한을 증명하지 않습니다.

## 검증 및 현재 결과

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.validate_llm '<실제 LLM run 디렉터리>' --expected-episodes 5
```

baseline validator는 성공 정상 trajectory를 요구합니다. LLM validator는 **데이터 유효성**과 **실험 완료 여부**를 구분하여 실패 episode도 보존·검증합니다.
필수 필드, 실제 admissibility, 이전 행동 이력, step 번호, 관찰 연속성, 요약과 raw 결과의 일치를 검사합니다.
평균 step은 시도한 episode별 실제 `env.step` 횟수의 산술 평균이며, API/agent 오류 수는 작업 실패와 별도로 보고합니다.

최초 구현 단계의 실제 LLM 결과 (키 누락 당시 기록; 최신 blocker는 위 quota 진단 참고):

- 사용 모델: `null` (미선정; API 키 없음)
- 실행 episode: **0**, 성공 **0**, 실패 **0** — 미실행이며 성능 수치가 아님
- 평균 step: `null`
- 실제 LLM trajectory JSONL: 없음
- 차단 상태: `runs_llm/20261103T030310.825180Z_e58d7e88/status.json`
- 모델 조회 차단 기록: `logs/openai_checks/20261103T030310.752826Z_e8276b00/status.json`
- 오프라인/실제 ALFWorld 통합 테스트: `logs/llm_setup_20261103T025242556726Z/` 참고

테스트의 artificial Responses fixture는 임시 파일에만 사용하고 삭제합니다. 실제 ALFWorld 연결·비정상 출력 거부·실패 보존을 검사하지만 실제 LLM 동작/성능을 증명하지 않습니다.
현재는 **fault injection 단계 진행 준비 미완료**입니다. 키 설정 후 모델 및 logprobs 지원 검증, 최소 5개 실제 LLM episode 수집과 결과 재검증이 남아 있습니다.
