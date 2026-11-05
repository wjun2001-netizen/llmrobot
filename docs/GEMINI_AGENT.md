# Gemini Developer API 기반 ALFWorld agent

현재 상태: **Gemini 모듈 및 테스트 구현 완료, 실제 모델/생성/logprobs 검증은 네트워크 차단으로 미완료**.
현재 프로세스에 `GEMINI_API_KEY`가 존재하지만 공식 API로 향한 CONNECT 요청이 프록시 `403 Forbidden`으로 차단됐습니다.
Google API에 도달하지 않았으므로 키 오류, 무료 티어 quota, 모델 지원 여부로 해석하지 않습니다.
`generativelanguage.googleapis.com`을 환경 설정 초안의 허용 목록에 추가했습니다. 초안 저장은 실행 환경 반영이나 게시가 아닙니다.

## 보존 및 추가 구조

기존 PDDL baseline과 OpenAI agent의 소스·설정 파일·로그를 변경하지 않았습니다.
작업 전의 기존 파일 80개를 SHA256으로 확인했습니다.

```text
check_gemini.py             # Gemini 모델 목록과 명시한 Flash 모델의 실제 probe
collect_gemini.py           # 별도 Gemini collector
src/gemini_agent.py         # Gemini Developer API REST client와 agent
src/validate_gemini.py      # token logprobs 및 공통 transition 검증
tests/test_gemini_pipeline.py
docs/GEMINI_AGENT.md
logs/gemini_checks/         # 실제 모델 목록·probe·차단·테스트 결과
runs_gemini/<UTC>_<uuid>/   # 앞으로 실행할 Gemini trajectory 전용 경로
```

Python 표준 라이브러리만 추가 사용합니다. 기존 requirements와 설치 스크립트를 수정하거나 모델/SDK를 설치하지 않습니다.
기존 공통 JSONL IO, 게임 선택 함수 및 transition 검증 함수는 읽기 전용으로 재사용합니다.
Gemini 경로에서 OpenAI API는 호출하지 않습니다.

## 인증과 모델 확인

- API: Google Gemini Developer API (`https://generativelanguage.googleapis.com/v1beta`)
- 키: `GEMINI_API_KEY`에서만 읽어 `x-goog-api-key` 헤더로 전송. URL query, 로그, 설정 JSON, source에 키를 넣지 않음
- 모델: CLI `--model` 또는 검증 후 지정하는 `ALFWORLD_GEMINI_MODEL`. 기본 모델 없음
- 인증서 검증 유지, redirect에 키 전달 금지, 오류 메시지의 키 값은 redaction
- 목록의 `supportedGenerationMethods`에 `generateContent`가 있고 이름에 Flash가 포함된 모델만 후보로 인정
- 모델 목록이나 공식 SDK 타입에 logprobs 필드가 있다는 이유만으로 실제 지원을 주장하지 않음

네트워크 반영 후 실행:

```bash
cd /workspace/llmrobot
.venv/bin/python check_gemini.py models
.venv/bin/python check_gemini.py probe --model '<실제 목록에서 확인한 Flash ID>'
```

probe는 실제 텍스트 생성, JSON plan/action 출력, 토큰별 log probability 반환을 확인합니다.
이때부터 Gemini API의 실제 사용량이 발생합니다. probe를 ALFWorld episode 실적에 포함하지 않습니다.
요청 설정은 `generationConfig.responseLogprobs: true`, `generationConfig.logprobs: 5`입니다.
`--top-logprobs`, `--max-output-tokens`로 수집 설정을 조절할 수 있습니다.

성공 응답에는 계정의 무료/유료 billing tier를 확정하는 필드가 없으므로 `free_tier_verified`와 `billing_tier`는 미확인일 때 `null`입니다.
무료 티어 대상 모델 및 해당 키의 실제 과금 등급은 현재 모델/계정 설정에서도 확인해야 합니다.
API 호출 성공과 무료 청구 확인을 동일한 사실로 보고하지 않습니다.

`--without-logprobs`는 명시적 생성 전용 진단 옵션입니다. 오류가 났다는 이유로 자동 사용하지 않습니다.
기본 수집은 logprobs를 요청하고 실제 반환 여부를 확인하며, 반환되지 않으면 지원을 주장하거나 다른 값을 만들어 채우지 않고 중단합니다.

## ALFWorld 입력과 행동 선택

모델에는 `initial_goal`, `current_observation`, `previous_actions`, `admissible_actions` 네 필드만 전달합니다.
계획은 현재의 짧은 작업 계획이고 내부 chain-of-thought가 아닙니다.
PDDL facts, expert plan, walkthrough는 보내지 않으며 환경에도 AlfredExpert wrapper를 적용하지 않습니다.

출력 JSON은 `plan` (문자열 또는 null)과 `action`입니다.
Gemini response schema에서 action enum을 현재 admissible 목록으로 제한하고, 수신 후에도 정확한 문자열 일치만 허용합니다.
대소문자·공백·의미를 보정하거나 fuzzy matching으로 다른 action으로 바꾸지 않습니다.
invalid JSON, 미완료 출력, safety refusal, 허용되지 않는 action은 원문을 보존하고 해당 episode를 중단합니다.
정답 대체, PDDL fallback, 출력 재작성, memory retrieval, fault injection, LSTM, intervention은 없습니다.

## 1개 → 검증 → 5개 실행

```bash
.venv/bin/python collect_gemini.py --model '<실제 검증한 Flash ID>' --episodes 1 --seed 42
.venv/bin/python -m src.validate_gemini '<위 명령이 출력한 run 디렉터리>' --expected-episodes 1

# 1개 episode가 정상 완료되고 실제 저장값이 검증된 뒤에만 실행
.venv/bin/python collect_gemini.py --model '<동일하게 검증한 Flash ID>' --episodes 5 --seed 42
.venv/bin/python -m src.validate_gemini '<5개 run 디렉터리>' --expected-episodes 5
```

각 collector 실행은 실제 모델 목록과 capability probe를 다시 확인합니다.
기본 max_steps는 50, 기본 생성 요청 최소 간격은 13초입니다. 이는 보수적인 요청 간격이며 실제 무료 티어 quota를 보장하지 않습니다.
`--request-interval`로 조절 가능하고, API quota 오류가 발생하면 반복 호출하지 않고 남은 episode를 중단합니다.
환경 seed와 게임 선택은 저장하지만 API 모델 출력의 결정성을 보장하지 않습니다.

## 저장 스키마와 보존

`runs_gemini/<UTC timestamp>_<uuid>/`에 배타적 파일 생성 모드를 사용하고 JSONL은 매 step flush/fsync합니다.

- `config.json`: seed, 실행 설정, 실제 게임/코드 해시, prompt, 환경 버전
- `api_config.json`, `models.json`, `capabilities.json`: 키 값 없이 API 설정과 실제 조회/probe 증거
- `trajectories.jsonl`: 실행한 env.step만 저장
- `decisions.jsonl`: 각 모델 출력 및 원본 응답, 잘못된 출력/API 오류도 보존
- `episode_outcomes.jsonl`, `episodes.json`: 최종 결과와 실제 step 수, 오류 유형
- `validation.json`, `status.json`, `run.log`: 데이터 검증, 실험 완료 여부, 통계

필수 transition 필드:

`episode_id, step, goal, observation, plan, action, admissible_actions, previous_actions, reward, done, success, agent_type, model_name, token_logprobs, logprobs_available`.

`agent_type`은 `llm`, `provider`는 `google_gemini_developer_api`로 별도 표시합니다.
`model_name`은 실제 반환된 `modelVersion`이고 누락 시 `null`, 요청한 식별자는 `requested_model`에 따로 보존합니다.
goal/observation/plan/action/history는 행동 전 정보이며 reward/done/success는 행동 후 실제 환경 값입니다.
success는 환경 `won`으로 기록하고 reward나 모델의 주장으로 추정하지 않습니다.
`next_observation`, 수집기 `truncated`, 실제 usage metadata와 response ID도 저장합니다.

`token_logprobs`는 `candidates[0].logprobsResult.chosenCandidates`의 원본 목록을 보존합니다.
각 token의 `token`, `tokenId`, `logProbability`를 변경하지 않습니다.
`logprobs_result`에는 topCandidates와 합계 등 반환된 원본 구조 전체를 보존하며, 전체 응답은 decisions 파일에도 저장합니다.
token별 값이 실제로 제공되지 않으면 `token_logprobs=null`, `logprobs_available=false`입니다. 평균 logprob만으로 token별 값을 복원하지 않습니다.

실패 episode 및 부분 실행을 삭제하거나 성공 episode로 대체하지 않습니다.
행동 실행 전 실패는 fake transition을 만들지 않고 decisions 및 episode 결과에 남깁니다.
validator는 데이터 유효성, 실험 완료 여부, task 실패, API/agent 오류를 구분합니다.

## Shannon entropy에 대한 범위

실제 token logprobs가 확보되면 선택 token의 surprisal과 top-k 기반 근사 분석에 사용할 수 있습니다.
그러나 선택된 token 확률과 top-k 후보만으로 전체 vocabulary 분포의 정확한 Shannon entropy를 계산할 수는 없습니다.
구조화 JSON/행동 enum 제약이 있는 출력의 token 불확실성도 행동 선택 불확실성과 구분해야 합니다.
현재는 실제 API/token 데이터가 없으므로 entropy 계산 데이터는 **충분하지 않습니다**. entropy 계산 자체는 구현하지 않았습니다.

## 현재 검증 결과

- GEMINI_API_KEY: 존재 확인, 값 미출력
- 실제 Google 모델 목록/모델명/생성: 미확인 — proxy CONNECT 403
- 무료 티어 및 logprobs 지원: 미확인
- 실제 1개 / 5개 episode: 모두 미실행
- 성공/실패: 0/0 (미실행), 평균 step null, 실제 Gemini trajectory JSONL 없음
- 테스트: 총 43개 통과 (새 Gemini 테스트 17개 + 기존 26개)
- Gemini fixture는 임시 디렉터리에만 사용했으며 실제 호출/trajectory 수에 포함하지 않음
- 기존 파일 80개 해시 유지: `logs/gemini_checks/20261103T062614.440173Z_8aaccca4/preservation_result.json`
- API 필드 출처: [공식 Google GenAI Python SDK](https://github.com/googleapis/python-genai/blob/main/google/genai/types.py), 확인한 소스 해시는 같은 로그 디렉터리의 `api_reference_evidence.json`

환경 설정에서 추가된 Gemini 도메인을 검토·저장하고 Publish한 후 실제 API 모델 조회부터 재개해야 합니다.
설정 초안 저장은 실행 정책 변경이나 새 환경에서의 성공을 증명하지 않습니다.
