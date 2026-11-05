# E/C/S 시계열 feature 추출

```bash
git pull --ff-only origin feat/local-llm-agent
python -m pip install -r requirements-features.txt
python extract_timeseries.py --runs-dir runs_fault --output features/timeseries.csv
```

추출기는 `runs_fault` 아래 여러 실험/episode의 `trajectories.jsonl`을 재귀 탐색합니다.
condition의 config/episodes와 상위 experiment config/model_info를 읽고 no_fault와 fault 모두 처리합니다.
정상 local run도 `--runs-dir runs_local` 또는 특정 run 경로로 처리할 수 있습니다.
Qwen causal 모델·ALFWorld 환경·GPU inference는 로딩하지 않습니다.
필요한 것은 과거 prompt 재구성을 위한 tokenizer와 CPU 문장 embedding 모델뿐입니다.
tokenizer와 embedding 가중치가 이미 cache에 있으면 `--local-files-only`를 사용할 수 있습니다.

## 정의

- **E:** 원본 JSONL의 `mean_entropy` 숫자를 그대로 사용합니다. 단위는 nats이며 재평균/스케일링/재계산하지 않습니다.
- **C:** `prompt_token_count / max_context_tokens`. 생성 token 수나 previous_actions 개수로 대체하지 않습니다.
- **S:** episode의 **original initial goal**과 해당 step의 plan을 sentence-transformers embedding으로 변환한 후 cosine similarity를 그대로 사용합니다.
  goal_drift 이후의 agent_goal을 S의 기준 goal로 바꾸지 않습니다. 음수 cosine도 그대로 저장하고 abs/0..1 변환을 하지 않습니다.

S의 기본 모델은 경량 `sentence-transformers/all-MiniLM-L6-v2`이며 CPU를 사용합니다.
`--similarity-model`과 `--similarity-revision`으로 바꿀 수 있습니다.
중복 goal/plan은 배치에서 한 번만 encode하고 실제 계산은 embedding dot product / 두 norm의 곱입니다.
Hamming distance나 문자열 exact-match를 사용하지 않습니다.
embedding 모델 자체의 max sequence length를 metadata에 남깁니다. 긴 plan은 모델의 표준 token 제한을 받을 수 있으므로 모델/revision/길이가 다른 실험의 S를 그대로 혼합하지 않는 것이 좋습니다.

## C의 실제 prompt 기준

### 앞으로 수집하는 로그

`LocalLLMAgent.decide()`에서 실제 tokenizer 처리 후의 input_ids 길이를 `prompt_token_count`로 기록합니다.
`max_context_tokens`와 모델의 effective_context_limit도 함께 저장합니다.
generation/entropy 설정 및 계산은 바꾸지 않고 생성 결과 metadata만 추가했습니다.
기존 collector가 generation metadata를 JSONL에 복사하므로 정상/fault 모두 이 값을 자동 보존합니다.
추출기는 이 경우 `prompt_count_source=recorded_at_generation`으로 표시하고 Qwen tokenizer를 다시 로딩하지 않습니다.

### 이전 로그

기존 trajectory를 수정하거나 그 안에 token 수를 덧써 넣지 않습니다.
저장된 source hash/config/system prompt 및 tokenizer revision을 사용해 원래 호출을 재구성합니다.

1. fault step은 `agent_goal`, `agent_observation` 사용; 정상 로그는 원래 goal/observation 사용
2. **원본 순서**의 admissible_actions에 action_id를 부여
3. 실제 previous_actions, JSON 직렬화 옵션과 저장된 system prompt를 사용
4. 원래의 `apply_chat_template(tokenize=False, add_generation_prompt=True, enable_thinking=...)` 적용
5. 원래와 같은 `tokenizer(prompt, return_tensors="pt", add_special_tokens=False)`의 input_ids 길이 계산

system prompt, chat markers, goal, 관찰, history와 action 목록 모두 포함한 token 수입니다.
context_overload의 추가 context와 goal_drift의 바뀐 goal도 실제 agent 입력과 같이 반영됩니다.
default max_context_tokens는 4096이지만 실제 run/row에 다른 값이 저장됐으면 **실제 설정을 우선** 사용합니다.
`--max-context-tokens`는 run 설정이 없는 경우의 fallback입니다. C를 1로 clip하거나 generation budget을 분모에서 빼지 않습니다.

`configs/prompt_formats.json`은 instrumentation 이전의 검증된 ACTION_ID prompt source hash/형식을 보존합니다.
현재 source도 자동 인식합니다. 알 수 없는 source나 다른 형식은 추정하지 않고 오류를 냅니다.
필요하면 사용자가 코드를 직접 대조한 뒤 `--allow-unverified-reconstruction`을 지정할 수 있지만 CSV에 unverified로 표시됩니다.
그 옵션도 임의의 문자열 형식을 ACTION_ID 형식으로 바꿔 지원하지 않습니다.
원래 tokenizer revision이 없는 과거 HF 로그에는 실제 사용한 `--tokenizer-revision <원래 commit/tag>`가 필요합니다.
저장된 revision과 다른 override는 기본적으로 거부합니다. 로컬 tokenizer 디렉터리는 해당 경로에서 읽을 수 있습니다.
reconstructed 수치는 source/revision을 전제로 한 재구성값이므로 observed 값과 출처를 구분합니다.

## 파일과 컬럼

기본 출력: `features/timeseries.csv`.
sidecar: `features/timeseries.csv.metadata.json`.

최소 컬럼:

```text
run_dir,episode_id,condition,step,E,C,S,prompt_token_count,max_context_tokens,
fault_type,fault_severity,fault_active,injection_step,success,outcome
```

추가 컬럼: `prompt_count_source,tokenizer_name,tokenizer_revision,similarity_model,data_kind`.
run_dir은 **해당 condition의 trajectory가 있는 디렉터리**입니다.
success는 원본 step success, outcome은 episodes.json의 최종 episode outcome입니다. 마지막 성공값을 이전 step의 success에 복사하지 않습니다.
no_fault의 severity/injection_step은 원본의 null을 빈 CSV cell로 표현합니다. 이는 누락 E/C/S와 다르며 sidecar의 nullable_columns에 명시합니다.

source별 원래 step 순서를 보존하고 step을 renumber하지 않습니다. 동일 pair에서는 no_fault 뒤 fault 순서로 출력합니다.
역순/중복 step을 발견하면 정렬해서 숨기지 않고 오류 처리합니다.
원본 trajectory 및 metadata의 SHA256을 기록하고 추출 중 변경되면 출력을 중단합니다.
원본은 읽기 전용으로 열며 output을 원본 trajectory/config/summary로 지정하는 것도 거부합니다.

## 누락/NaN 검증

E/C/S와 token count/context budget은 필수이며 NaN/inf/None/잘못된 타입을 허용하지 않습니다.
plan/initial goal/outcome 누락, zero-norm embedding, fault의 agent 입력 provenance 누락도 오류입니다.
누락값을 0이나 previous_actions 개수로 바꿔 채우지 않습니다.
기본은 fail-fast이며 잘못된 CSV를 발행하지 않습니다.

명시적으로 유효 row만 추출하려면:

```bash
python extract_timeseries.py --runs-dir runs_fault --output features/timeseries.csv --skip-invalid
```

제외한 file/line/reason은 sidecar에 저장합니다. 원래 step 번호를 유지하므로 중간 gap이 있을 수 있습니다.
기존 feature CSV도 기본적으로 덮어쓰지 않습니다. 새 output 경로 또는 `--overwrite`가 필요하며, 이 옵션도 원본 trajectory를 덮어쓰지 못합니다.

```bash
python extract_timeseries.py --runs-dir runs_fault --output features/timeseries.csv --overwrite
```

## 현재 검증과 5행 예시

Cloud에 실제 runs_fault trajectory가 없어 실제 연구 feature CSV를 생성했다고 주장하지 않습니다.
실제 tokenizer/embedding 가중치를 내려받지 않았고 Qwen/GPU/ALFWorld episode도 실행하지 않았습니다.
sentence-transformers 패키지 import는 확인했고, unit test는 synthetic trajectory, byte tokenizer 및 인위적 embedding vector로 수행했습니다.

`features/examples/timeseries.fixture.csv`는 **테스트 fixture 예시 5행**이며 연구 데이터가 아닙니다.
`data_kind=unit_test_fixture`, `tokenizer_name=fixture/tokenizer`, `similarity_model=fixture/embedding-vectors`로 구분합니다.
이 값으로 학습하거나 실제 Qwen/MiniLM의 측정 결과로 해석하면 안 됩니다. 예시의 temporary run_dir은 실제 연구 run 경로가 아닙니다.
실제 데이터는 위 기본 명령을 로컬에서 실행해 생성해야 합니다.

테스트:

```bash
python -m unittest discover -s tests -p test_timeseries_features.py -v
python -m unittest discover -s tests -p test_local_llm.py -v
python -m unittest discover -s tests -p test_fault_injection.py -v
```

이번 추가는 feature extraction까지만이며 LSTM, 학습 classifier, intervention은 구현하지 않았습니다.
