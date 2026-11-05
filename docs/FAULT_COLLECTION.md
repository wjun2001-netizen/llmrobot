# Runtime input fault 데이터 생성

지원 fault는 `goal_drift`, `observation_loss`, `context_overload` 세 가지이며 severity는 `low/high`입니다.
기존 정상 collector, LocalLLMAgent, ACTION_ID parsing/mapping, generation/entropy 구현을 변경하지 않았습니다.
LSTM, feature classifier, intervention, memory_noise, HHI는 구현하지 않았습니다.

## 로컬 PC에서 1개 episode

```bash
git pull --ff-only origin feat/local-llm-agent
python collect_fault.py --agent local --model Qwen/Qwen3-4B \
  --fault goal_drift --severity low --injection-step 2 --episodes 1 --seed 42
```

`--injection-step`은 **해당 step의 agent.decide 호출 직전**입니다. step 1부터 주입할 수 없으며 지정 범위는 2..max_steps입니다.
위 명령은 step 2부터 종료까지 agent에게 전달하는 goal만 바꾸고 환경 goal은 그대로 사용합니다.
실험할 게임이 너무 일찍 끝나면 실제 주입은 발생하지 않으며, 이는 별도 기록됩니다.

특정 게임을 고정하려면:

```bash
python collect_fault.py --agent local --model Qwen/Qwen3-4B \
  --fault observation_loss --severity low --episodes 1 --seed 42 \
  --episode-id 'train/<실제 task 디렉터리>/<실제 trial 디렉터리>' --injection-step 2
```

또는 `--game '/실제/경로/game.tw-pddl'`을 사용합니다. 포함 디렉터리 경로도 허용합니다.
`--episode-id`는 dataset의 `json_2.1.1/` 아래 상대 경로이며 기존 정상 JSONL의 episode_id를 그대로 사용할 수 있습니다.
고정 game/episode_id는 `--episodes 1`과 함께 사용합니다. `--episodes N`은 서로 다른 게임 수이고 paired에서는 각 게임을 두 번 실행합니다.

## 동일 게임/seed paired 실행

```bash
python collect_fault.py --agent local --model Qwen/Qwen3-4B \
  --fault goal_drift --severity low --paired --episodes 1 --seed 42

python collect_fault.py --agent local --model Qwen/Qwen3-4B \
  --fault context_overload --severity high --paired --episodes 1 --seed 42 \
  --episode-id 'train/<실제 task 디렉터리>/<실제 trial 디렉터리>'
```

`--paired`는 **동일한 game path와 환경 seed로 no_fault → fault**를 실행합니다.
모델은 한 번 로딩하고 각 조건 시작 전 torch seed를 동일하게 재설정합니다. 환경은 조건마다 새로 생성하고 RNG/history를 재설정합니다.
fault 생성은 별도 해시 기반 RNG를 사용해 모델/환경의 RNG를 소비하지 않습니다.

단독 no_fault 실행도 가능합니다.

```bash
python collect_fault.py --agent local --model Qwen/Qwen3-4B \
  --fault goal_drift --condition no_fault --episodes 1 --seed 42 --game '<실제 game 경로>'
```

원래부터 task_failure/step_limit인 baseline은 `baseline_already_failed`로 구분합니다.
baseline 성공, 유효 주입 도달, 주입 전 action/state prefix 일치, fault 조건 task 실패가 모두 확인되면
`baseline_success_fault_task_failure`라는 **관측 결과**를 기록합니다.
이는 인과성을 증명하는 학습 classifier가 아닙니다. GPU 비결정성 등의 한계가 있고, 주입 전 prefix가 다르면 `pre_injection_divergence`로 표시합니다.
형식/생성/환경 오류는 task 실패와 구분합니다. baseline generation/environment 오류이면 동일한 장애를 반복하지 않고 fault 조건을 미실행으로 기록합니다.

## 주입 시점 결정

1. `--injection-step`이 있으면 해당 값을 사용합니다.
2. paired이면 방금 수집한 no_fault의 실제 실행 step 수를 사용합니다.
3. 단독 fault에 `--baseline-run '<기존 정상 run 디렉터리>'`를 주면 동일 episode_id의 실제 step 수를 사용합니다.
4. 명시한 `--reference-steps N`을 사용합니다.
5. 위 기준이 없으면 게임 파일의 PDDL walkthrough **길이만** 기준으로 사용합니다. expert 행동 자체를 모델에 전달하지 않습니다.
6. 길이 정보가 없으면 기준 20을 사용합니다.

기본 계산은 `min(max_steps, max(2, length // 2 + 1))`입니다.
`timing.json`에 실제 선택한 step, 출처, 기준 길이를 저장합니다. PDDL 기준 길이를 실제 LLM baseline 길이라고 부르지 않습니다.
paired에서는 외부 `--baseline-run`을 함께 사용할 수 없습니다. 현재 no_fault의 실제 결과가 기준입니다.

`injection_step`은 예정된 시작점입니다. summary의 `first_active_step` 및 `first_effective_input_step`은 실제 도달 여부입니다.
episode가 먼저 끝나면 실제 시점은 null이고 `injection_reached/effective_perturbation_reached`는 false입니다.
빈 goal/관찰 등으로 perturbation이 적용되지 않았으면 fault_active가 true여도 payload.applied는 false입니다.
`status.json`의 `fault_data_ready`는 모든 fault 조건에 실제 입력 변형이 도달했는지까지 확인합니다.
완료(exit 0)는 데이터/실험 실행 완료를 뜻하며, fault 적용이나 task 성공을 무조건 보장하지 않습니다.

## 세 종류의 실제 변형

- **Goal drift:** 원본 goal은 유지하고 agent_goal만 바꿉니다. 일반적인 put/task 문장에서 low는 다른 receptacle, high는 가능한 경우 object도 바꿉니다.
  entity 목록에서 원래와 다른 값을 deterministic하게 선택합니다. 알 수 없는 문장은 대체 goal fallback으로 표시합니다.
  대체 entity가 실제 장면에 존재하거나 대체 목표가 풀리는지 확인한 것으로 주장하지 않습니다. `fault_payload`에 전략/원래 값/대체 값을 남깁니다.
- **Observation loss:** `you see ...`의 visibility detail을 일부/전체 제거하고 `[OBSERVATION_GAP]` marker를 남깁니다.
  해당 구문이 없으면 deterministic한 연속 token span을 제거합니다. 제거한 원본 span과 위치를 기록합니다.
  admissible action 목록을 제거하지 않으므로 그 목록에서 정보가 드러날 수 있으며 관찰 손실이 실패를 보장하지 않습니다.
- **Context overload:** agent_observation 뒤에 `[UNRELATED_CONTEXT]`로 구분한 무관한 archive/weather 기록을 붙입니다.
  low=8줄, high=64줄이며 내용·문자 수·토큰 수를 payload에 기록합니다.
  토큰 수는 실제 로컬 tokenizer의 `encode(추가 블록, add_special_tokens=False)` 길이입니다. 전체 chat prompt의 토큰 증가량과는 다릅니다.
  tokenizer가 없거나 계수에 실패하면 token count는 null이며 임의 추정값을 만들지 않습니다.
  high가 context 한도를 넘으면 기존 agent가 generation_error로 중단하며, 이를 task 실패로 오인하지 않습니다.

## 데이터 경로와 필드

```text
runs_fault/<UTC>_<uuid>/
  config.json                  # 모델/seed/게임 hash/소스 hash/요청된 fault 및 timing 기준
  model_info.json
  outcomes.jsonl
  results.json
  status.json
  episodes/0000/
    timing.json
    pair.json                  # paired일 때만
    no_fault/                  # 해당 조건을 실행한 경우
      config.json
      trajectories.jsonl
      decisions.jsonl
      episodes.json
      validation.json
    fault/
      config.json
      trajectories.jsonl
      decisions.jsonl
      episodes.json
      validation.json
```

모든 파일은 새 실행 디렉터리에 생성합니다. 기존 코드/trajectory/log를 덮어쓰거나 삭제하지 않습니다.
각 조건의 원래 step 필드(plan, action_id/action, admissible_actions, raw_model_output, token_entropies/mean_entropy, reward/done/success)를 유지합니다.
추가 필드는 다음과 같습니다.

`fault_type, fault_severity, fault_active, injection_step, original_goal, agent_goal, original_observation, agent_observation, fault_payload`.

기존 `goal`, `observation` 필드는 계속 환경 원본입니다. 모델이 받은 변형 입력은 agent_goal/agent_observation입니다.
true success는 항상 환경 won으로 기록하며 goal drift의 대체 목표로 재판정하지 않습니다.
fault는 원래 agent.decide 직전 입력 경계에만 적용하고, env/state 객체 및 admissible_actions를 injector에 전달하지 않습니다.
어댑터가 원본 순서의 action 복사본만 agent에 넘기며 반환된 ID/action이 원본 매핑과 일치하는지도 다시 확인합니다.
정상 collector와 model.generate/entropy 함수는 그대로 재사용합니다. JSONL stream adapter는 metadata만 추가하고 원래 goal/observation을 검사합니다.

invalid_output은 decisions.jsonl에 fault metadata/원문/entropy를 남깁니다. 실행하지 않은 step에 가짜 reward/done/success transition을 만들지 않습니다.

검증:

```bash
python -m src.validate_fault '<run>/episodes/0000/no_fault'
python -m src.validate_fault '<run>/episodes/0000/fault'
```

`config.json`, `pair.json`, summary의 실제 주입 여부와 prefix 일치를 함께 확인해야 합니다.
주입 이전/미주입 데이터도 보존하되 fault 효과가 확인된 데이터라고 표기하지 않습니다.

## 검증 범위

Cloud에서는 fake agent/environment 및 CPU unit test만 실행했습니다. 실제 Qwen 다운로드/실행이나 ALFWorld episode를 실행하지 않았습니다.
기존 정상 local 테스트 29개를 그대로 유지하며, input boundary·metadata·paired outcome 및 deterministic fault 생성에 대한 새 테스트를 추가했습니다.
새 Python 의존성은 없습니다. 로컬 실행은 기존 requirements-local.txt 환경을 그대로 사용합니다.
