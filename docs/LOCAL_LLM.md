# 로컬 NVIDIA LLM 수집

이 단계의 연구 backend는 Hugging Face Transformers causal LLM입니다.
기본 후보는 `Qwen/Qwen3-4B`이며 configuration과 CLI로 바꿀 수 있습니다.
Codex Cloud에서는 **코드 작성, import 검사, CPU tensor 기반 unit test와 mock 검사만 수행**했습니다.
pretrained 가중치 다운로드, CUDA inference, bitsandbytes GPU 연산, ALFWorld episode는 실행하지 않았습니다.
아래 실행 명령은 사용자의 로컬 NVIDIA PC에서만 수행합니다.

## 기존 구현 보존과 파일

- 원래 `collect_normal.py`의 전체 본문을 `collect_pddl.py`에 **바이트 단위 동일하게** 보존했습니다.
- `collect_normal.py`는 `--agent {pddl,openai,gemini,local}` 라우터입니다. 기본값은 `pddl`이며 기존 옵션과 helper imports를 유지합니다.
- 기존 `collect_llm.py`, `collect_gemini.py`, OpenAI/Gemini agent와 validator, 기존 raw run은 수정하지 않았습니다.
- 기존 requirements 내용도 `requirements-pddl.txt`에 그대로 보존했고, 현재 `requirements.txt`에는 optional 설치 안내 주석만 추가했습니다.
- 추가: `local_llm_agent.py`, `collect_local.py`, `src/validate_local.py`, `configs/local_llm.json`, `requirements-local.txt`, `tests/test_local_llm.py`.

## 설치: 로컬 PC에서 수행

권장 기반은 Linux x86_64 또는 WSL2 Linux, Python 3.12, NVIDIA 드라이버, CUDA 지원 PyTorch입니다.
ALFWorld/TextWorld 및 bitsandbytes의 native Windows 호환성을 검증한 구성은 아닙니다.

```bash
cd llmrobot
python3.12 -m venv .venv-local
source .venv-local/bin/activate
python -m pip install --upgrade pip
```

먼저 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 **현재 NVIDIA 드라이버에 맞는 CUDA wheel**을 설치합니다.
이 단계에서 CPU-only PyTorch를 선택하지 않습니다. 이후:

```bash
python -m pip install -r requirements-local.txt
python scripts/download_data.py
python collect_normal.py --agent local --check-imports

# 이 두 검사는 로컬 NVIDIA PC에서만 수행
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'import bitsandbytes; print(bitsandbytes.__version__)'
```

추가 패키지: `torch>=2.6,<3`, `transformers==4.57.6`, `accelerate==1.12.0`, `bitsandbytes>=0.45.5,<1`, `safetensors>=0.5.3,<1`.
ALFWorld/TextWorld 의존성은 기존 requirements pins를 그대로 사용합니다.
Cloud 수치 검증에는 별도 경로의 CPU-only `torch==2.6.0+cpu`를 사용했으며 이 CPU wheel을 로컬 NVIDIA 요구사항으로 저장하지 않았습니다.
`--check-imports`는 imports 및 package metadata만 확인합니다. CUDA와 bitsandbytes runtime 동작 또는 실제 모델 로딩을 증명하지 않습니다.

## 모델 설정과 첫 실행

`configs/local_llm.json`:

```json
{
  "model_name": "Qwen/Qwen3-4B",
  "revision": null,
  "load_in_4bit": true,
  "compute_dtype": "float16",
  "device": "cuda:0",
  "max_new_tokens": 256,
  "max_context_tokens": 4096,
  "enable_thinking": false,
  "save_logits": false
}
```

CLI 값은 JSON 설정을 override합니다. 설정 파일을 생략해도 위 기본값을 사용합니다.
revision을 지정하거나 첫 실행에서 기록된 resolved commit을 이후 설정에 고정하면 모델 revision을 명확히 추적할 수 있습니다.
모델 ID 대신 Transformers가 지원하는 **로컬 모델 디렉터리**를 지정해도 됩니다.

```bash
# 최초 smoke: 로컬 PC에서만, 작은 출력/context budget으로 시작
python collect_normal.py --agent local --model Qwen/Qwen3-4B \
  --episodes 1 --max-new-tokens 128 --max-context-tokens 2048

python -m src.validate_local '<출력된 run 디렉터리>' --expected-episodes 1

# 요청한 표준 CLI
python collect_normal.py --agent local --model Qwen/Qwen3-4B --episodes 5

# 설정 파일 사용
python collect_normal.py --agent local --config configs/local_llm.json --episodes 5
```

모델 로딩은 `.load()`의 명시적 호출 시에만 수행합니다. import/help/check-imports는 모델을 다운로드하지 않습니다.
CUDA가 없으면 다운로드 전에 중단합니다. 인증이 필요한 HF 저장소의 경우 로컬 HF 인증을 사용하며 키 값을 설정 JSON이나 trajectory에 저장하지 않습니다.
`trust_remote_code=False`, `use_safetensors=True`, 단일 GPU device map을 사용합니다. 자동 CPU offload나 다른 모델 대체는 하지 않습니다.

4-bit는 NF4 + double quantization이며 compute dtype은 기본 float16입니다.
지원 GPU에서는 `--compute-dtype bfloat16`을 지정할 수 있습니다. `--no-load-in-4bit`로 quantization을 끌 수 있지만 VRAM 요구량이 커집니다.
4B 모델의 4-bit 가중치 단순 계산 약 2 GB는 **전체 VRAM 요구량이 아닙니다**.
비양자화 layer, quantization metadata, KV cache, attention 중간값, 저장되는 token scores가 추가됩니다.
실제 가중치 다운로드는 기본적으로 quantization 이전 safetensors이므로 VRAM 절약과 다운로드/디스크 절약은 다릅니다.
로컬에서 충분한 디스크와 VRAM을 확보하고 작은 context/output 설정부터 확인해야 합니다. 특정 VRAM 용량에서의 동작을 Cloud에서 검증하지 않았습니다.

## Agent 입력·출력과 유효 행동

입력은 `initial_goal`, `current_observation`, `previous_actions`, `admissible_actions` 네 항목입니다.
출력은 JSON `plan`과 `action`이며 plan은 제공되지 않으면 null입니다.
action은 현재 admissible_actions와 **정확한 문자열 일치**가 확인될 때만 env.step에 전달합니다.
invalid JSON, 미완료 출력, 허용되지 않는 action은 원문과 entropy를 보존하고 해당 episode를 중단합니다.
공백/대소문자 보정, fuzzy matching, PDDL 정답 대체, 출력 재작성은 하지 않습니다.

local 경로에는 AlfredExpert wrapper가 없습니다. PDDL facts/expert plan/walkthrough를 모델에 전달하지 않습니다.
전체 이전 실행 행동은 전달하지만 별도의 memory retrieval은 없습니다.
Qwen3의 기본 `enable_thinking=false`를 chat template에 전달해 불필요한 thinking 출력을 억제합니다.
context 한도를 넘으면 이전 행동/관찰을 몰래 자르지 않고 generation_error를 기록합니다.
다른 모델로 바꿀 때는 설치된 Transformers가 해당 architecture와 chat template를 지원하는지 확인해야 합니다.

## Shannon entropy 정의

한 ALFWorld step마다 다음 옵션으로 `model.generate()`를 호출합니다.

```python
return_dict_in_generate=True
output_scores=True
```

checkpoint의 generation defaults를 복사하지 않고 새로운 `GenerationConfig`를 구성합니다.
Transformers 4.50+의 기본값 병합도 막기 위해 `use_model_defaults=False`를 명시합니다.
`do_sample=False`, `num_beams=1`, penalty/forced token/bad word/action token mask 없이 greedy로 생성합니다.
따라서 sampling의 top-k/top-p 또는 유효 action 토큰 제약으로 vocabulary를 잘라낸 분포를 측정하지 않습니다.
`generate.scores`는 Transformers 정의상 **processed scores**입니다. 이 설정에서는 vocabulary 필터와 warper를 적용하지 않으며 그 전체 분포를 entropy의 출처로 기록합니다.
generation 정책을 바꾸거나 logits processor를 추가할 때는 이 의미도 바뀌므로 그대로 비교하면 안 됩니다.

각 생성 위치 t의 전체 vocabulary 점수 z에 대해 float32로 다음을 계산합니다.

```text
log_p = log_softmax(z)
p = exp(log_p)
H_t = -sum_v p[v] * log_p[v]     # 전체 vocabulary, 자연로그 → nats
mean_entropy = sum_t H_t / 생성 토큰 수
```

0 확률 항은 0으로 처리하며, NaN/+inf 또는 모든 점수가 -inf인 비정상 분포는 가짜 값으로 대체하지 않고 오류 처리합니다.
step 평균은 **해당 호출이 생성한 모든 토큰**의 평균입니다. plan·action·JSON 구문·EOS를 포함하고 prompt 토큰은 제외합니다.
`generated_token_ids`와 entropy 길이를 검증하고 EOS 포함 여부도 저장합니다.
이 값은 quantized 모델의 전체 vocabulary 조건부 분포에 대한 수치 계산이며, action 후보 수준의 entropy나 episode 평균과는 다릅니다.
bits 단위가 필요하면 nats 값을 `ln(2)`로 나눌 수 있습니다.

## 로그와 raw logits

`runs_local/<UTC>_<uuid>/`에 다음 파일을 저장합니다.

```text
config.json                  # 모델/quantization/생성 설정, seed, 게임/소스 해시, 패키지 버전
model_info.json              # 요청 모델과 revision, 실제 resolved revision
trajectories.jsonl           # 실제 env.step별 필수 데이터 + entropy
decisions.jsonl              # raw text, 생성 token IDs, entropy; invalid 출력도 보존
episode_outcomes.jsonl       # episode별 즉시 저장 checkpoint
episodes.json
validation.json
status.json
run.log
logits/episode_0000/step_00001.pt   # 명시적으로 활성화한 경우에만
```

JSONL 형태를 보여주는 **설명용 예시**입니다. 실제 실험 결과가 아닙니다.

```json
{"episode_id":"<실제 게임 상대 경로>","step":1,"goal":"<환경 목표>","observation":"<환경 관찰>","plan":"<모델 계획>","action":"<실제 허용 행동>","admissible_actions":["<실제 허용 행동>"],"previous_actions":[],"reward":0,"done":false,"success":false,"agent_type":"llm","backend":"local","model_name":"Qwen/Qwen3-4B","token_entropies":[1.2,0.3],"mean_entropy":0.75}
```

추가로 `generated_token_ids`, `score_vocab_size`, `entropy_unit`, `entropy_source`, `resolved_revision`, `next_observation`, `truncated`, `logits_file`을 저장합니다.
success는 실제 환경 won 값이며 reward나 LLM 주장으로 추정하지 않습니다. 제공되지 않는 환경 값은 null로 남깁니다.
실패 episode와 partial output을 삭제하거나 다른 성공 episode로 대체하지 않습니다.
실행마다 고유 디렉터리와 배타적 파일 생성 모드를 사용해 기존 raw 파일을 덮어쓰지 않습니다.

raw logits 기본 저장은 꺼져 있습니다.

```bash
python collect_normal.py --agent local --model Qwen/Qwen3-4B --episodes 1 --save-logits
```

활성화하면 추가로 `output_logits=True`를 요청하고 **unprocessed `generate.logits`**를 token IDs와 함께 CPU tensor `.pt` 파일로 보존합니다.
기본 entropy의 출처인 `scores`와 raw logits 파일의 출처를 혼동하지 않도록 metadata에 구분합니다.
logits 저장은 CPU RAM·VRAM·디스크 비용을 늘립니다. `output_scores=True` 자체도 생성 중 token score tensor를 메모리에 보유하므로 4-bit라고 이 메모리가 없어지지는 않습니다.
raw logits 디렉터리는 Git 추적에서 제외하지만 실제 파일을 삭제하지 않습니다.

## Backend 선택과 검증 범위

```bash
python collect_normal.py --agent pddl --episodes 5
python collect_normal.py --agent openai --model '<검증한 모델>' --episodes 5
python collect_normal.py --agent gemini --model '<검증한 Flash ID>' --episodes 5
python collect_normal.py --agent local --model Qwen/Qwen3-4B --episodes 5
```

OpenAI/Gemini는 기존 API 구현으로만 dispatch됩니다. local에 API fallback을 만들지 않았습니다.
현재 연구 실행 지침은 local이며 위 외부 API 예시는 기존 호환성 설명일 뿐 자동 실행되지 않습니다.

Cloud에서 수행한 것은 CPU PyTorch/Transformers/Accelerate import 및 새 unit test입니다.
모델/토크나이저 pretrained 로더와 CUDA 확인은 mock으로 검사했고 실제 환경도 mock을 사용했습니다.
기존 테스트 중 실제 ALFWorld에 연결하는 통합 테스트는 이번 단계에서 실행하지 않았습니다.
실제 NVIDIA 4-bit inference와 ALFWorld 성능은 미검증입니다. 모델 다운로드·episode 결과·성공률을 꾸며 기록하지 않았습니다.
fault injection, LSTM, intervention은 구현하지 않았습니다.

로컬에서 처음 확인할 순서는 **CUDA PyTorch/드라이버 → bitsandbytes → 모델 접근/디스크/VRAM → 1개 episode와 entropy alignment → 5개 episode**입니다.
