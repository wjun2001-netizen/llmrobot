# E/C/S sliding window 생성 (학습·split 없음)

```bash
git pull --ff-only origin feat/local-llm-agent
python make_lstm_windows.py \
  --input features/timeseries.csv \
  --output features/lstm/windows.csv \
  --window-size 5
```

표준 라이브러리만 사용하므로 추가 패키지 설치, 모델 로딩, GPU 실행이 필요 없습니다.
출력은 CSV이며 NPZ는 이 구현의 지원 형식이 아닙니다. LSTM 모델 학습이나 train/test split은 하지 않습니다.
기존 timeseries.csv와 trajectory를 수정하지 않습니다.

## 경계와 순서

- 상태는 각 step의 `[E, C, S]` 그대로이며 추가 normalization/feature fitting을 하지 않습니다.
- window 기본 길이 5, stride 1입니다. 길이 L인 유효 sequence에서는 최대 L-4개를 만듭니다.
- 그룹 키는 **run_dir + episode_id + condition**입니다. 같은 ID가 여러 run에 반복돼도 서로 섞지 않습니다.
- 원래 step 순서를 보존합니다. 역순/중복은 오류이며 자동 정렬로 숨기지 않습니다.
- window 내부 step은 연속해야 합니다. gap을 건너뛰어 이어 붙이거나 보간/padding하지 않습니다.
- E/C/S의 NaN, inf, 빈 값, 숫자로 파싱할 수 없는 값이 포함된 window는 제외합니다. 해당 row를 삭제하고 양쪽을 이어 붙이지 않습니다.
- 5 step보다 짧은 episode에서는 window가 0개입니다. 여러 짧은 episode를 합쳐 window를 만들지 않습니다.
- outcome/fault metadata가 한 trajectory 내부에서 모순되면 오류를 냅니다.

## 기본 label

label은 실제 failure onset 검출값이 아닌 **최종 trajectory outcome에 근거한 weak label**입니다.

| 최종 outcome / 조건 | 기본 처리 |
| --- | --- |
| success | 모든 window label 0 |
| fault + step_limit/task_failure, 모든 step이 injection_step 이상 | label 1 |
| fault + step_limit/task_failure, 주입 전 또는 경계가 섞인 window | label 0 |
| no_fault + step_limit/task_failure | 사용자 선택에 따라 학습 대상에서 제외 |
| invalid_output/generation_error/environment_error 등 | label을 만들어 넣지 않고 그룹 제외 |

주입은 injection_step의 agent 입력 직전 적용되므로 `step == injection_step`은 post-injection입니다.
예: injection_step=4, W=5일 때 [1..5], [2..6], [3..7]은 기본 0이며 [4..8]부터 failure window label 1입니다.
fault를 실제로 못 만난 failure trajectory의 window는 모두 0입니다.
이 기준은 주입 시점 이후의 failure trajectory window를 표시하는 규칙이며, failure가 바로 그 시점에 시작했다거나 fault가 원인임을 증명하지 않습니다.

선택적으로 마지막 step만 post-injection이어도 positive로 처리하려면 `--post-injection-policy end`를 사용합니다.
기본값은 `all`입니다. 사용한 정책은 sidecar에 기록됩니다.
no_fault failure를 별도의 실험에서 positive로 쓰려는 명시적 override는 `--no-fault-failure-policy positive`입니다. 기본은 `exclude`입니다.

## baseline 실패/애매한 pair 제외

```bash
python make_lstm_windows.py \
  --input features/timeseries.csv \
  --output features/lstm/windows.filtered.csv \
  --window-size 5 \
  --exclude-baseline-already-failed
```

이 옵션은 확인된 baseline_already_failed pair의 **fault 조건까지 포함해 전체 그룹**을 제외합니다.
no_fault 실패만 제외하는 기본 정책과 구분합니다.

comparison 정보는 다음 순서로 읽습니다.

1. CSV의 `pair_comparison` 또는 `comparison` 컬럼이 이미 있으면 사용
2. run_dir의 상위 `pair.json` 사용 (episode_id 일치 검사)
3. CSV에 동일 pair 경로/episode의 no_fault 최종 outcome이 있으면 baseline failure 여부를 판별

pair 경로는 표준 `.../no_fault`, `.../fault`의 공통 상위 디렉터리 또는 condition 공통 run_dir입니다.
episode_id만으로 서로 다른 run의 baseline을 연결하지 않습니다.
CSV에서 baseline success만 확인된 경우 `baseline_success_from_csv`로 출처를 구분하며 prefix divergence까지 확인했다고 주장하지 않습니다.

추가로 `--exclude-comparison pre_injection_divergence` 같은 비교 결과를 제외할 수 있습니다.
더 자세한 pair 결과를 판별할 자료가 없으면 기본은 오류입니다. 제외 옵션을 사용했는데도 확인 없이 데이터를 남기지 않습니다.
메타데이터를 확인할 수 없는 그룹도 함께 제외하려면:

```bash
python make_lstm_windows.py --input features/timeseries.csv \
  --output features/lstm/windows.filtered.csv --window-size 5 \
  --exclude-baseline-already-failed --missing-comparison-policy exclude
```

원본 CSV에 comparison 컬럼을 추가할 필요는 없습니다. 경로를 옮겨 pair.json을 찾을 수 없다면 원본을 그대로 두고 메타데이터가 포함된 별도 CSV 사본을 입력할 수도 있습니다.

## 출력과 tensor 순서

`features/lstm/windows.csv`의 컬럼:

```text
run_dir,episode_id,condition,fault_type,fault_severity,start_step,end_step,
E_1,E_2,E_3,E_4,E_5,
C_1,C_2,C_3,C_4,C_5,
S_1,S_2,S_3,S_4,S_5,
label,outcome,injection_step,pair_comparison,data_kind
```

window_size를 바꾸면 각 feature의 컬럼 수도 함께 바뀝니다.
CSV는 요청대로 feature-major 컬럼입니다. 향후 LSTM의 time-major X[t]=[E_t,C_t,S_t]를 만들 때에는 컬럼 이름을 사용해 순서를 맞춰야 합니다.
`data_kind`가 입력에 있으면 보존하고 없으면 unspecified입니다. unit_test_fixture 예시를 실제 학습 데이터로 혼동하지 않아야 합니다.

stdout과 `.csv.metadata.json`에는 다음을 기록합니다.

- 전체 sample 수
- class_counts: label 0/1 각각의 개수
- positive_samples / negative_samples
- 짧은 그룹, gap/NaN window, 알 수 없는 outcome, no_fault failure, comparison 제외 수
- input SHA256, 원본 변경 여부, group/label/window 정책, 비교에 사용한 pair.json SHA256
- train_test_split=false

유효 window가 없으면 header-only CSV와 0/0 class count를 기록합니다. 이것은 학습할 수 있는 sample이 있다는 의미가 아닙니다.
기존 derived output도 기본적으로 덮어쓰지 않습니다. 새 경로나 `--overwrite`가 필요합니다.
`--overwrite`도 input CSV나 사용한 pair metadata를 덮어쓰지는 못합니다.

## 테스트

```bash
python -m unittest discover -s tests -p test_lstm_windows.py -v
```

경계 분리, 반복 episode ID, stride/feature 순서, pre/mixed/post label, no_fault 실패 제외,
baseline comparison 제외, NaN/gap 차단, class count stdout, source 보존과 overwrite 방지를 검사합니다.
모델 학습과 실제 데이터 수집은 하지 않습니다.
