# Local LLM 구현 검증

검증일: 2026-11-04 (Asia/Seoul).

- 기본 후보 Qwen/Qwen3-4B, configuration/CLI로 변경 가능.
- 로컬 Transformers causal LLM + 선택적 NF4/double-quant 4-bit 로더.
- fresh greedy GenerationConfig와 `use_model_defaults=False`로 checkpoint sampling/filter 병합 차단.
- `return_dict_in_generate=True`, `output_scores=True`; 전체 vocabulary float32 log-softmax로 token Shannon entropy(nats) 계산.
- token별 entropy 및 단일 ALFWorld step의 산술 평균 저장. prompt 제외, 생성된 JSON/plan/action/EOS 포함.
- raw logits 기본 저장 없음. 명시적 --save-logits에서만 unprocessed generate.logits를 별도 배타적 `.pt` 파일에 저장.
- 원본 PDDL 본문은 collect_pddl.py에 바이트 단위 동일하게 보존. collect_normal.py는 backend 라우터이며 default pddl과 helper imports 유지.
- OpenAI/Gemini 소스 및 기존 raw logs 변경 없음. requirements 원본도 requirements-pddl.txt에 보존.

## 실제로 수행한 검증

- 별도 CPU-only 환경: torch 2.6.0+cpu, transformers 4.57.6, accelerate 1.12.0.
- local unit test 23개 + 기존 순수 validator unit test 8개 = 총 31개 통과.
- CUDA 확인/pretrained loader/model.generate/ALFWorld environment는 mock 사용.
- entropy 수치는 실제 CPU tensor로 균등·skewed·극단 logits·0 확률 분포를 검증.
- 실제 Transformers generation 설정 병합 함수를 직접 검사하여 checkpoint filter가 유입되지 않음을 확인.
- 네 backend의 --help와 local --check-imports 성공. import 검사에서 CUDA 여부를 조회하거나 모델을 로딩하지 않음.
- 보호한 원본 소스·raw 파일 42개는 collect_normal.py→collect_pddl.py 보존 매핑을 적용해 해시 유지 확인.
- baseline .venv 변경 없음. 코드 및 unit test 검증용 CPU 패키지는 /workspace/local-llm-check에만 설치.

## 실행하지 않은 것

pretrained 모델/토크나이저 다운로드 0, 실제 CUDA inference 0, 실제 bitsandbytes GPU 연산 0,
ALFWorld episode 0, 실제 local LLM trajectory 0.
이전 provider 테스트 중 실제 ALFWorld를 시작하는 integration suite도 실행하지 않음.
unit test의 임시 fake trajectory는 연구 corpus가 아니며 정리됨.
fault injection, LSTM, intervention 구현 없음.

## 로컬 사용자 실행

docs/LOCAL_LLM.md의 NVIDIA 설치 절차와 CLI 참고.
드라이버/CUDA PyTorch 및 bitsandbytes 확인 → 모델 로딩/메모리 확인 → episode 1개 + src.validate_local 검증 → episode 5개 순서 권장.
실제 NVIDIA 호환성, 4-bit 메모리 사용량, 모델 출력 형식/성공률은 Cloud 검사로 증명되지 않음.

증거: tests_verified.log, baseline_validation_tests.log, import_check.json, cli_help_checks.json,
preservation_result.json, status.json.
