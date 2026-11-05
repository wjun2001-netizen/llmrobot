# LLM agent 단계 결과

검증일: 2026-11-03 (Asia/Seoul). 실행 디렉터리 식별자는 UTC.

## 구현

- 기존 PDDL baseline 코드·설치 파일·baseline 테스트·raw runs 등 31개 파일 SHA256 불변 확인.
- 별도 `collect_llm.py`, `src/llm_agent.py`, `src/llm_io.py`, `src/validate_llm.py`, `check_openai.py` 추가.
- OpenAI 공식 Responses API. 인증은 `OPENAI_API_KEY`에서만 읽음. 모델 자동 선택·다운로드·고정 없음.
- 네 입력(goal, observation, 이전 실행 행동, admissible actions)만 전달. expert plan/facts/walkthrough 전달 안 함.
- 생성 시 action enum 제한 및 수신 후 exact membership 검사. plan/action 수정, fuzzy matching, expert 대체, 재질의 없음.
- 실제 env.step만 trajectory에 기록. 잘못된 출력·API 오류는 decisions 및 episode outcome에 별도 보존.
- logprobs 요청 및 원본 token/bytes/logprob/top_logprobs 저장 구현. entropy 계산은 미구현.
- LLM 데이터는 `runs_llm/`, 기존 PDDL 데이터는 `runs/`에 분리.

## 검증한 것

- 테스트 26개 통과 (기존 baseline 8개 + LLM 18개).
- 실제 ALFWorld에 expert wrapper 없이 연결되고 policy_commands/expert_plan/facts가 노출되지 않음 확인.
- 테스트용 Responses fixture로 실제 ALFWorld 연동, 이전 행동 이력, step 제한 실패 보존 검증.
- 잘못된 action은 원문을 저장하고 env.step을 호출하지 않음 확인.
- 누락 키가 API 요청 및 episode 실행 전에 종료 코드 2로 차단됨을 실제 CLI로 확인.
- 원본 파일 배타적 생성, 중복 키·형식 오류·incomplete/refusal 거부 검증.
- 테스트 fixture는 임시 디렉터리에만 기록하며 연구 corpus 또는 실제 LLM 실적에 포함하지 않음.

## 실제 OpenAI 실행 결과

| 항목 | 결과 |
| --- | --- |
| 계정에서 접근 가능한 모델 | 미조회: OPENAI_API_KEY 없음 |
| 모델별 Responses / logprobs 지원 | 미검증 |
| 사용한 모델 | null, 미선정 |
| 실행 LLM episode 수 | 0 |
| 성공 / 실패 episode 수 | 0 / 0 (미실행) |
| 평균 step 수 | null |
| 실제 LLM trajectory JSONL | 생성하지 않음 |
| fault injection 단계 준비 | 미완료 |

차단 상태: `../runs_llm/20261103T030310.825180Z_e58d7e88/status.json`.
테스트/키 누락 실행/보존 해시/API 형식 출처: `llm_setup_20261103T025242556726Z/`.
API 요청 형식은 공식 OpenAI Python SDK 소스로 확인했으나 실제 모델 지원을 의미하지 않음.

## 다음 실행 조건

필요한 인증 변수는 `OPENAI_API_KEY`이며 안전하게 직접 환경에 주입해야 함. 키를 채팅이나 저장소에 기록하지 않음.
프록시 secret target에서는 `OPENAI_`가 예약 접두사이므로 그 방식으로 선언하지 않음.
설정 초안에는 OPENAI_API_KEY의 **직접 환경 변수 요구사항 이름만** 저장했으며 값/secret binding은 생성하지 않았음.
`api.openai.com` allowlist와 LLM 시작 지침도 초안 저장 완료. 실제 키 주입/네트워크 반영/게시/새 task 검증은 미완료.
기존 install_script와 baseline 시작 지침은 보존함.

키 설정 후 `check_openai.py models`로 실제 목록을 조회하고, 목록의 후보를 `check_openai.py probe --model <ID>`로 검증해야 함.
모델 목록 자체에는 logprobs 지원 flag가 없으므로 실제 probe에서 logprobs가 반환되어야 함.
이후 `collect_llm.py --model <검증한 ID> --episodes 5 --seed 42` 및 `src.validate_llm`을 실행할 수 있음.
모델 설정은 CLI 또는 선택 후 `ALFWORLD_LLM_MODEL`로 지정 가능하며 현재는 값을 지정하지 않았음.

PDDL baseline은 환경의 privileged expert plan에서 행동을 선택함.
LLM 경로는 공개 관찰·목표·실행 행동 이력·유효 행동 목록만으로 plan/action을 생성하고 성공을 보장하지 않음.
실패도 유효한 연구 데이터로 보존함. memory retrieval, fault injection, LSTM, intervention은 구현하지 않았음.
