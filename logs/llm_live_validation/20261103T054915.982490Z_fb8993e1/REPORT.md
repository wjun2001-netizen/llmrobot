# OpenAI 실제 연결 검증 결과

검증일: 2026-11-03 (Asia/Seoul).

- 현재 프로세스에 `OPENAI_API_KEY`가 존재함을 확인. 값은 출력·저장하지 않음.
- 공식 `GET https://api.openai.com/v1/models` 성공. 실제 계정 접근 모델 목록을 별도 보존.
- 목록에 있는 `gpt-4.1-mini-2025-04-14`를 경량 text agent 후보로 선택해 Responses + Structured Outputs + output logprobs probe를 수행.
- probe가 HTTP 429로 거부됨. 추가 진단에서 API 오류 type이 `insufficient_quota`임을 확인.
- 최소 요청에서도 같은 오류 유형이 확인되었으므로 모델별 지원 여부나 ALFWorld 코드의 문제로 판단하지 않음. 다른 모델로 우회하거나 동일 요청을 반복하지 않음.
- 인증 정보가 등록되지 않았던 이전 상태와 구분: 현재 키 전달과 모델 목록 조회는 성공했으며, 생성 요청의 API quota가 외부 blocker임.

## 실제 실행 통계

| 항목 | 결과 |
| --- | --- |
| 검토한 후보 | gpt-4.1-mini-2025-04-14 |
| 검증 완료하여 사용한 모델 | null |
| Responses 생성 / logprobs 반환 검증 | 미완료, insufficient_quota로 차단 |
| 실제 LLM episode | 0 |
| 성공 / 실패 episode | 0 / 0 (미실행; 성능 결과 아님) |
| 평균 step | null |
| 새 trajectory JSONL | 없음 |

1개 episode의 저장 검증이 선행 조건이므로 5개 batch 실행도 시작하지 않았음.
이전 PDDL baseline 및 이전 LLM 차단 로그는 해시 비교로 보존 확인.
fault injection, LSTM, intervention 및 모델 변경/설치는 수행하지 않음.

## 보존된 증거

- 모델 목록: `../../openai_checks/20261103T054854.480242Z_35813df6/models.json`
- probe 상태: `../../openai_checks/20261103T054907.469089Z_49fc3701/status.json`
- 안전하게 추출한 quota 진단: `api_diagnosis.json` (오류 본문·인증 헤더·키 값 미저장)
- 실행 통계 및 기존 파일 해시 비교: `status.json`, `preservation_before.json`

## 재개 조건과 순서

OpenAI API 프로젝트의 결제 상태, 잔여 크레딧, 사용 한도를 확인하고 quota를 복구해야 함.
현재 키가 잘못됐다는 증거는 없으므로 새로운 키 등록을 요구하지 않음.
quota 복구 후 다음 순서로 재개:

1. 같은 후보의 실제 Responses/logprobs probe 재시도.
2. 지원이 확인되면 후보 모델명을 기록하고 LLM episode 1개 실행.
3. plan/action/admissibility/reward/done/success/logprobs의 실제 저장값 검증.
4. 정상 완료를 확인한 뒤 5개 episode batch 실행 및 성공·실패 원본 보존.
5. 디스크 JSONL 재검증 및 성공/실패 수, 평균 step, logprobs 수집 결과 보고.
