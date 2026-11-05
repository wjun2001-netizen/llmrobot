# 실행 결과

검증일: 2026-11-03 (Asia/Seoul). run 식별자는 UTC 사용.

- 환경: Debian 13.6, Python 3.12.14, venv pip 25.0.1
- 최종 패키지: ALFWorld 0.4.2, TextWorld 1.6.2, fast-downward-textworld 20.6.4
- text-only, ALFWorld PDDL expert, seed 42, max_steps 50, train/pick_and_place_simple

| 검증 실행 | episode 수 | 성공 | 실패 | step 수 |
| --- | ---: | ---: | ---: | --- |
| 초기 직접 smoke (metadata 보완 후) | 1 | 1 | 0 | 4 |
| 수집기 1개 검증 (TW 1.7.0) | 1 | 1 | 0 | 3 |
| 최초 batch (TW 1.7.0) | 5 | 5 | 0 | 3, 3, 3, 4, 4 |
| 최종 기준 batch (TW 1.6.2) | 5 | 5 | 0 | 3, 3, 3, 4, 4 |
| 최종 반복 batch (TW 1.6.2) | 5 | 5 | 0 | 3, 3, 3, 4, 4 |

초기 데이터 누락으로 reset에 실패한 시도 1회는 위 성공 실행 수에 포함하지 않음.
총 17회 성공 실행, 중복을 제외한 게임 6개. 최종 정상 corpus의 서로 다른 게임은 5개.

기준 JSONL: `../runs/20261103T024323.126935Z_6dd6830c/trajectories.jsonl`

5개 episode의 최종 done/success 모두 true. 중간 step의 success 값도 빠짐없이 기록됨.
필수 필드 누락 0, null 0, 빈 observation/action 각각 0.
수집 불가능한 요청 필드 없음. `plan`은 PDDL expert의 계획이며 LLM reasoning은 제공되지 않음.

동일 seed 반복 JSONL의 SHA256이 기준 파일과 일치함:
`37ee64ecc269065298181dbd33837d213eedf5fd97475288f56206bc07de598b`.
기존 모든 raw 파일은 별도 run 디렉터리에 유지됨.
`pip check` 통과, `setup.sh` 반복 실행 통과, JSONL 독립 재검증 통과, 검증기 테스트 8개 통과.

주요 오류는 metadata 누락과 TextWorld 1.7.0 wheel metadata 결함.
공식 JSON 추가 및 TextWorld 1.6.2 고정으로 해결. 실패한 소스 빌드 로그도 보존.
세부 정보 및 재실행 명령은 README 참고.
