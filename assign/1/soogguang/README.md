# MMMU-val 평가 파이프라인 (Qwen3-VL-4B-Instruct)

Qwen3-VL-4B-Instruct를 MMMU validation 900문제(30과목)로 평가해요. 명령어 한 줄로 추론 → 채점 → 리포트까지 실행되고, `--model_path`만 바꾸면 파인튜닝 체크포인트를 같은 조건으로 평가할 수 있어요.

- 모델: `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- 데이터: `MMMU/MMMU` @ `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68` (validation)
- 결과 (seed 3개): **62.33 ± 0.68** (공식 발표치 67.4) → `reports/report_ours_3seeds.md`

## 실행

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 기본 평가 (seed 3개, API 키 불필요, RTX 4090 기준 seed당 약 2.5시간)
bash scripts/run_mmmu_eval.sh --mode ours --seeds "42 3407 1234" --out_dir results --gpu_memory_utilization 0.85

# 파인튜닝 체크포인트 평가 (LoRA는 merge한 폴더)
bash scripts/run_mmmu_eval.sh --mode ours --model_path /path/to/merged_ckpt --out_dir results_ft --gpu_memory_utilization 0.85

# 빠른 동작 확인 (과목당 2문제)
bash scripts/run_mmmu_eval.sh --mode ours --seeds 42 --limit 2 --out_dir results_smoke
```

- 결과: `results/ours/report_ours.md` (과목별 표, macro 평균, 공식 발표치 비교, 진단 지표, 사용한 설정 전체)
- 중단되면 `--resume`을 붙여 끝난 과목부터 이어서 실행 (문제별 seed 고정이라 결과 동일)
- `--mode official`: Qwen 공식 코드 방식 재현 (seed 1개 전체 적용, GPT 판정 채점). `OPENAI_API_KEY` 필요

## 구성

| 경로 | 역할 |
|---|---|
| `src/mmmu_common.py` | 고정 revision, 데이터 로딩, 프롬프트 생성 (Qwen 공식 `run_mmmu.py`와 동일 형식) |
| `src/infer.py` | vLLM 추론, `predictions.jsonl` 저장 |
| `src/score_ours.py` | 규칙 기반 채점 (API·무작위 없음). 잘린 응답은 오답 |
| `src/score_official.py` | Qwen 공식 채점 재현 (규칙 → GPT 판정 → 무작위) |
| `src/score.py` / `src/report.py` | 채점 실행, 과목별·분야별 집계와 리포트 생성 |
| `scripts/run_mmmu_eval.sh` | 전체 실행 (추론 → 채점 → 리포트) |
| `scripts/download_assets.sh` | (선택) 모델·데이터를 고정 revision으로 미리 다운로드 |
| `scripts/show_prompt.py` | 챗 템플릿이 적용된 실제 입력 문자열 출력 |
| `scripts/audit_ours_mc.py` | 채점 결과 수동 점검용 샘플 출력 |
| `tests/test_pipeline.py` | 프롬프트 형식·채점 규칙 테스트 (`python -m pytest -q tests`) |
| `reports/` | 실험 결과 리포트 |

## 주요 설정

| 항목 | 값 | 출처 |
|---|---|---|
| 프롬프트 | `Question: … Options: A. … Please select the correct answer from the options above.` (이미지는 텍스트 앞) | Qwen 공식 `evaluation/mmmu/run_mmmu.py` |
| 샘플링 | temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5 | Qwen3-VL 모델 카드 |
| max_new_tokens | 16384 | 모델 카드 VL 권장값 |
| 이미지 해상도 | min 1280×28×28, max 5120×28×28 픽셀 | Qwen 공식 코드 |
| seed | 42, 3407, 1234 (문제별 고정: `seed × 100000 + 문제번호`) | |

실험은 로컬 저장소 커밋 `d5420a8`과 같은 코드로 실행했어요.