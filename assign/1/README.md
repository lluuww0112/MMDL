# MMMU-val 평가 파이프라인 — Qwen3-VL-4B-Instruct

Qwen3-VL-4B-Instruct를 MMMU validation 900문제(30과목 × 30문제)로 평가하는 파이프라인이에요. 명령어 한 줄로 **추론 → 채점 → 리포트**까지 실행되고, `--model_path`만 바꾸면 파인튜닝한 체크포인트를 같은 조건으로 다시 평가할 수 있어요.

| | |
|---|---|
| 모델 | `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`, bf16 |
| 데이터 | `MMMU/MMMU` @ `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`, validation, 900문제 |
| 결과 (seed 3개) | **62.33 ± 0.68** (공식 발표치 67.4) → [`reports/report_ours_3seeds.md`](reports/report_ours_3seeds.md) |
| 환경 | RTX 4090 24GB × 1, vLLM 0.11.0, seed 1회당 약 2.5시간, peak VRAM 23.4 GB |

---

## 실행

저장소 루트에서 실행해요. API 키는 필요 없어요.

```bash
pip install -r assign/1/requirements.txt

bash assign/1/scripts/run_mmmu_eval.sh --mode ours \
  --model_path Qwen/Qwen3-VL-4B-Instruct \
  --data_path MMMU/MMMU \
  --seeds "42 3407 1234" --out_dir results --gpu_memory_utilization 0.85
```

| 인자 | 설명 |
|---|---|
| `--model_path` | HF repo id 또는 로컬 체크포인트 폴더. 파인튜닝 후에는 이 값만 바꿔요 (LoRA는 merge한 폴더) |
| `--data_path` | `MMMU/MMMU` 또는 같은 revision을 받아 둔 로컬 폴더 |
| `--seeds` | 반복 실행할 seed 목록. 결과는 seed별 점수와 평균 ± 표준편차로 나와요 |
| `--out_dir` | 결과 저장 위치 (실행한 폴더 기준 상대경로) |
| `--gpu_memory_utilization` | 24GB GPU에서는 0.85 권장 (0.90에서 OOM 발생) |
| `--resume` | 중단된 실행을 끝난 과목부터 이어서 실행 |
| `--limit N` | 과목당 N문제만 실행 (동작 확인용, 예: `--limit 2`) |

```bash
# 파인튜닝 체크포인트 평가
bash assign/1/scripts/run_mmmu_eval.sh --mode ours --model_path /path/to/merged_ckpt \
  --data_path MMMU/MMMU --out_dir results_ft --gpu_memory_utilization 0.85

# 빠른 동작 확인 (60문제)
bash assign/1/scripts/run_mmmu_eval.sh --mode ours --seeds 42 --limit 2 --out_dir results_smoke

# (선택) 모델·데이터를 고정 revision으로 미리 다운로드
bash assign/1/scripts/download_assets.sh /data/hf
#   → --model_path /data/hf/Qwen3-VL-4B-Instruct --data_path /data/hf/MMMU
```

---

## 파이프라인

```
① 데이터 로딩 ─→ ② 프롬프트 ─→ ③ 응답 생성 (vLLM) ─→ ④ 답 추출·채점 ─→ ⑤ 집계·리포트
   900문제        Qwen 공식 형식    샘플링, 16384토큰      규칙 기반, API 없음    과목별·분야별·macro
```

파인튜닝 전후로 ③의 모델만 바뀌고 나머지는 고정이라, 점수 차이를 모델 변화의 효과로 해석할 수 있어요.

### ② 프롬프트

```
<|im_start|>user
{image_tokens}Question: {question}
Options:
A. {option_A}
B. {option_B}
…
Please select the correct answer from the options above.<|im_end|>
<|im_start|>assistant
```

- 이미지는 모두 텍스트 앞에 번호 순서대로 들어가요 (`{image_tokens}`, 이미지 1장당 `<|vision_start|><|image_pad|><|vision_end|>`).
- 주관식(53문제)은 `Question: {question}`만 들어가요. system 프롬프트는 없어요.
- 실제 입력 문자열 확인: `PYTHONPATH=assign/1/src python assign/1/scripts/show_prompt.py Accounting`

### ③ 생성 설정

| 항목 | 값 |
|---|---|
| 샘플링 | temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0, presence_penalty 1.5 |
| max_new_tokens | 16384 |
| 이미지 해상도 | min_pixels 1280×28×28, max_pixels 5120×28×28 |
| seed | 문제마다 고정 (`seed × 100000 + 문제번호`). 배치 구성이나 중단·재시작과 무관하게 같은 결과 |
| vLLM | max_model_len 49152, max_num_seqs 64 |

### ④ 채점 규칙 (`src/score_ours.py`)

외부 API와 무작위 대체 없이, 같은 응답이면 항상 같은 점수가 나오도록 규칙만으로 채점해요.

1. 출력 길이 제한에 걸려 **잘린 응답은 오답** (최종 답을 내지 못했으므로)
2. **객관식**: 아래 순서로 처음 성공한 규칙을 사용
   - 최종 답 선언 중 **마지막** 것: `\boxed{B}`, `Answer: B`, `The answer is (B)`, `**Final Answer:** B`, `Answer choice: B` 등
   - 응답이 선택지 글자로 시작: `B.`, `(B)`, `**B.**`, `B` (관사 "A …"는 제외)
   - MMMU 공식 `parse_multi_choice_response` (무작위 선택 제거)
   - 모두 실패하면 오답
3. **주관식**: 마지막 `\boxed{…}` 또는 `Final Answer:` 줄이 있으면 그것만 비교, 없으면 MMMU 공식 `parse_open_response` + `eval_open`

검증: 공식 채점(규칙 + GPT 추출) 결과와 비교해 끝까지 답한 객관식 1,500건 중 98.8%가 일치했고, 규칙을 고정한 뒤 새 응답 815건에서 추출 오류는 약 0.4%였어요. 테스트: `python -m pytest -q assign/1/tests`

### ⑤ 출력

`<out_dir>/ours/` 아래에 저장돼요.

| 파일 | 내용 |
|---|---|
| `report_ours.md` | 과목별 표, macro 평균(± seed 간 표준편차), 분야별 점수, 공식 발표치 비교, 진단 지표, 사용한 설정 전체 |
| `seed{N}/predictions.jsonl` | 문제별 프롬프트, 원문 응답, 종료 이유(`stop`/`length`), 출력 토큰 수 |
| `seed{N}/scores_ours.jsonl` | 문제별 추출 답, 사용한 규칙, 정답 여부 |
| `seed{N}/predictions_meta.json` | 실행 인자 전체, 과목별 소요 시간 |
| `env.txt`, `pip_freeze.txt`, `vram_seed{N}.log` | GPU·드라이버·코드 커밋, 패키지 버전, VRAM 사용량 |

---

## 파일 구성

| 경로 | 역할 |
|---|---|
| `src/mmmu_common.py` | 모델·데이터 revision 고정, 데이터 로딩, 프롬프트 생성 |
| `src/infer.py` | vLLM 추론, `predictions.jsonl` 저장 |
| `src/score_ours.py` | 채점 규칙 |
| `src/score.py`, `src/report.py` | 채점 실행, 집계와 리포트 생성 |
| `src/score_official.py` | Qwen 공식 채점(규칙 → GPT 판정 → 무작위) 재현. 비교 실험용 |
| `scripts/run_mmmu_eval.sh` | 전체 실행 (추론 → 채점 → 리포트) |
| `scripts/download_assets.sh` | 모델·데이터 미리 다운로드 (선택) |
| `scripts/show_prompt.py` | 챗 템플릿이 적용된 실제 입력 문자열 출력 |
| `scripts/audit_ours_mc.py` | 채점 결과 수동 점검용 샘플 출력 |
| `tests/test_pipeline.py` | 프롬프트 형식·채점 규칙 테스트 |
| `reports/` | 결과 리포트 |

---

## 출처

| 부분 | 출처 |
|---|---|
| 프롬프트 형식, 이미지 해상도, vLLM 입력 준비 | Qwen 공식 평가 코드 [`QwenLM/Qwen3-VL/evaluation/mmmu/run_mmmu.py`](https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu) (commit 96588727) |
| 샘플링 값 | 같은 폴더의 `infer_instruct.sh`, [Qwen3-VL-4B-Instruct 모델 카드](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| max_new_tokens 16384 | 모델 카드의 VL 권장 출력 길이. 공식 스크립트는 32768이지만, 사전 테스트에서 32768로도 반복 루프가 끝나지 않는 경우가 있어 16384 사용 |
| 채점 | [MMMU 공식 `eval_utils.py`](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py) 기반 자체 구현 |

공식 채점 방식으로 비교하려면 `--mode official`로 실행해요 (seed를 전체에 한 번 적용, GPT 판정 채점, `OPENAI_API_KEY` 필요).
