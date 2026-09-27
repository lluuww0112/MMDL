# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: _(기입 필요)_
- **팀원**: _(기입 필요)_
- **작성일**: 2026-09-27
- **재현 커맨드**: `python eval/qwen/mmmu.py --config config/qwen.yaml --device cuda:1`

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (`ebb281ec70b05090aa6165b016eac8ec08e71b17`) |
| 추론 백엔드 | `eval.model.ProjectQwen3VL (simple)`, attention: `sdpa` |
| 사용 GPU | `cuda:1` (GPU 모델/VRAM은 실행 로그에 기록되지 않음) |
| 실측 peak VRAM | 기록되지 않음 |
| 총 소요 시간 | 505.82초 (약 8분 26초, 900문제) |
| 의존성 | 프로젝트의 `lmms_eval` 실행 환경 |
| 실행 커맨드 | `python eval/qwen/mmmu.py --config config/qwen.yaml --device cuda:1` |
`config/qwen.yaml`이 모델 ID·revision·생성 설정을 제공하고, 스크립트가 `lmms_eval`의 `mmmu_val_qwen` task를 실행한다. 데이터셋은 task 설정의 Hugging Face dataset ID `lmms-lab-encoder/MMMU`이며 split은 `validation`이다.

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문**:

```text
Question: {question}
Answer with the option letter only.
```

- **출처**: 프로젝트의 `mmmu_val_qwen` task 설정 (`qwen3_vl` 형식)
- **선택 이유**: 객관식 MMMU 문항에서 선택지 문자만 생성하도록 지시해 후처리 파싱의 모호성을 줄인다. 개방형 문항에는 별도로 `Please answer the question directly.`를 사용한다.

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---:|
| `do_sample` | `True` |
| `temperature` | `0.7` |
| `top_p` | `0.8` |
| `top_k` | `20` |
| `repetition_penalty` | `1.0` |
| `presence_penalty` | `1.5` |
| `seed` | `3407` (`lmms_eval` seed: 0/1234/1234/1234) |

- **출처**: 이번 재현 실행의 저장된 model arguments. 별도의 공식 recipe 대조는 수행하지 않았다.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---:|
| `max_new_tokens` | `128` |
| 이미지 해상도 처리 | `min_pixels=1,003,520`, `max_pixels=4,014,080` |

**선택 근거**: 저장된 실행 설정을 그대로 사용했다. 높은 최소 해상도는 도표·수식·세부 시각정보가 포함된 MMMU 문항을 보존하는 대신, 추론 시간과 VRAM 사용량을 증가시킬 수 있다.

## 4. 채점(파싱) 방식

- 사용한 파서/로직: `mmmu_process_results` 및 `mmmu_aggregate_results` (프로젝트 `lmms_eval` task 구현)
- 동작 방식 요약: 생성 응답에서 MMMU 답안을 파싱해 정답과 비교하고, 30개 세부 과목의 정확도를 macro average로 집계한다. 실행 결과에는 `mmmu_acc`가 보고됐다.

## 5. 결과

| No. | Subject | Data Num | Acc |
|---|---|---:|---:|
| 1 | Accounting | 30 | 46.67 |
| 2 | Agriculture | 30 | 46.67 |
| 3 | Architecture_and_Engineering | 30 | 33.33 |
| 4 | Art | 30 | 63.33 |
| 5 | Art_Theory | 30 | 76.67 |
| 6 | Basic_Medical_Science | 30 | 70.00 |
| 7 | Biology | 30 | 56.67 |
| 8 | Chemistry | 30 | 26.67 |
| 9 | Clinical_Medicine | 30 | 66.67 |
| 10 | Computer_Science | 30 | 43.33 |
| 11 | Design | 30 | 80.00 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 33.33 |
| 13 | Economics | 30 | 53.33 |
| 14 | Electronics | 30 | 33.33 |
| 15 | Energy_and_Power | 30 | 46.67 |
| 16 | Finance | 30 | 26.67 |
| 17 | Geography | 30 | 46.67 |
| 18 | History | 30 | 73.33 |
| 19 | Literature | 30 | 80.00 |
| 20 | Manage | 30 | 50.00 |
| 21 | Marketing | 30 | 56.67 |
| 22 | Materials | 30 | 36.67 |
| 23 | Math | 30 | 26.67 |
| 24 | Mechanical_Engineering | 30 | 30.00 |
| 25 | Music | 30 | 23.33 |
| 26 | Pharmacy | 30 | 60.00 |
| 27 | Physics | 30 | 40.00 |
| 28 | Psychology | 30 | 73.33 |
| 29 | Public_Health | 30 | 53.33 |
| 30 | Sociology | 30 | 53.33 |
| | **Overall (macro avg)** | **900** | **50.67** |

계산식: `Overall = mean(30개 과목 accuracy)`. 전체 정확도는 결과 JSON의 공식 집계값 `mmmu_acc = 0.50667`을 사용했다.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---:|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 50.67 |
| 차이 (Δ) | -16.73 |

## 7. 격차 분석

공식 수치 대비 16.73%p 낮았다. 이번 실행은 4B instruct checkpoint를 `do_sample=True`와 temperature 0.7로 평가했으므로, 결정적 decoding 또는 공식 평가 recipe와 차이가 있을 수 있다. 또한 prompt·파싱 구현·이미지 전처리·모델 revision 및 실행 환경의 차이가 결과에 영향을 줄 수 있다. 과목별로는 Design/Literature(80.00%), Art_Theory(76.67%), History/Psychology(73.33%)가 강한 반면 Music(23.33%), Chemistry/Finance/Math(각 26.67%), Mechanical_Engineering(30.00%)이 낮아, 계산·과학·공학 중심의 취약성이 전체 격차에 기여했을 가능성이 있다.

## 8. 기타 특이사항 / 한계 (Optional)

- 결과 JSON에는 과목별 aggregate가 없어, 과목별 표는 `20260926_050655_samples_mmmu_val_qwen.jsonl`의 답안 로그를 기준으로 산출했다.
- 답안 문자 직접 비교 기준 합계는 452/900(50.22%)로 JSON의 공식 집계 50.67%(약 456/900)와 4문항 차이가 있다. 최종 overall에는 evaluator가 보고한 공식 집계값을 사용했다. 파서의 세부 fallback/정규화 처리를 추가 점검할 필요가 있다.
