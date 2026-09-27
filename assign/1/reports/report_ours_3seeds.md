# MMMU-val 결과 — scoring=ours, runs=seed1234, seed3407, seed42

| No. | Subject | Data Num | seed1234 | seed3407 | seed42 | Acc (mean) | Time (s, mean) |
|---|---|---|---|---|---|---|---|
| 1 | Accounting | 30 | 76.67 | 70.00 | 73.33 | 73.33 | 414.0 |
| 2 | Agriculture | 30 | 50.00 | 46.67 | 43.33 | 46.67 | 17.7 |
| 3 | Architecture_and_Engineering | 30 | 43.33 | 50.00 | 46.67 | 46.67 | 1011.0 |
| 4 | Art | 30 | 56.67 | 60.00 | 56.67 | 57.78 | 224.6 |
| 5 | Art_Theory | 30 | 83.33 | 86.67 | 86.67 | 85.56 | 83.6 |
| 6 | Basic_Medical_Science | 30 | 70.00 | 73.33 | 70.00 | 71.11 | 84.0 |
| 7 | Biology | 30 | 56.67 | 56.67 | 53.33 | 55.56 | 180.2 |
| 8 | Chemistry | 30 | 46.67 | 36.67 | 43.33 | 42.22 | 389.9 |
| 9 | Clinical_Medicine | 30 | 66.67 | 73.33 | 60.00 | 66.67 | 144.1 |
| 10 | Computer_Science | 30 | 66.67 | 53.33 | 70.00 | 63.33 | 236.1 |
| 11 | Design | 30 | 76.67 | 73.33 | 76.67 | 75.56 | 73.3 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 33.33 | 33.33 | 36.67 | 34.44 | 166.3 |
| 13 | Economics | 30 | 86.67 | 83.33 | 83.33 | 84.44 | 54.9 |
| 14 | Electronics | 30 | 53.33 | 43.33 | 33.33 | 43.33 | 575.3 |
| 15 | Energy_and_Power | 30 | 40.00 | 46.67 | 56.67 | 47.78 | 980.8 |
| 16 | Finance | 30 | 63.33 | 66.67 | 66.67 | 65.56 | 343.8 |
| 17 | Geography | 30 | 53.33 | 53.33 | 46.67 | 51.11 | 245.2 |
| 18 | History | 30 | 66.67 | 73.33 | 66.67 | 68.89 | 101.6 |
| 19 | Literature | 30 | 83.33 | 80.00 | 80.00 | 81.11 | 10.6 |
| 20 | Manage | 30 | 63.33 | 66.67 | 70.00 | 66.67 | 158.1 |
| 21 | Marketing | 30 | 80.00 | 86.67 | 86.67 | 84.44 | 181.1 |
| 22 | Materials | 30 | 46.67 | 50.00 | 60.00 | 52.22 | 738.1 |
| 23 | Math | 30 | 60.00 | 60.00 | 60.00 | 60.00 | 336.3 |
| 24 | Mechanical_Engineering | 30 | 30.00 | 50.00 | 40.00 | 40.00 | 797.1 |
| 25 | Music | 30 | 23.33 | 16.67 | 33.33 | 24.44 | 829.8 |
| 26 | Pharmacy | 30 | 66.67 | 83.33 | 80.00 | 76.67 | 181.3 |
| 27 | Physics | 30 | 80.00 | 76.67 | 76.67 | 77.78 | 128.6 |
| 28 | Psychology | 30 | 80.00 | 83.33 | 76.67 | 80.00 | 116.2 |
| 29 | Public_Health | 30 | 80.00 | 86.67 | 90.00 | 85.56 | 245.7 |
| 30 | Sociology | 30 | 63.33 | 60.00 | 60.00 | 61.11 | 21.4 |
| | **Overall (macro avg)** | 900 | 61.56 | 62.67 | 62.78 | **62.33 ± 0.68** | 9077 |

계산식: Overall = mean(30개 과목 accuracy). 과목당 30문제로 균등하므로 900문제 micro 정답률과 같아야 함 (검산: seed1234: micro=61.56, 554/900, seed3407: micro=62.67, 564/900, seed42: micro=62.78, 565/900)
± 는 seed 간 표본 표준편차.

## 공식 수치와 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 62.33 |
| 차이 (Δ) | -5.07 |

## 분야별 정확도

| Category | #Subj | Acc |
|---|---|---|
| Art & Design | 4 | 60.83 |
| Business | 5 | 74.89 |
| Science | 5 | 57.33 |
| Health & Medicine | 5 | 66.89 |
| Humanities & Social Science | 4 | 72.78 |
| Tech & Engineering | 7 | 48.57 |

## 진단 정보 (격차 분석용)

| run | 추출 방식 분포 | finish_reason | 잘린 응답 | 평균/최대 출력 토큰 | 문제유형별 acc | peak VRAM (nvidia-smi) |
|---|---|---|---|---|---|---|
| seed1234 | {'final_answer_pattern': 725, 'truncated': 84, 'leading_option': 43, 'open_final_answer': 43, 'mmmu_open': 5} | {'stop': 816, 'length': 84} | 84 (그중 정답 0) | 2785 / 16384 | multiple-choice: 62.93 (n=847), open: 39.62 (n=53) | 23.4 GB |
| seed3407 | {'final_answer_pattern': 724, 'truncated': 89, 'leading_option': 38, 'mmmu_parser': 3, 'open_final_answer': 41, 'mmmu_open': 5} | {'stop': 811, 'length': 89} | 89 (그중 정답 0) | 2854 / 16384 | multiple-choice: 64.23 (n=847), open: 37.74 (n=53) | 23.3 GB |
| seed42 | {'final_answer_pattern': 720, 'truncated': 85, 'leading_option': 41, 'mmmu_parser': 5, 'open_final_answer': 43, 'mmmu_open': 5, 'failed': 1} | {'stop': 815, 'length': 85} | 85 (그중 정답 0) | 2793 / 16384 | multiple-choice: 64.34 (n=847), open: 37.74 (n=53) | 23.4 GB |

## 사용한 설정 (predictions_meta.json 에서 자동 기록)

```json
{
  "model_path": "Qwen/Qwen3-VL-4B-Instruct",
  "model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
  "data_path": "MMMU/MMMU",
  "data_revision": "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68",
  "cache_dir": null,
  "max_new_tokens": 16384,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "repetition_penalty": 1.0,
  "presence_penalty": 1.5,
  "seed": 1234,
  "seed_mode": "per_request",
  "min_pixels": 1003520,
  "max_pixels": 4014080,
  "max_model_len": 49152,
  "gpu_memory_utilization": 0.85,
  "max_num_seqs": 64,
  "bound_processor_pixels": false,
  "use_cot": false,
  "resume": false,
  "limit_per_subject": 0
}
```
