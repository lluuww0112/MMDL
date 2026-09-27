# MMMU-val 결과 — scoring=official, runs=seed42

| No. | Subject | Data Num | Acc | Time (s) |
|---|---|---|---|---|
| 1 | Accounting | 30 | 76.67 | 314.9 |
| 2 | Agriculture | 30 | 46.67 | 16.0 |
| 3 | Architecture_and_Engineering | 30 | 40.00 | 888.9 |
| 4 | Art | 30 | 60.00 | 202.1 |
| 5 | Art_Theory | 30 | 90.00 | 21.6 |
| 6 | Basic_Medical_Science | 30 | 76.67 | 201.6 |
| 7 | Biology | 30 | 60.00 | 40.4 |
| 8 | Chemistry | 30 | 56.67 | 396.0 |
| 9 | Clinical_Medicine | 30 | 70.00 | 204.2 |
| 10 | Computer_Science | 30 | 76.67 | 294.9 |
| 11 | Design | 30 | 76.67 | 13.3 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 36.67 | 215.0 |
| 13 | Economics | 30 | 80.00 | 60.3 |
| 14 | Electronics | 30 | 66.67 | 735.7 |
| 15 | Energy_and_Power | 30 | 46.67 | 944.9 |
| 16 | Finance | 30 | 73.33 | 329.1 |
| 17 | Geography | 30 | 56.67 | 252.1 |
| 18 | History | 30 | 63.33 | 204.7 |
| 19 | Literature | 30 | 83.33 | 11.0 |
| 20 | Manage | 30 | 73.33 | 80.0 |
| 21 | Marketing | 30 | 80.00 | 214.6 |
| 22 | Materials | 30 | 63.33 | 591.8 |
| 23 | Math | 30 | 60.00 | 298.1 |
| 24 | Mechanical_Engineering | 30 | 46.67 | 793.3 |
| 25 | Music | 30 | 26.67 | 749.8 |
| 26 | Pharmacy | 30 | 70.00 | 246.3 |
| 27 | Physics | 30 | 70.00 | 219.2 |
| 28 | Psychology | 30 | 86.67 | 23.0 |
| 29 | Public_Health | 30 | 76.67 | 251.4 |
| 30 | Sociology | 30 | 53.33 | 16.7 |
| | **Overall (macro avg)** | 900 | **64.78** | 8840 |

계산식: Overall = mean(30개 과목 accuracy). 과목당 30문제로 균등하므로 900문제 micro 정답률과 같아야 함 (검산: seed42: micro=64.78, 583/900)

## 공식 수치와 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 64.78 |
| 차이 (Δ) | -2.62 |

## 분야별 정확도

| Category | #Subj | Acc |
|---|---|---|
| Art & Design | 4 | 63.33 |
| Business | 5 | 76.67 |
| Science | 5 | 60.67 |
| Health & Medicine | 5 | 66.00 |
| Humanities & Social Science | 4 | 71.67 |
| Tech & Engineering | 7 | 55.24 |

## 진단 정보 (격차 분석용)

| run | 추출 방식 분포 | finish_reason | 잘린 응답 | 평균/최대 출력 토큰 | 문제유형별 acc | peak VRAM (nvidia-smi) |
|---|---|---|---|---|---|---|
| seed42 | {'rule': 390, 'judge': 418, 'random': 92} | {'stop': 814, 'length': 86} | 86 (그중 정답 13) | 2813 / 16384 | multiple-choice: 63.40 (n=847), open: 86.79 (n=53) | 23.5 GB |

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
  "seed": 42,
  "seed_mode": "global",
  "min_pixels": 1003520,
  "max_pixels": 4014080,
  "max_model_len": 49152,
  "gpu_memory_utilization": 0.9,
  "max_num_seqs": 64,
  "bound_processor_pixels": false,
  "use_cot": false,
  "limit_per_subject": 0
}
```

판정 모델: `gpt-3.5-turbo-0125`
