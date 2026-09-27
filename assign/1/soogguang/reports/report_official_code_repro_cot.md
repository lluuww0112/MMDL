# MMMU-val 결과 — scoring=official, runs=seed42

| No. | Subject | Data Num | Acc | Time (s) |
|---|---|---|---|---|
| 1 | Accounting | 30 | 76.67 | 411.5 |
| 2 | Agriculture | 30 | 46.67 | 29.8 |
| 3 | Architecture_and_Engineering | 30 | 56.67 | 905.5 |
| 4 | Art | 30 | 56.67 | 237.8 |
| 5 | Art_Theory | 30 | 80.00 | 21.5 |
| 6 | Basic_Medical_Science | 30 | 83.33 | 207.3 |
| 7 | Biology | 30 | 46.67 | 281.3 |
| 8 | Chemistry | 30 | 56.67 | 285.8 |
| 9 | Clinical_Medicine | 30 | 66.67 | 209.4 |
| 10 | Computer_Science | 30 | 73.33 | 290.1 |
| 11 | Design | 30 | 80.00 | 12.1 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 33.33 | 59.5 |
| 13 | Economics | 30 | 90.00 | 43.2 |
| 14 | Electronics | 30 | 73.33 | 595.8 |
| 15 | Energy_and_Power | 30 | 53.33 | 894.9 |
| 16 | Finance | 30 | 76.67 | 301.3 |
| 17 | Geography | 30 | 60.00 | 250.6 |
| 18 | History | 30 | 80.00 | 206.6 |
| 19 | Literature | 30 | 83.33 | 11.0 |
| 20 | Manage | 30 | 76.67 | 218.2 |
| 21 | Marketing | 30 | 80.00 | 108.4 |
| 22 | Materials | 30 | 43.33 | 686.3 |
| 23 | Math | 30 | 63.33 | 320.7 |
| 24 | Mechanical_Engineering | 30 | 40.00 | 798.3 |
| 25 | Music | 30 | 33.33 | 1182.6 |
| 26 | Pharmacy | 30 | 83.33 | 244.3 |
| 27 | Physics | 30 | 83.33 | 220.9 |
| 28 | Psychology | 30 | 80.00 | 65.7 |
| 29 | Public_Health | 30 | 80.00 | 252.8 |
| 30 | Sociology | 30 | 53.33 | 17.4 |
| | **Overall (macro avg)** | 900 | **66.33** | 9371 |

계산식: Overall = mean(30개 과목 accuracy). 과목당 30문제로 균등하므로 900문제 micro 정답률과 같아야 함 (검산: seed42: micro=66.33, 597/900)

## 공식 수치와 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 66.33 |
| 차이 (Δ) | -1.07 |

## 분야별 정확도

| Category | #Subj | Acc |
|---|---|---|
| Art & Design | 4 | 62.50 |
| Business | 5 | 80.00 |
| Science | 5 | 62.00 |
| Health & Medicine | 5 | 69.33 |
| Humanities & Social Science | 4 | 74.17 |
| Tech & Engineering | 7 | 55.24 |

## 진단 정보 (격차 분석용)

| run | 추출 방식 분포 | finish_reason | 잘린 응답 | 평균/최대 출력 토큰 | 문제유형별 acc | peak VRAM (nvidia-smi) |
|---|---|---|---|---|---|---|
| seed42 | {'rule': 300, 'random': 96, 'judge': 504} | {'stop': 811, 'length': 89} | 89 (그중 정답 14) | 2988 / 16384 | multiple-choice: 65.05 (n=847), open: 86.79 (n=53) | 22.4 GB |

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
  "gpu_memory_utilization": 0.85,
  "max_num_seqs": 64,
  "bound_processor_pixels": false,
  "use_cot": true,
  "resume": true,
  "limit_per_subject": 0
}
```

판정 모델: `gpt-3.5-turbo-0125`
