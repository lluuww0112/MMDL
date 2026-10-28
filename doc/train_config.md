# 학습 config 설정 레퍼런스

학습에는 `config/stage/{0,1,2}.yaml` 또는 `config/train.yaml`을 사용합니다.

# History

| Option | Type | Description |
| --- | --- | --- |
| `history_dir` | string | run 디렉터리를 저장할 상위 경로입니다. |
| `seed` | int | 모델/Trainer 난수 시드입니다. 데이터 표본 선택에는 `data.sample_seed`가 우선합니다. |
| `history.load_checkpoint` | `null` or string | 재개할 checkpoint 파일 또는 이전 run 이름입니다. `null`이면 stage 0부터 새로 시작합니다. run 디렉터리는 완료된 stage 다음 단계로 진행하거나 미완료 stage를 재개합니다. |
| `history.save_name` | string | 저장할 checkpoint 디렉터리 이름입니다. |

# Model

| Option | Type | Description |
| --- | --- | --- |
| `path` | string | 기반 Qwen3-VL의 Hugging Face ID 또는 로컬 모델 경로입니다. |
| `revision` | string | 불러올 모델 revision/commit입니다. |
| `dtype` | string | 모델 정밀도입니다. CUDA에서 `bf16`/`fp16`이면 해당 mixed precision Trainer 모드도 사용합니다. |
| `attn_implementation` | string | Transformers attention 구현입니다. `sdpa`는 PyTorch scaled dot-product attention입니다. |

# Adapter

| Option | Type | Description |
| --- | --- | --- |
| `type` | `kradapter` or `none` | `kradapter`이면 지정 Linear layer를 KRAdapter로 교체하고, `none`이면 adapter를 넣지 않습니다. |
| `scaling` | float | KRAdapter scaling 값입니다. 예시 값은 2입니다. |
| `target_modules` | string | 교체할 Linear projection 이름입니다. `[k_proj, v_proj]`는 attention key/value projection만 대상으로 합니다. FFN까지 포함하고 싶은 경우 o_proj을 사용합니다.|

KRAdapter를 사용하는 경우 vision encoder와 llm backbone은 feezing 됩니다. 또한 KRAdpater는 llm backbone에만 적용됩니다.

# Data

| Option | Type | Description |
| --- | --- | --- |
| `train_path` | string | 학습 데이터 위치입니다. 기본 `hf://...` 값은 로컬 Hugging Face cache의 MMFineReason 데이터를 뜻합니다. JSON 파일 경로도 가능합니다. |
| `max_train_samples` | `null` or int | 검증을 통과한 학습 표본에서 무작위로 선택할 최대 개수입니다. `null`이면 전체를 사용합니다. |
| `sample_seed` | int | train/validation subset 선택을 재현하는 시드입니다. |
| `min_len` | `null` or int | `original_answer` 토큰 수(종료 토큰 포함)의 하한입니다. 짧은 응답을 제외합니다. |
| `max_len` | `null` or int | `original_answer` 토큰 수의 상한입니다. 긴 응답을 제외하며, `min_len`보다 작을 수 없습니다. |


# Coconut

| Option | Type | Description |
| --- | --- | --- |
| `enabled` | bool | config에 남아 있는 플래그입니다. 현재 학습 경로는 항상 COCONUT trainer를 사용하므로 이 값만으로 끌 수는 없습니다. |
| `c_thought` | int | reasoning step 하나를 교체할 latent token 수입니다. stage 1 이상에서는 1 이상이어야 합니다. |
| `max_latent_stage` | int | 실행할 마지막 stage 번호입니다. stage 0은 원문 `original_answer`를 예측하고, stage N은 reasoning step 1..N을 latent slot으로 교체합니다. |
| `epochs_per_stage` | int | stage 하나당 학습 epoch 수입니다. |
| `pad_latent_to_max` | bool | generation feature에서 현재 stage 대신 `max_latent_stage`만큼 latent slot을 둘지 정합니다. 현재 train feature와 checkpoint MMMU 추론에는 적용되지 않습니다. |

# Training

| Option | Type | Description |
| --- | --- | --- |
| `optimizer` | string | config에는 `adamw`가 지정되어 있으나 현재 구현은 이 값을 읽지 않고 AdamW를 고정으로 사용합니다. |
| `lr` | 양의 실수 | AdamW 학습률입니다. |
| `weight_decay` | float | AdamW weight decay입니다. COCONUT marker embedding은 decay 없이 학습합니다. |
| `batch_size` | int | GPU 한 장당 micro-batch 크기입니다. |
| `gradient_accumulation_steps` | int | optimizer update 전 누적할 micro-batch 수입니다. 유효 배치 크기는 대략 `batch_size × gradient_accumulation_steps × GPU 수`입니다. |
| `max_grad_norm` | float | gradient clipping 최대 norm입니다. |
| `gradient_checkpointing` | bool | activation을 재계산해 VRAM을 절약합니다. 학습 시간은 증가할 수 있습니다. |
| `num_workers` | int | DataLoader worker 수입니다. |
| `save_every_epochs` | int | config에 남아 있는 저장 간격입니다. 현재 callback은 이 값을 읽지 않고 매 epoch 끝에 checkpoint를 저장합니다. |
| `logging_steps` | int | Trainer log 간격입니다. config에 없으면 10을 사용합니다. |

