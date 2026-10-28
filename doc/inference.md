# Model

| Option | Type | Description |
| --- | --- | --- |
| `path` | string | Hugging Face 모델 ID 또는 로컬 모델 디렉터리입니다. |
| `revision` | string | 내려받을 모델의 revision입니다. 재현을 위해 commit hash로 고정합니다. |
| `loader` | string | 모델 loader import 경로입니다. 현재 MMMU vLLM runner는 이 키를 직접 읽지 않습니다. |
| `device` | string | 기본 단일-device 표기입니다. 현재 MMMU vLLM runner는 이 키를 직접 읽지 않습니다. |
| `device_map` | bool | 자동 device mapping 여부입니다. 현재 MMMU vLLM runner는 이 키를 직접 읽지 않습니다. |
| `vllm_gpu_utilization` | float | vLLM이 KV cache 등에 확보할 GPU 메모리 비율입니다. OOM이면 낮추고, 여유가 있으면 높입니다. |
| `max_model_len` | int | 입력과 생성 결과를 합친 최대 context 길이입니다. 클수록 KV cache 메모리가 증가합니다. |
| `dtype` | string | 가중치/추론 정밀도입니다. GPU가 지원하면 보통 `bf16`을 사용합니다. |

# Evaluation

| Option | Type | Description |
| --- | --- | --- |
| `revision` | string | `MMMU/MMMU` validation split의 dataset revision입니다. 평가 데이터 변경을 막기 위해 고정합니다. |


# Generation

| Option | Type | Description |
| --- | --- | --- |
| `do_sample` | bool | `true`면 샘플링, `false`면 결정적 생성입니다. |
| `temperature` | float | 샘플링 분포의 온도입니다. 낮을수록 보수적이고 높을수록 다양해집니다. |
| `top_p` | float | 누적 확률 상위 토큰만 남기는 nucleus sampling 임계값입니다. |
| `top_k` | int | 확률 상위 K개 토큰만 후보로 남깁니다. |
| `repetition_penalty` | int | 이미 생성한 토큰의 반복을 억제합니다. `1.0`이면 추가 패널티가 없습니다. |
| `presence_penalty` | float | 이미 등장한 토큰의 재선택을 억제합니다. |
| `seed` | int | 생성 샘플링 난수 시드입니다. |
| `sample` | `global` / `per_request` | 시드 적용 방식입니다. `per_request`는 문항마다 고정 시드를 적용하고, `global`은 전역 난수 스트림을 사용합니다. |
| `max_new_tokens` | int | 문항 하나가 생성할 최대 토큰 수입니다. 길수록 시간과 메모리 사용량이 늘어납니다. |

# Prompt

| Option | Type | Description |
| --- | --- | --- |
| `conversation` | string | `model/conversation.py`에 등록된 대화 template 이름입니다. MMMU question/options의 입력 형식을 결정합니다. |

# Image

| Option | Type | Description |
| --- | --- | --- |
| `min_pixels` | int | Qwen processor가 리사이즈할 이미지의 최소 픽셀 예산입니다. |
| `max_pixels` | int | 이미지의 최대 픽셀 예산입니다. 높이면 작은 글자·도표 보존에는 유리하지만 vision token, VRAM, 처리 시간이 증가합니다. |

