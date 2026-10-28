# Custom Dataset

학습 config의 `data.train_path`와 `data.val_path`에 자체 JSON 데이터셋 경로를 지정할 수 있습니다. COCONUT 학습은 **샘플 하나당 이미지 한 장**만 지원합니다.

```yaml
# config/my_stage0.yaml
data:
  train_path: ./data/my_dataset/train.json
  val_path: ./data/my_dataset/validation.json  # validation이 필요 없으면 null
  max_train_samples: null
  max_val_samples: null
  sample_seed: 42
  min_len: null
  max_len: null
```

저장소 루트에서 실행한다면 위 상대 경로는 저장소 루트를 기준으로 해석됩니다. 이미지의 상대 경로는 JSON 파일이 있는 디렉터리를 기준으로 해석됩니다.

## 파일 구조 예시

```text
data/my_dataset/
├── train.json
├── validation.json
└── images/
    ├── geometry_001.png
    └── chart_002.jpg
```

`train.json`과 `validation.json`은 **비어 있지 않은 JSON 배열**이어야 합니다.

```json
[
  {
    "image": "images/geometry_001.png",
    "question": "삼각형의 넓이를 구하세요.",
    "original_answer": "밑변은 6이고 높이는 4입니다. 따라서 넓이는 6 × 4 ÷ 2 = 12입니다.",
    "steps": [
      "밑변이 6이고 높이가 4인 것을 확인합니다.",
      "삼각형 넓이 공식인 밑변 × 높이 ÷ 2를 적용합니다."
    ],
    "answer": "12"
  },
  {
    "image": "https://example.com/chart_002.jpg",
    "question": "그래프가 나타내는 증가 추세를 한 문장으로 설명하세요.",
    "original_answer": "값은 관측 기간 동안 전반적으로 증가하며 마지막 구간에서 가장 크게 상승합니다.",
    "steps": [
      "처음 값과 마지막 값을 비교합니다.",
      "구간별 기울기가 양수인지 확인합니다."
    ],
    "answer": "전반적인 증가 추세이며 마지막 구간의 상승폭이 가장 큽니다."
  }
]
```

## 필드 명세

| 필드 | 필수 | 형식 | 사용처 |
| --- | --- | --- | --- |
| `image` | 예 | 문자열 | 이미지 경로 또는 URL입니다. 빈 문자열은 허용되지 않습니다. |
| `question` | 예 | 문자열 | Qwen user message의 텍스트입니다. |
| `answer` | 예 | 문자열 | 최종 답변입니다. stage 1 이상에서 남은 reasoning 뒤에 supervision target으로 붙습니다. |
| `original_answer` | 아니오 | 문자열 | stage 0의 supervision target입니다. 없으면 `answer`를 사용합니다. reasoning을 포함한 전체 teacher response를 넣는 것을 권장합니다. |
| `steps` | 아니오 | 문자열 배열 | 순서가 있는 reasoning step입니다. 없으면 빈 배열을 사용합니다. stage 1 이상에는 실질적으로 필요합니다. |

각 객체는 `image`, `question`, `answer`를 모두 가져야 하며, `question`, `answer`, `original_answer`는 문자열이어야 합니다. `steps`가 있다면 모든 원소가 문자열인 배열이어야 합니다. 다른 추가 필드는 무시됩니다.

## 이미지 경로 규칙

| `image` 값 | 동작 |
| --- | --- |
| `images/a.png` | JSON 파일이 있는 디렉터리를 기준으로 해석합니다. 예: `data/my_dataset/train.json`이면 `data/my_dataset/images/a.png`입니다. |
| `/mnt/datasets/a.png` | 절대 경로를 그대로 사용합니다. |
| `https://...` 또는 `http://...` | URL을 그대로 Qwen image processor에 전달합니다. |

각 row에는 이미지 하나만 넣을 수 있습니다. 여러 이미지 배열, video, `image_1` 같은 별도 필드는 학습 dataloader가 읽지 않습니다. 이미지의 실제 리사이즈 범위는 학습 실행에 전달한 Qwen YAML의 `image.min_pixels`와 `image.max_pixels`가 결정합니다.

## Stage와 `steps`의 관계

| 학습 stage | 데이터에서 사용하는 필드 |
| --- | --- |
| stage 0 | `question`, `image`, `original_answer` |
| stage N (N ≥ 1) | `question`, `image`, `steps`, `answer` |

stage N에서는 첫 N개의 `steps`를 latent token으로 바꾸고, 나머지 step과 `answer`만 정답으로 학습합니다. 따라서 stage 2를 학습할 데이터는 적어도 두 개 이상의 의미 있는 reasoning step을 갖는 것이 좋습니다. step 수가 stage보다 적으면 오류는 나지 않지만, 남은 reasoning supervision 없이 `answer`만 학습될 수 있습니다.

`steps`에는 최종 답변을 중복해 넣지 않는 것을 권장합니다. dataloader는 남은 `steps` 뒤에 항상 `answer`를 붙이므로, 중복하면 같은 답변을 두 번 학습합니다. 각 step은 입력 시 자동으로 줄바꿈이 하나 추가됩니다.

## 길이 필터와 표본 수

`data.min_len`과 `data.max_len`은 tokenizer로 계산한 `original_answer` 길이에 적용됩니다. 범위를 벗어난 row는 오류 없이 제외됩니다. 필터 적용 뒤 남은 row가 없다면 학습이 실패합니다.

커스텀 JSON에서 `max_train_samples`와 `max_val_samples`를 설정하면 파일 순서대로 유효한 row를 읽어 해당 개수에서 멈춥니다. 즉, 이 경우 `sample_seed`로 무작위 subset을 만들지 않습니다. 무작위성을 원하면 JSON을 미리 섞어 저장하거나 필요한 subset을 별도로 생성합니다.

## Train/validation 분리와 실행

validation도 train과 동일한 JSON schema를 사용합니다. stage 1 이상에서 validation loss를 보려면 validation set에도 `steps`와 `answer`를 준비해야 합니다.

```bash
bash train/run.sh \
  --model-config config/qwen.yaml \
  --train-config config/my_stage0.yaml \
  --device cuda:0
```

stage curriculum과 checkpoint 재개 방법은 [Curriculumn Training](train.md), 각 YAML 옵션의 세부 의미는 [Train Configure Setting](train_config.md)을 참고합니다.

## 점검 목록

- JSON 최상위 값이 객체 하나가 아닌 `[{...}, {...}]` 배열인지 확인합니다.
- 상대 이미지 경로가 JSON 파일 기준으로 실제 존재하는지 확인합니다.
- stage 0에는 충분한 `original_answer`, stage 1 이상에는 순서가 있는 `steps`를 넣습니다.
- `min_len`/`max_len`을 사용할 때, 대부분의 데이터가 필터로 제외되지 않는지 확인합니다.
- 긴 이미지나 긴 reasoning으로 VRAM이 부족하면 이미지 pixel 범위, `max_len`, `batch_size`를 함께 낮춥니다.
