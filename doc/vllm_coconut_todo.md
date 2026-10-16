# vLLM COCONUT 평가 구현 TODO

## 현재 상태

- `model/qwen_arch.py`에 학습/Transformers 경로용 COCONUT 구현이 있다.
- `coconut_latent_layout()`은 latent token의 연속성·배치 정렬을 검증하는 공용 함수다.
- `eval/custom/model.py`는 `CoconutQwen3VLForConditionalGeneration`을 vLLM registry에 등록한다.
- 아직 vLLM의 prefill 실행을 latent 단위로 분할하지 않았으므로, registry 모델만으로는 COCONUT latent-feedback 추론이 수행되지 않는다.

## 구현할 항목

1. vLLM model-runner adapter 추가
   - vLLM이 전달하는 flattened `input_ids`, `positions`, request별 sequence 길이 및 attention metadata를 읽는다.
   - 각 prefill request에서 COCONUT latent 위치를 복원하여 `coconut_latent_layout()`으로 검증한다.
   - latent가 없는 request는 upstream Qwen3-VL prefill 경로를 그대로 사용한다.
   - latent가 있는 request는 첫 latent 직전까지 prefill한 후, 매 latent마다 직전 token의 hidden state를 latent embedding에 주입하고 다음 구간을 실행한다.
   - image/video tensor는 첫 prefill 구간에만 전달한다.

2. vLLM KV-cache 및 attention metadata 처리
   - 각 분할 구간이 원 요청의 KV-cache slot과 position을 그대로 공유하도록 attention metadata를 slice/rebuild한다.
   - chunked prefill, CUDA graph, pipeline/tensor parallelism이 COCONUT 분할 prefill과 함께 동작하는지 확인한다.
   - 초기 구현은 `tensor_parallel_size=1`, `max_num_seqs=1`, chunked prefill 비활성화로 제한하고, 동작 검증 뒤 병렬·batch 지원을 확장한다.

3. KRAdapter 체크포인트 로더
   - `--target`이 checkpoint 파일 또는 history 실행 디렉터리를 받도록 한다.
   - 실행 디렉터리는 `metadata.json`의 `latest_checkpoint`을 우선 선택한다.
   - 인접한 `config.yaml`에서 `adapter`와 `coconut` 설정을 읽는다.
   - KRAdapter state (`weight`, `v_weight`)를 vLLM Qwen projection에 적용한다.
   - 첫 구현은 adapter update를 base projection weight에 merge하는 방식으로 구현한다. 이때 원본 checkpoint의 start/end latent embedding도 함께 복원한다.

4. tokenizer / model config 준비
   - `<|start-latent|>`, `<|latent|>`, `<|end-latent|>`를 tokenizer에 등록한다.
   - vLLM model config의 vocabulary 크기와 embedding weight 크기가 checkpoint와 동일한지 검증한다.
   - checkpoint stage와 `c_thought`, `pad_latent_to_max` 값으로 평가 prompt 뒤에 넣을 latent 수를 계산한다.

5. `eval/custom/mmmu.py`를 vLLM 경로로 전환
   - 시작 시 `eval.custom.model.register()`를 호출한다.
   - `LLM(..., model=..., load_format=<custom loader>)`으로 registry 모델과 target checkpoint loader를 사용한다.
   - 기존 `eval/qwen/mmmu.py`의 MMMU record 생성, image 처리, prediction JSONL 및 `eval/qwen/custom_scoring.py` 채점을 재사용한다.
   - 결과 metadata에 target, checkpoint stage, latent 수, adapter 설정, vLLM 설정을 기록한다.

6. 검증
   - latent 없는 prompt에서 custom vLLM 모델 logits가 upstream vLLM Qwen3-VL과 일치하는지 확인한다.
   - 작은 단일 이미지 prompt에서 Transformers COCONUT과 vLLM COCONUT의 prefill logits 및 greedy 첫 토큰이 일치하는지 확인한다.
   - stage 0, stage N, `pad_latent_to_max=true`, 중단 후 재개 checkpoint를 각각 확인한다.
   - MMMU smoke test(`--limit-per-subject 1`) 후 전체 validation 평가를 실행한다.

## 주의 사항

- 단순히 KRAdapter만 merge하거나 latent token을 일반 token으로 전달하면 COCONUT의 hidden-state feedback이 빠지므로 올바른 COCONUT 평가가 아니다.
- 기존 vLLM Qwen3-VL runner는 flattened token 기반이다. Transformers의 `past_key_values`를 그대로 호출하는 방식은 사용할 수 없다.
- model-runner adapter 구현 전에는 custom registry 모델을 전체 MMMU 평가에 사용하지 않는다.
