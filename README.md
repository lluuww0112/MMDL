# Multimodal Deep Learning Projects

<p align="center">
  <img src="./asset/image.png" width="40%">
</p>

프로젝트 구성은 다음과 같습니다.   
주요 디렉토리는 config, eval, train 입니다.
```bash
.
├── asset
│
├── config 
│   ├── qwen.yaml # 추론 config
│   ├── train.yaml # 학습 config template
│   └── stage # curriculumn 학습용 config
│       ├── 0.yaml
│       ├── 1.yaml
│       └── 2.yaml
│
├── eval
│   ├── custom
│   │   └── mmmu.py # 커스텀 모델 평가코드
│   └── qwen
│       ├── custom_scoring.py # 커스텀 평가코드
│       └── mmmu.py # 베이스라인 평가 코드
│
├── model
│   ├── model_loader.py 
│   ├── qwen_arch.py # 모델 구현체
│   ├── conversation.py # mmmu pormpt template
│   └── adapter.py # KRAdapter 구현체
│
├── train
│   ├── data_loader.py 
│   ├── prepare_data.py  
│   ├── coconut.py # coconut 프레임워크 구현
│   └── run.sh # 학습 실행
│
├── doc # .md 파일
├── reports # assign 작성본
├── results # 추론결과
└── utils 
```


# Quick Start
### (1) Set-up
   
```bash
python3 -m venv ~/venvs/MMDL 
source ~/venvs/MMDL/bin/activate
pip install -r requirements.txt
```


### (2) Evaluation


```bash
# Baseline
CUDA_VISIBLE_DEVICES=0 python eval/qwen/mmmu.py \
  --config config/qwen.yaml \ # 평가 설정 전달 인자
  --dtype bf16 \
  --backend vllm 
```

```bash
# Custom Model
CUDA_VISIBLE_DEVICES=0 python eval/custom/mmmu.py \
  --config config/qwen.yaml \ # 평가 설정 전달 인자
  --target history/<run-name> # 커스텀 모델 가중치 저장 위치 설정
```

# Train
```bash
# 단일 실행시 MMFineReason 다운로드
python /train/prepare_data.py 

# 학습 실행
bash train/run.sh \
    --model-config ./config/qwen.yaml \
    --train-config ./config/stage/0.yaml \
    --device cuda:0 # 사용 device 선택, cuda:1 등등
```

# More Details
- [Expanding Conversation Template](/doc/conversation.md)   
- [Inference Configure Setting](/doc/inference.md)
- [Train Configure Setting](/doc/train_config.md)
- [Curriculumn Training](/doc/train.md)
- [Custom Dataset](/doc/dataset.md)

# References
1. Albert, P., Zhang, F. Z., Saratchandran, H., van den Hengel, A., & Abbasnejad, E. (2025, October). Towards higher effective rank in parameter-efficient fine-tuning using khatri-rao product. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV) (pp. 1292-1302). IEEE.    
2. Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., & Tian, Y. (2024). Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769.
3. Lin, H., Liu, Z., Zhu, Y., Qin, C., Lin, J., Shang, X., ... & Wu, L. (2026). Mmfinereason: Closing the multimodal reasoning gap via open data-centric methods. arXiv preprint arXiv:2601.21821.


