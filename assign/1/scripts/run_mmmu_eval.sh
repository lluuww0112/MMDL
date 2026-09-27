#!/usr/bin/env bash
# MMMU-val 평가 한 번에 실행: 추론(vLLM) -> 채점 -> report.md
#
# 사용 예:
#   bash scripts/run_mmmu_eval.sh --mode official --model_path Qwen/Qwen3-VL-4B-Instruct --data_path MMMU/MMMU --out_dir results
#   bash scripts/run_mmmu_eval.sh --mode ours     --model_path /path/to/merged_ckpt     --data_path /data/MMMU  --out_dir results
#   bash scripts/run_mmmu_eval.sh --mode ours --limit 2 --out_dir results_smoke     # 스모크 테스트 (과목당 2문제)
#
#   official : Qwen 공식 코드 재현 — seed 42 1회(global seed), GPT 판정기 채점 (OPENAI_API_KEY 필요)
#   ours     : 같은 프롬프트/샘플링/해상도 — seed 3개(문제별 seed 고정), API 없는 결정론적 채점
#   --use_cot : Qwen 공식 --use-cot 문구를 붙여 실행 (결과 폴더 <mode>_cot)
#   --resume  : 중단된 실행을 이어서 (끝난 과목은 건너뜀, 로그는 이어쓰기)
#   추가 infer 옵션은 환경변수로: INFER_EXTRA="--bound_processor_pixels" bash scripts/run_mmmu_eval.sh ...
#   두 모드 모두 predictions 는 'ours' 채점으로도 한 번 더 채점해 둔다 (채점 방식 효과 분리용).
set -euo pipefail

MODE=""; MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"; DATA_PATH="MMMU/MMMU"; OUT_DIR="results"
SEEDS=""; LIMIT=0; JUDGE_MODEL="gpt-3.5-turbo-0125"; MAX_NEW_TOKENS=16384; MAX_MODEL_LEN=49152
GPU_UTIL=0.90; CACHE_DIR=""; TAG=""; USE_COT=0; RESUME=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --model_path) MODEL_PATH="$2"; shift 2;;
    --data_path) DATA_PATH="$2"; shift 2;;
    --cache_dir) CACHE_DIR="$2"; shift 2;;
    --out_dir) OUT_DIR="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    --judge_model) JUDGE_MODEL="$2"; shift 2;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --max_model_len) MAX_MODEL_LEN="$2"; shift 2;;
    --gpu_memory_utilization) GPU_UTIL="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --use_cot) USE_COT=1; shift 1;;
    --resume) RESUME=1; shift 1;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done
[[ "$MODE" == "official" || "$MODE" == "ours" ]] || { echo "--mode official|ours 필수"; exit 1; }

if [[ "$MODE" == "official" ]]; then SEED_MODE=global;      SEEDS=${SEEDS:-"42"}
else                                  SEED_MODE=per_request; SEEDS=${SEEDS:-"42 3407 1234"}; fi
if [[ "$MODE" == "official" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "official 모드는 GPT 판정기를 쓰므로 OPENAI_API_KEY 가 필요합니다"; exit 1; fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "$USE_COT" == 1 && -z "$TAG" ]]; then TAG="cot"; fi   # 예: results/official_cot
RUN_DIR="$OUT_DIR/${MODE}${TAG:+_$TAG}"
mkdir -p "$RUN_DIR"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

# 재현성 기록: 환경/하드웨어/코드 버전
{ echo "date: $(date -Iseconds)"; echo "host: $(hostname)"; echo "cmd: $0 $*";
  echo "git: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo n/a)";
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
  python -V; } > "$RUN_DIR/env.txt"
pip freeze > "$RUN_DIR/pip_freeze.txt" 2>/dev/null || true

for SEED in $SEEDS; do
  SD="$RUN_DIR/seed$SEED"; mkdir -p "$SD"
  echo "=== [$MODE] seed=$SEED  -> $SD"
  # peak VRAM 측정: 1초 간격 nvidia-smi 로깅
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "$RUN_DIR/vram_seed$SEED.log" 2>/dev/null &
  SMI_PID=$!
  python "$ROOT/src/infer.py" \
    --model_path "$MODEL_PATH" --data_path "$DATA_PATH" ${CACHE_DIR:+--cache_dir "$CACHE_DIR"} \
    --output_file "$SD/predictions.jsonl" \
    --seed "$SEED" --seed_mode "$SEED_MODE" \
    --max_new_tokens "$MAX_NEW_TOKENS" --max_model_len "$MAX_MODEL_LEN" \
    --gpu_memory_utilization "$GPU_UTIL" --limit_per_subject "$LIMIT" $([[ "$USE_COT" == 1 ]] && echo --use_cot) $([[ "$RESUME" == 1 ]] && echo --resume) ${INFER_EXTRA:-} 2>&1 | tee $([[ "$RESUME" == 1 ]] && echo -a) "$SD/infer.log"
  kill $SMI_PID 2>/dev/null || true

  if [[ "$MODE" == "official" ]]; then
    python "$ROOT/src/score.py" --pred_file "$SD/predictions.jsonl" --method official --judge_model "$JUDGE_MODEL"
  fi
  python "$ROOT/src/score.py" --pred_file "$SD/predictions.jsonl" --method ours
done

[[ "$MODE" == "official" ]] && python "$ROOT/src/report.py" --run_dir "$RUN_DIR" --method official
python "$ROOT/src/report.py" --run_dir "$RUN_DIR" --method ours
echo "완료: $RUN_DIR/report_*.md"