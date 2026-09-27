#!/usr/bin/env bash
# (선택) 모델/데이터를 pinned revision 으로 미리 받아두기. 오프라인 노드나 경로 고정이 필요할 때 사용.
#   bash scripts/download_assets.sh /data/hf
# 이후: --model_path /data/hf/Qwen3-VL-4B-Instruct --data_path /data/hf/MMMU
set -euo pipefail
DEST="${1:?저장 경로를 인자로 주세요}"
hf download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --local-dir "$DEST/Qwen3-VL-4B-Instruct"
hf download MMMU/MMMU --repo-type dataset --revision 98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68 \
  --local-dir "$DEST/MMMU" --include "*/validation-*" "README.md"
