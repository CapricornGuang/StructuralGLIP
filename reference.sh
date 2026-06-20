#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-configs/pretrain/glip_Swin_T_O365_GoldG_polyp_cvc300.yaml}"
TASK_CONFIG="${TASK_CONFIG:-$CONFIG_FILE}"
OUTPUT_DIR="${OUTPUT_DIR:-output/polyp-test}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth}"
JSON_FILE="${JSON_FILE:-blip_json/cvc300_val_noloc.json}"

for required_file in "$CONFIG_FILE" "$TASK_CONFIG" "$MODEL_CHECKPOINT" "$JSON_FILE"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Missing required file: $required_file" >&2
        exit 1
    fi
done

python test.py --json "$JSON_FILE" \
      --config-file "$CONFIG_FILE" --weight "$MODEL_CHECKPOINT" \
      --task_config "$TASK_CONFIG" \
      OUTPUT_DIR "$OUTPUT_DIR" \
      TEST.IMS_PER_BATCH 2 SOLVER.IMS_PER_BATCH 2 \
      TEST.EVAL_TASK detection \
      DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
      DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False \
      DATASETS.USE_OVERRIDE_CATEGORY True \
      DATASETS.USE_CAPTION_PROMPT True
