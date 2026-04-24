#!/bin/bash
# Attend Stage 1, sauvegarde S3, lance Stage 2, sauvegarde S3
set -e
PROJ="/home/onyxia/work/Named_Entity_Extraction_Archelec_Corpus/projet_archelec"
MODELS="/home/onyxia/archelec_models"
S3="s3/oualy/archelec_models"

echo "[$(date)] En attente de Stage 1 (PID $1)..."
wait $1
echo "[$(date)] Stage 1 terminé. Sauvegarde S3..."
mc cp -r "$MODELS/stage1_model" "$S3/stage1_model"
echo "[$(date)] stage1_model sauvegardé sur S3."

echo "[$(date)] Lancement Stage 2..."
cd "$PROJ"
uv run python scripts/run_stage2.py > /home/onyxia/stage2_train.log 2>&1
echo "[$(date)] Stage 2 terminé. Sauvegarde S3..."
mc cp -r "$MODELS/stage2_model" "$S3/stage2_model"
echo "[$(date)] stage2_model sauvegardé sur S3."
echo "[$(date)] Tout est sauvegardé. Pour restaurer : mc cp -r $S3 $MODELS"
