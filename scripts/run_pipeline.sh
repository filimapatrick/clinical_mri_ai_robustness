#!/bin/bash
set -e

echo "===================================================="
echo " Running End-to-End Clinical MRI AI Robustness Pipeline"
echo "===================================================="

if [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONPATH=$PYTHONPATH:.

DATA_RAW="/Volumes/MyHDD/ABDN_DATA/Data_from_brainlife/proj-6554f423b094062da63aa4c9"

echo "\n[Step 1/5] Ingesting Brainlife Dataset..."
python src/data/ingest_brainlife.py --data_dir "$DATA_RAW" --output ./data/dataset_catalog.csv

echo "\n[Step 2/5] Preprocessing 3D MRI Scans (N4, RAS, Isotropic Resampling)..."
python src/data/preprocess.py --catalog ./data/dataset_catalog.csv --out_dir ./data/processed --num_workers 4

echo "\n[Step 3/5] Profiling Objective Image Quality Metrics (IQMs)..."
python src/evaluation/metrics.py --profile_iqa --data_dir ./data/processed --out_file ./data/processed/iqa_metrics.csv

echo "\n[Step 4/5] Applying Controlled Progressive Degradations..."
python src/degradation/pipeline.py --input_dir ./data/processed --output_dir ./data/processed/degraded --num_workers 4

echo "\n[Step 5/5] Model Training & Robustness Benchmarking..."
python src/models/baseline.py --catalog ./data/dataset_catalog.csv --processed_dir ./data/processed --task task4 --out_dir ./checkpoints/rf

echo "\n✨ End-to-End Pipeline Completed Successfully!"
