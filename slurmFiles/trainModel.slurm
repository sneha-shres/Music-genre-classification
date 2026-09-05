#!/bin/bash

#SBATCH --job-name=trainCNN
#SBATCH --output=slurmJobs/job.%j.out
#SBATCH --gres=gpu:v100:1
#SBATCH --time=03:00:00

echo "Running on $(hostname):"

# Load the conda environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate MusicTagging

# Print the config file to log
echo "===== CONFIG FILE CONTENTS ====="
cat src/config.json
echo "================================"

echo "===== TRAIN MODEL CONTENTS ====="
cat src/models/train_model.py

export PYTHONPATH=$PWD
#Run Script
python src/models/predict_model.py

