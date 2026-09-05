#!/bin/bash

#SBATCH --job-name=buildFeatures
#SBATCH --output=slurmJobs/job.%j.out
#SBATCH --gres=gpu:a30:1
#SBATCH --time=01:00:00

echo "Running on $(hostname):"

# Load the conda environment
module load Anaconda3
eval "$(conda shell.bash hook)"
# conda remove -n MusicTagging
# conda create -n MusicTagging
conda activate MusicTagging
# pip install -r requirements.txt


#Run Script
python src/features/build_features.py

