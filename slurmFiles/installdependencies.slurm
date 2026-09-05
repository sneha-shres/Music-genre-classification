#!/bin/bash

#SBATCH --job-name=envDependencies
#SBATCH --output=slurmJobs/job.%j.out
#SBATCH --gres=gpu:h100:2
#SBATCH --time=03:00:00

echo "Running on $(hostname):"

# Load the conda environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda create -n MusicTagging python=3.10 -y
conda activate MusicTagging
pip install -r requirements.txt


#Run Script
# python src/features/build_features.py

