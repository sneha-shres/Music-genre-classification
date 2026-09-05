#!/bin/bash

#SBATCH --job-name=trial
#SBATCH --output=slurmJobs/job.%j.out
#SBATCH --gres=gpu:h100>1
#SBATCH --time=00:10:00

echo "Running on $(hostname):"

# Load the conda environment

module load Anaconda3

# Load the conda environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate MusicTagging


#Run Script
python src/data/make_dataset.py
