#!/bin/bash
#SBATCH -c 8                                # Number of CPU cores
#SBATCH --gres=gpu:ampere:2                  # Request 2 Ampere GPUs
#SBATCH --job-name=train_msvd_qa             # Optional job name
#SBATCH --output=/home/davidg3/MA-LMM_logs/train/msvd_train_qa_%j.out        # STDOUT + STDERR to file (useful for debugging)


### Environment Setup ###
echo "--- ENVIRONMENT SETUP ---"

module load anaconda3
echo "Modules loaded:" 

echo "Activating Conda environment: MALMM"
source activate MALMM
echo "Environment loaded."

echo "Current working directory: $(pwd)" 


# Run your command
bash run_scripts/msvd/train_qa.sh

