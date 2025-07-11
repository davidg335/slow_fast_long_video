#!/bin/bash
#SBATCH -c 8                                # Number of CPU cores
#SBATCH --gpus=2                                # Request 2 GPUs (any type for now)
#SBATCH -C 'gmem24|gmem32|gmemT48|gmem48|gmem80'    # GPU memory constraint (matches -C)
#SBATCH --job-name=train_msvd_qa             # Optional job name
#SBATCH --output=logs/train/msvd_test_qa_%j.out        # STDOUT + STDERR to file (useful for debugging)


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

