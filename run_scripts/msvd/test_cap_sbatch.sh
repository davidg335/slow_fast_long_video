#!/bin/bash
#SBATCH -c 10                                # Number of CPU cores
#SBATCH --gres=gpu:ampere:2                  # Request 2 Ampere GPUs
#SBATCH --job-name=test_msvd_cap             # Optional job name
#SBATCH --output=logs/msvd_test_cap_%j.out        # STDOUT + STDERR to file (useful for debugging)

# Activate any environment you need here (conda, module load, etc.)
module load anaconda3
source activate MALMM

# Run your command
bash run_scripts/msvd/test_cap.sh saved_model/MSVD_cap/checkpoint_best.pth
