#!/bin/bash
#SBATCH -c 10                                # Number of CPU cores
#SBATCH --gres=gpu:ampere:2                  # Request 2 Ampere GPUs
#SBATCH --job-name=test_msvd_qa             # Optional job name
#SBATCH --output=logs/msvd_test_qa_%j.out        # STDOUT + STDERR to file (useful for debugging)


### Environment Setup ###
echo "--- ENVIRONMENT SETUP ---"

module load anaconda3
echo "Modules loaded:" 

echo "Activating Conda environment: MALMM"
source activate MALMM
echo "Environment loaded."

echo "Current working directory: $(pwd)" 


# Run your command
bash "/home/davidg3/MA-LMM/run_scripts/msvd/test_qa.sh" "/home/davidg3/MA-LMM/saved_model/MSVD_qa/checkpoint_best.pth"
