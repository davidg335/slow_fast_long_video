#!/bin/bash
#SBATCH -c 10                                # Number of CPU cores
#SBATCH --gres=gpu:ampere:2                  # Request 2 Ampere GPUs
#SBATCH --job-name=test_breakfast             # Optional job name
#SBATCH --output=logs/breakfast_%j.out        # STDOUT + STDERR to file (useful for debugging)


### Environment Setup ###
echo "--- ENVIRONMENT SETUP ---"

module load anaconda3
echo "Modules loaded:" 

echo "Activating Conda environment: MALMM"
source activate MALMM
echo "Environment loaded."

echo "Current working directory: $(pwd)" 


# Run your command
bash run_scripts/breakfast/test.sh saved_model/Breakfast/checkpoint_best.pth
