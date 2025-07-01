#!/bin/bash
#SBATCH -c 10                               # Number of CPU cores
#SBATCH --gres=gpu:ampere:2                  # Request 2 Ampere GPUs
#SBATCH --job-name=test_msrvtt_qa             # Optional job name
#SBATCH --output=logs/msrvtt_test_qa_%j.out        # STDOUT + STDERR to file (useful for debugging)


### Environment Setup ###
echo "--- ENVIRONMENT SETUP ---"

module load anaconda3
echo "Modules loaded:" 

echo "Activating Conda environment: MALMM"
source activate MALMM
echo "Environment loaded"

cd $SLURM_SUBMIT_DIR
echo "Current working directory: $(pwd)" 


# Run your command
bash run_scripts/msrvtt/test_qa.sh saved_model/MSRVTT_qa/checkpoint_best.pth
