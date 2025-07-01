#!/bin/bash
#SBATCH -c 10                                # Number of CPU cores
#SBATCH --gres=gpu:ampere:2                  # Request 2 Ampere GPUs
#SBATCH --job-name=test_msrvtt_cap             # Optional job name
#SBATCH --output=logs/msrvtt_test_cap_%j.out        # STDOUT + STDERR to file (useful for debugging)



### Environment Setup ###
echo "--- ENVIRONMENT SETUP ---"

module load anaconda3
echo "Modules loaded:" 

echo "Activating Conda environment: MALMM"
source activate MALMM

cd $SLURM_SUBMIT_DIR
echo "Current working directory: $(pwd)" 


# Run your command
bash run_scripts/msrvtt/test_cap.sh saved_model/MSRVTT_cap/checkpoint_best.pth
