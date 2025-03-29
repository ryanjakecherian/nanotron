#!/bin/bash
#SBATCH --job-name=tiny_llama_train    # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --gpus=2                        # Number of GPUs
#SBATCH --cpus-per-task=8               # CPU cores per task
#SBATCH --mem=32G                       # Memory allocation
#SBATCH --time=24:00:00                 # Time limit (HH:MM:SS)
#SBATCH --output=slurm-%j.out           # Standard output log
#SBATCH --error=slurm-%j.err            # Error log file

# Load necessary modules (if needed)
module load cuda/11.8   # Adjust based on your environment
module load python/3.9  # Adjust based on your environment

# Activate virtual environment (if applicable)
source ~/myenv/bin/activate

# Run the command
python3 config_tiny_llama.py && (cd ..; CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 run_train.py --config-file examples/config_tiny_llama.yaml)