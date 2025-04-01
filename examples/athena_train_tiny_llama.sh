#!/bin/bash
#SBATCH --job-name=tiny_llama_train    # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --gres=gpu:2                   # Number of GPUs
#SBATCH --cpus-per-task=8               # CPU cores per task
#SBATCH --mem=10G                       # Memory allocation
#SBATCH --time=48:00:00                     # Time limit hrs:min:sec
#SBATCH --partition=gpu                     # Partition (compute (default) / gpu)
#SBATCH --output=slurm_output/slurm-%j.out           # Standard output log
#SBATCH --error=slurm_output/slurm-%j.err            # Error log file

# Load necessary modules (if needed)
module load all/CUDA   # Adjust based on your environment
# module load python/3.11  # Adjust based on your environment

# >>>> CONDA >>>>
source ~/miniconda3/etc/profile.d/conda.sh # Init conda necessary as slurm starts with non-interactive shell
conda activate nanotron

# Run the command
python3 config_tiny_llama.py && (cd ..; CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 run_train.py --config-file examples/config_tiny_llama.yaml)