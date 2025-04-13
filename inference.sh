#!/bin/bash
#SBATCH --mem=80GB
#SBATCH --time=1:00:00
#SBATCH --job-name=chess_sae
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=log_sae/chess_sae_out.%j.log  # Output log
#SBATCH --error=log_sae/chess_sae_err.%j.log   # Error log
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 
source "/home3/s3799042/venv_sae/bin/activate"
srun python3 inference.py