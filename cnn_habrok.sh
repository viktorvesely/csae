#!/bin/bash
#SBATCH --mem=16GB
#SBATCH --time=8:00:00
#SBATCH --job-name=cnnsae
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --nodes=1

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

cd sae
srun python csae_train.py --name oursactivation10 --lsfactor 10