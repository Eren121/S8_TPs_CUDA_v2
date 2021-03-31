#!/bin/bash
# Job pour CUDA

# Réservation d'un noeud avec ses 28 coeurs et ses 4 GPU

#SBATCH --error=job.err.txt
#SBATCH --output=job.out.txt

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G

module load cuda/11.0
module load python

srun ../matmul.out 100
