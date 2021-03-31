#!/bin/bash
#
# Usage: ./job.sh <nom>
# bw, add2vec, farhenheit...
#
#
# Génère une soumission Slurm

nom=$1
job_name=tp_cuda
script=job.slurm.sh

out=../data/${job_name}_out.txt
err=../data/${job_name}_err.txt

scancel -n "$job_name" -b

# Vider les fichiers de sortie (sans les supprimer)
: > $out
: > $err

sbatch -J "$job_name" --error "$err" --output "$out" "$script"


# Afficher la sortie du job (stdout + stderr) en temps réel
tail -f "$out" "$err"