#!/bin/bash --login 

#SBATCH --job-name=wav2vec2phone
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --constraint='GPURAM_Min_16GB&GPURAM_Max_32GB'

echo "Activating environment wav2vec" > $1.txt
conda activate wav2vec

echo "Launching wav2vec2 $1" >> $1.txt
python recipes/$1/recipe.py recipes/$1/wav2vec2_phoneme.yml >> $1.txt
