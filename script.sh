#!/bin/bash

#SBATCH --job-name=wav2vec2phone
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --constraint='GPURAM_16GB'

echo "Launching Wav2Vec2 fine-tuning"
python recipe.py hparams/wav2Vec