#!/bin/bash --login 

#SBATCH --job-name=wav2vec2phone
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --constraint='GPUArch_Pascal&GPURAM_Min_16GB&GPURAM_Max_32GB'

echo "Activating environment wav2vec" > output.txt
conda activate wav2vec

echo "Launching Wav2Vec2 fine-tuning" >> output.txt
python recipe.py wav2vec2_phoneme.yml >> output.txt
