#!/bin/bash --login 

#SBATCH --job-name=wav2vec2phone
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=15G
#SBATCH --time=24:00:00
#SBATCH --constraint='GPURAM_Min_24GB'
#SBATCH --nodelist=eris

echo "Activating environment wav2vec" > $1.txt
conda activate wav2vec

echo "Launching wav2vec2 $1" >> $1.txt

if [ ! -f "recipes/$1/wav2vec2_phoneme.yml" ]; then
  python configuration/override.py recipes/base_configuration.yml recipes/$1/override.yml recipes/$1/wav2vec2_phoneme.yml
fi

python recipes/$1/recipe.py recipes/$1/wav2vec2_phoneme.yml >> $1.txt
