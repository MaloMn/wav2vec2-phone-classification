#!/bin/bash --login 

#SBATCH --job-name=wav2vec2phone
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --constraint='GPURAM_Min_12GB'
#SBATCH --array=0-7%4

echo "Activating environment wav2vec" > array_$SLURM_ARRAY_TASK_ID.txt
conda activate wav2vec

echo "Launching wav2vec2 array" >> array_$SLURM_ARRAY_TASK_ID.txt
python recipes/array-frozen/prepare_jobs.py $SLURM_ARRAY_TASK_ID
python recipes/array-frozen/recipe.py recipes/array-frozen/temp/$SLURM_ARRAY_TASK_ID/temp.yml >> array_$SLURM_ARRAY_TASK_ID.txt
