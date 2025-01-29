#!/bin/bash --login 

#SBATCH --job-name=wav2vec2phone
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --constraint='GPURAM_Min_12GB'
#SBATCH --array=0-113%4

conda activate wav2vec

PATIENTS=(40 24 25 26 27 20 21 22 23 28 29 344 347 340 341 343 "345-1" "345-2" 379 378 "16-1" 371 370 373 372 375 374 377 376 319 318 313 312 311 310 317 316 315 314 393 392 390 "16-2" 368 366 367 364 365 362 363 360 361 308 309 "348-1" "348-2" 300 303 304 305 306 307 381 382 383 384 385 386 387 388 39 38 "322-1" 32 31 "322-2" 37 36 35 34 339 338 335 334 337 336 331 330 333 332 "350-2" "350-1" 17 19 33 "301-1" "301-2" 1 320 321 326 329 "324-2" "324-1" "369-2" "369-1" 357 356 355 354 353 352 359 358)

sed "s/\$PATIENTID/${PATIENTS[$SLURM_ARRAY_TASK_ID]}/g" recipes/unfrozen-cp/wav2vec2_phoneme.yml > recipes/unfrozen-cp/temp/${PATIENTS[$SLURM_ARRAY_TASK_ID]}.yml

python recipes/unfrozen-cp/recipe.py recipes/unfrozen-cp/temp/${PATIENTS[$SLURM_ARRAY_TASK_ID]}.yml

rm recipes/unfrozen-cp/temp/${PATIENTS[$SLURM_ARRAY_TASK_ID]}.yml
