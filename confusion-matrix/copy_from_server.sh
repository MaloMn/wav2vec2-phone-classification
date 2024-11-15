#!/bin/bash

dataset=$1
seed=$2

echo "$dataset"/"$seed"

# $var can be "best-relu-long-context/2001"
# Copy all final files to correct locations locally
mkdir -p bref/"$dataset"/
scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/results/"$dataset"/"$seed"/output_test.json bref/"$dataset"/
mkdir -p c2si/"$dataset"/
scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/results/"$dataset"/"$seed"/output_hc_dap.json c2si/"$dataset"/
scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/results/"$dataset"/"$seed"/output_hc_lec.json c2si/"$dataset"/
scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/results/"$dataset"/"$seed"/output_hc_dap.json c2si/"$dataset"/
scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/results/"$dataset"/"$seed"/output_hc_lec.json c2si/"$dataset"/
scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/results/"$dataset"/"$seed"/output_patients_lec.json c2si/"$dataset"/
scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/results/"$dataset"/"$seed"/output_patients_dap.json c2si/"$dataset"/
