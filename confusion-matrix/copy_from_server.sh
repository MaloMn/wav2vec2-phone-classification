#!/bin/bash

#dataset=$1
#seed=$2
#
#echo "$dataset"/"$seed"
#
## $var can be "best-relu-long-context/2001"
## Copy all final files to correct locations locally
#mkdir -p bref/"$dataset"/
#scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/results/"$dataset"/"$seed"/output_test.json bref/"$dataset"/
#mkdir -p c2si/"$dataset"/
#scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/results/"$dataset"/"$seed"/output_hc_dap.json c2si/"$dataset"/
#scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/results/"$dataset"/"$seed"/output_hc_lec.json c2si/"$dataset"/
#scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/results/"$dataset"/"$seed"/output_hc_dap.json c2si/"$dataset"/
#scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/results/"$dataset"/"$seed"/output_hc_lec.json c2si/"$dataset"/
#scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/results/"$dataset"/"$seed"/output_patients_lec.json c2si/"$dataset"/
#scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/results/"$dataset"/"$seed"/output_patients_dap.json c2si/"$dataset"/


#for i in {0..24}; do
#  echo $i
#  mkdir -p bref/array-"$i"/
#  scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/confusion-matrix/bref/array-"$i"/output_test_accuracies.json bref/array-"$i"/
#  mkdir -p c2si/array-"$i"/
#  scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/confusion-matrix/c2si/array-"$i"/output_hc_dap_accuracies.json c2si/array-"$i"/
#  scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/confusion-matrix/c2si/array-"$i"/output_hc_lec_accuracies.json c2si/array-"$i"/
#done;



for i in {0..24}; do
  echo $i
  mkdir -p bref/ft-array-"$i"/
  scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/confusion-matrix/bref/ft-array-"$i"/output_test_accuracies.json bref/ft-array-"$i"/
  mkdir -p c2si/ft-array-"$i"/
  scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/confusion-matrix/c2si/ft-array-"$i"/output_hc_dap_accuracies.json c2si/ft-array-"$i"/
  scp -r mmaisonneuve@hades:/data/coros1/mmaisonneuve/wav2vec2-phoneme-classification/confusion-matrix/c2si/ft-array-"$i"/output_hc_lec_accuracies.json c2si/ft-array-"$i"/
done;