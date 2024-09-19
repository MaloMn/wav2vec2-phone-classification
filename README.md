# Phone Classification using Wav2Vec2

This repository contains [Speechbrain](https://github.com/speechbrain/speechbrain) recipes to fine-tune Wav2Vec2 models on a phone classification task.
Following factors were analysed:
  1. Fine-tuning Wav2Vec2,
  2. Pre-training datasets,
  3. Model size,
  4. fine-tuning datasets.

Results of this work have been published at the Interspeech 2024 conference.

## Code
- The `recipes` folder contains all Speechbrain recipes.
- Results obtained are available in the `confusion-matrix/` folder.

## Data
For confidentiality reasons, datasets are not included.
This work relies on the [C2SI](https://hal.science/hal-02921918), [CommonPhone](https://arxiv.org/abs/2201.05912) and [BREF](https://www.isca-archive.org/eurospeech_1991/larnel91_eurospeech.html) corpora. 

## Recipes
*Details of some of the Speechbrain recipes set up in this repository.* 
- ``unfrozen-cp-3k-large-accents`` is the best recipe published in the Interspeech paper listed below.
- ``unfrozen-cp-3k-large-accents-argmax`` takes the maximum of all 6 segments (1024-dim). LeakyReLu.
- ``unfrozen-cp-3k-large-concatenate`` take both central segments (2048-dim) as input to the classifier.

## How to cite
If you use this work, please cite as:

```bib
@inproceedings{maisonneuve24,
  title     = {Towards objective and interpretable speech disorder assessment: a comparative analysis of CNN and transformer-based models},
  author    = {Malo Maisonneuve and Corinne Fredouille and Muriel Lalain and Alain Ghio and Virginie Woisard},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {1970--1974},
  doi       = {10.21437/Interspeech.2024-267},
  issn      = {2958-1796},
}
```
