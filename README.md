# Sign2GPT: Leveraging Large Language Models for Gloss-Free Sign Language Translation

This repo is the official implementation of "Sign2GPT: Leveraging Large Language Models for Gloss-Free Sign Language Translation".

## Environment Setup

Build the Docker image using the `Dockerfile` and `environment.yml` files provided.

## Dataset Preparation

Request access for CSL-Daily from the authors of "Improving Sign Language Translation with Monolingual Data by Sign Back-Translation" (Zhou et al.) and add it to folder named `csldaily`.

1. Create the tsv file for easier reading of the dataset using `scripts/csldaily/tsv_processing.py`

2. Since the dataset is divided into frames, using the script `scripts/csldaily/video_creator.py` to convert the individual frames to videos which are saved in `dataset_creation/csl-daily` folder. You will need to do this for every sequence in the dataset.

3. Convert the videos into lmdbs using `scripts/csldaily/image_lmdb_creator.py`. You will need to do this for every video in the dataset.

4. Create the pseudo-gloss dictionary pickle file with `scripts/csldaily/pseudo_gloss_zn.py`



# Citation
```
@inproceedings{
  wong2024signgpt,
  title={Sign2{GPT}: Leveraging Large Language Models for Gloss-Free Sign Language Translation},
  author={Ryan Wong and Necati Cihan Camgoz and Richard Bowden},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=LqaEEs3UxU}
}
```
