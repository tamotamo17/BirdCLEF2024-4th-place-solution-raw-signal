# BirdCLEF2024-4th-place-solution TmT's part
This repo contains training code of raw signal model from the [4th place solution of BirdCLEF2024](https://www.kaggle.com/competitions/birdclef-2024/discussion/511845)
## Environment
Google Colaboratory T4
## Data
- [Competition data](https://www.kaggle.com/competitions/birdclef-2024/data)
- [Background noise from unlabeled data](https://www.kaggle.com/datasets/tamotamo/bc24-unlabeled-background-crop)
- [Additional data](https://www.kaggle.com/datasets/yokuyama/birdclef2024-additional-cleaned)

Prepare 5 seconds clip using `prepare_dataset.ipynb` to speed up the training.

## Train
Run `train.ipynb`

## Conversion to openvino format
Upload your model to kaggle dataset, then convert pytroch models to openvino format using [this notebook](https://www.kaggle.com/code/tamotamo/convert-pytorch-model-to-openvino)

## Inference
Upload your openvino model to kaggle dataset, run [this notebook](https://www.kaggle.com/code/ajobseeker/b24-final?scriptVersionId=182393504) (this notebook uses other models, so please make them following this repo)
