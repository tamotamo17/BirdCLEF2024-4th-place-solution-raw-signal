# BirdCLEF2024-4th-place-solution TmT's part
This repository contains training code of raw signal models from the [4th place solution of BirdCLEF2024](https://www.kaggle.com/competitions/birdclef-2024/discussion/511845).  
You can find the code for our other models, the Mel-spectrogram model, in [this repository](https://github.com/yoku001/BirdCLEF2024-4th-place-solution-melspec).
## Environment
Google Colaboratory with T4 GPU
## Data
- [Competition data](https://www.kaggle.com/competitions/birdclef-2024/data)
- [Background noise from unlabeled data](https://www.kaggle.com/datasets/tamotamo/bc24-unlabeled-background-crop)
- [Additional data](https://www.kaggle.com/datasets/yokuyama/birdclef2024-additional-cleaned)

Prepare 5 seconds clip using `prepare_dataset.ipynb` or use [this cropped dataset](https://www.kaggle.com/datasets/tamotamo/birdclef2024-crop/data) to speed up the training.

## Train
Run `train.ipynb`

## Conversion to openvino format
Upload your model to kaggle dataset, then convert pytroch models to openvino format using [this notebook](https://www.kaggle.com/code/tamotamo/convert-pytorch-model-to-openvino).

## Inference
Upload your openvino model to kaggle dataset, then run [this notebook](https://www.kaggle.com/code/ajobseeker/b24-final?scriptVersionId=182393504).
