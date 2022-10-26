# PM2F2N

[Title] PM2F2N: Patient Multi-view Multi-modal Feature Fusion Networks for
Clinical Outcome Prediction

[Authors] Ying Zhang, Baohang Zhou, Kehui Song, Xuhui Sui, Guoqing Zhao, Ning Jiang and Xiaojie Yuan

[EMNLP 2022 Findings]

## Preparation

1. Clone the repo to your local.
2. Download Python version: 3.6.13
3. Download the preprocessed data from this [link](https://pan.baidu.com/s/1nATps0aCua5QfLXGzw7feQ) and the extraction code is **1234**. Put the downloaded files into the **data** folder.
4. Open the shell or cmd in this repo folder. Run this command to install necessary packages.

```cmd
pip install -r requirements.txt
```

## Experiments

1. For Linux systems, we have shell scripts to run the training procedures. You can run the following command:

```cmd
./train.model.sh
```

2. You can also input the following command to train the model. There are different choices for some hyper-parameters shown in square barckets. The meaning of these parameters are shown in the following tables.

|  Parameters | Value | Description|
|  ----  | ----  | ---- |
| epoch | int | Training times |
| patience | int | Early stopping |
| weights | string | Saved model path |
| save_features | int | Whether to save multimodal features |
| task | string | Choose the clinical outcome task |

```cmd
CUDA_VISIBLE_DEVICES=0 python mmg_main.py \
    --seed 0 \
    --epoch 300 \
    --patience 20 \
    --weights ./weights/mort_icu.h5 \
    --task mort_icu \
    --mode train \
    --save_features 0
```

3. After training the model, you can run the test script to evaluate the model on the test set.

```cmd
./test.model.sh

or

CUDA_VISIBLE_DEVICES=0 python mmg_main.py \
    --seed 0 \
    --weights ./weights/mort_icu.h5 \
    --task mort_icu \
    --mode test \
    --save_features 0
```

4. We also provide the weights of the model to reimplement the results in our paper. The saved weights have been in **weights** folder. You can run the test script directly after downloading the processed data.

## References

1. We utilize the [MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract) pipline tools to acquire the raw data. And you can also follow the guide lines to process the raw data for your research.
