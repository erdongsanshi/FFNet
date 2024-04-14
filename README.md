# Fuss-Free Network: A Simplified and Efficient Neural Network for Crowd Counting

This repository includes the official implementation of the paper:

Original paper [Link]（http://arxiv.org/abs/2404.07847）

## Installation

```
pip install -r requirements.txt
```
## Data Preparation

- Download crowd-counting datasets, e.g., [ShanghaiTech](https://drive.google.com/file/d/1pSXVqS9NxIKs8W4-DAH38StWiWvBv1Zh/view?usp=drive_link)
                                          [NWPU](https://drive.google.com/file/d/1Mt9aEyejhsx3rCIaW2jepFQGacv5qFzw/view?usp=drive_link)

- You can also implement your own datasets processing according to the processing file

## Training
Adjust the parameters of ```train.py``` as needed.
update root "data-dir" and "dataset" in ./train.py.
```bash
python train.py
```
You can set ```--wandb 1``` in ```train.py``` after having logged in to wandb (```wandb login``` in console) to visualize the training process

## Testing
* python test_image_patch.py
* Due to crop training with size of 256x256, the validation image is divided into several patches with size of 256x256, and the overlapping area is averaged.
* Download the pretrained model from Baidu-Disk(提取码: se59) [link](https://pan.baidu.com/s/16qY_cFIUAUaDRsdr5vNsWQ)


## Pretrained Models

| Dataset                  | Model Link  | MAE |
| ------------------------ | ----------- | --- |
| ShanghaiTech PartA       |  [SHA_model.pth](https://drive.google.com/file/d/1vQLWSIYTUXMJKMnVlKgiuYGbsVqPyN9W/view?usp=drive_link)   | 48.30 |
| ShanghaiTech PartB       |  [SHB_model.pth](https://drive.google.com/file/d/1LhRde7Ztpg1pn3C7DIfZj9tkP5FrHh1P/view?usp=drive_link)   |  6.1  |
| NWPU(Val)                |  [NWPU_model.pth](https://drive.google.com/file/d/1zCkfSvTV2Cx5boEZ3a6lgyQMCPeeNGmi/view?usp=drive_link)  | 41.26 |
* Due to the image clipping size is different during training, the clipping size should be adjusted accordingly during validation. The clipping size is 256×256 for SHA, 512×512 for SHB, and 384×384 for NWPU.

## Acknowledgement

We thank the authors of [DM-Count](https://github.com/cvlab-stonybrook/DM-Count) and [CCtrans](https://github.com/wfs123456/CCTrans) for open-sourcing their work.
