# MoTT: Particle Tracking Method in Fluorescence Microscopy Images
MoTT is the official repository of the paper "[A Motion Transformer for Single Particle Tracking in Fluorescence Microscopy Images](https://link.springer.com/chapter/10.1007/978-3-031-43993-3_49)" , which has been accepted by MICCAI 2023. This work provides a powerful tool for studying the complex spatiotemporal behavior of subcellular structures. ([Springer Link](https://link.springer.com/chapter/10.1007/978-3-031-43993-3_49), [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.07.20.549804v1))


## Environment
The code is developed using python 3.7.3 on Ubuntu 18.04. NVIDIA GPUs are needed. The code is developed and tested using 1 NVIDIA GeForce RTX 2080 Ti card.

## Quick Start
### Installation
1. Create a virtual environment
```
conda create -n mott python==3.7.3 -y
conda activate mott
```
2. (Optional) Install Gurobi solver.
Refer to [How-do-I-install-Gurobi-for-Python](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python).  
3. Install pytorch==1.8.0+cu111 torchvision==0.9.0+cu111
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
4. Clone this repo
```
git clone https://github.com/imzhangyd/MoTT.git
cd MoTT
```
5. Install dependencies:
 ```
 pip install -r requirements.txt
 ```

### Dataset and pretrained model
Download the dataset and pretained model by https://drive.google.com/drive/folders/1-0mp2tQ3YXu4wHK3GbboI-ombeWpYdzC?usp=sharing. And make the code structure like this:

```
|-- MoTT  
`-- |-- dataset
    `-- |-- ISBI_mergesnr_trainval_data # for training, ISBI training data
        |    |-- MICROTUBULE snr 1247 density high_train.txt
        |    |-- MICROTUBULE snr 1247 density high_val.txt
        |    |-- MICROTUBULE snr 1247 density low_train.txt
        |    |-- MICROTUBULE snr 1247 density low_val.txt
        |    ... ...
        `-- deepblink_det # for testing, detection results by deepBlink detector, ISBI challenge data
        |    |-- MICROTUBULE snr 7 density low.xml
        |    |-- MICROTUBULE snr 4 density low.xml
        |    |-- MICROTUBULE snr 2 density low.xml
        |    ... ...
        `-- ground_truth # Ground truth used as GT detection for prediction or as label for evaluation, ISBI challenge data
        |    |-- MICROTUBULE snr 7 density low.xml
        |    |-- MICROTUBULE snr 4 density low.xml
        |    |-- MICROTUBULE snr 2 density low.xml
        |    ... ...
    `-- pretrained_model    # our pretrained model checkpoint
    `-- src    # tracking performance evaluation java code
    `-- transformer
    `-- engine
        |    |-- inference.py
        |    |-- trainval.py
    `-- Dataset.py    # dataset class when train
    `-- Dataset_match.py    # dataset class when prediction
    `-- generate_trainvaldata.py    # generate trainval data from original xml file
    `-- traickingPerformanceEvaluation.jar    # tracking performance evaluation tool
    `-- utils.py
    `-- train_tracking_eval.py
    `-- tracking.py
    `-- eval.py
    `-- vis_tracks.py    # vis tracking results
    ... ...


```

### Example
1. Train, tracking and evaluation.
```
python train_tracking_eval.py \
--trainfilename='MICROTUBULE snr 1247 density low' \
--train_path='dataset/ISBI_mergesnr_trainval_data/MICROTUBULE snr 1247 density low_train.txt' \
--val_path='dataset/ISBI_mergesnr_trainval_data/MICROTUBULE snr 1247 density low_val.txt' \
--use_tb \
--ckpt_save_root='./checkpoint' \
--test_path='dataset' \
--testsnr_list 4 7 \
--eval_save_path='./prediction/' \
--train \
--tracking \
--eval
```

2. Train only, no tracking and evaluation.
```
python train_tracking_eval.py \
--trainfilename='MICROTUBULE snr 1247 density low' \
--train_path='dataset/ISBI_mergesnr_trainval_data/MICROTUBULE snr 1247 density low_train.txt' \
--val_path='dataset/ISBI_mergesnr_trainval_data/MICROTUBULE snr 1247 density low_val.txt' \
--use_tb \
--ckpt_save_root='./checkpoint' \
--train
```

3. Train and tracking, no evaluation
```
python train_tracking_eval.py \
--trainfilename='MICROTUBULE snr 1247 density low' \
--train_path='dataset/ISBI_mergesnr_trainval_data/MICROTUBULE snr 1247 density low_train.txt' \
--val_path='dataset/ISBI_mergesnr_trainval_data/MICROTUBULE snr 1247 density low_val.txt' \
--use_tb \
--ckpt_save_root='./checkpoint' \
--test_path='dataset' \
--testsnr_list 4 7 \
--eval_save_path='./prediction/' \
--train \
--tracking
```

4. Tracking using pretrained checkpoint and evaluation
```
python train_tracking_eval.py \
--trainfilename='MICROTUBULE snr 1247 density low' \
--test_path='dataset' \
--testsnr_list 4 7 \
--eval_save_path='./prediction/' \
--model_ckpt_path='./pretrained_model/MICROTUBULE_snr_1247_density_low/20220406_11_18_51.chkpt' \
--tracking \
--eval
```

5. Tracking using the pretrained checkpoint, no evaluation
```
python train_tracking_eval.py \
--trainfilename='MICROTUBULE snr 1247 density low' \
--test_path='dataset' \
--testsnr_list 4 7 \
--eval_save_path='./prediction/' \
--model_ckpt_path='./pretrained_model/MICROTUBULE_snr_1247_density_low/20220406_11_18_51.chkpt' \
--tracking
```

6. Evaluation only
```
python eval.py \
--GTxmlpath='./dataset/ground_truth/MICROTUBULE snr 7 density low.xml' \
--pred_xmlpath='./prediction/......'
```

7. Tracking only
```
python tracking.py \
--test_path='dataset/deepblink_det/MICROTUBULE snr 7 density low.xml' \
--model_ckpt_path='./pretrained_model/MICROTUBULE_snr_1247_density_low/20220406_11_18_51.chkpt' \
--eval_save_path='./prediction/'
```

### Detect and track on your own data
Refer to [Tracking private data.md](https://github.com/imzhangyd/MoTT/blob/main/Tracking%20private%20data.md).

## Cite this paper
```
@InProceedings{10.1007/978-3-031-43993-3_49,
author="Zhang, Yudong
and Yang, Ge",
editor="Greenspan, Hayit
and Madabhushi, Anant
and Mousavi, Parvin
and Salcudean, Septimiu
and Duncan, James
and Syeda-Mahmood, Tanveer
and Taylor, Russell",
title="A Motion Transformer for Single Particle Tracking in Fluorescence Microscopy Images",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="503--513",
isbn="978-3-031-43993-3"
}
```
## Acknowledgement
The model of this repository is based on the repository [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch).


