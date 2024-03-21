# Use MoTT to track pedestrains in the pedestrain_tracking branch
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
2. Install pytorch==1.8.0+cu111 torchvision==0.9.0+cu111
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
3. Clone this repo
```
git clone https://github.com/imzhangyd/MoTT.git
cd MoTT
```
4. Install dependencies:
 ```
 pip install -r requirements.txt
 ```

### Dataset preparation
Download the MOT17 dataset from https://motchallenge.net/data/MOT17 or [Get_all_data_link](https://motchallenge.net/data/MOT17.zip) and [Get_files_(no_image)_only](https://motchallenge.net/data/MOT17Labels.zip). Then unzip them and put them under ./dataset.

1. Generate training and validation dataset file from original .txt label file.

```
# 1 Convert gt.txt to .csv file
python ./data_preparation/txt2csv.py \
--txtpath_format='./dataset/MOT17Labels/train/**FRCNN/gt/gt.txt' \
--outputfolder='./dataset/MOT17_trainval_test/gt_pedescsv'

# 2 Generate train and val files for each sequence
python ./data_preparation/generate_trainvaldata_box.py \
--csvpath='./dataset/MOT17_trainval_test/gt_pedescsv/02gt_pedes.csv' \
--savefolder='./dataset/MOT17_trainval_test/trainval_box_onefuture'

python ./data_preparation/generate_trainvaldata_box.py \
--csvpath='./dataset/MOT17_trainval_test/gt_pedescsv/04gt_pedes.csv' \
--savefolder='./dataset/MOT17_trainval_test/trainval_box_onefuture'

python ./data_preparation/generate_trainvaldata_box.py \
--csvpath='./dataset/MOT17_trainval_test/gt_pedescsv/05gt_pedes.csv' \
--savefolder='./dataset/MOT17_trainval_test/trainval_box_onefuture'

python ./data_preparation/generate_trainvaldata_box.py \
--csvpath='./dataset/MOT17_trainval_test/gt_pedescsv/09gt_pedes.csv' \
--savefolder='./dataset/MOT17_trainval_test/trainval_box_onefuture'

python ./data_preparation/generate_trainvaldata_box.py \
--csvpath='./dataset/MOT17_trainval_test/gt_pedescsv/10gt_pedes.csv' \
--savefolder='./dataset/MOT17_trainval_test/trainval_box_onefuture'

python ./data_preparation/generate_trainvaldata_box.py \
--csvpath='./dataset/MOT17_trainval_test/gt_pedescsv/11gt_pedes.csv' \
--savefolder='./dataset/MOT17_trainval_test/trainval_box_onefuture'

python ./data_preparation/generate_trainvaldata_box.py \
--csvpath='./dataset/MOT17_trainval_test/gt_pedescsv/13gt_pedes.csv' \
--savefolder='./dataset/MOT17_trainval_test/trainval_box_onefuture'

# 3 Merge train/val files from all sequences
cat ./dataset/MOT17_trainval_test/trainval_box_onefuture/past7_depth1_near5/*_train.txt | sed '/^$/d' > ././dataset/MOT17_trainval_test/trainval_box_onefuture/past7_depth1_near5/merge_train.txt

cat ./dataset/MOT17_trainval_test/trainval_box_onefuture/past7_depth1_near5/*_val.txt | sed '/^$/d' > ./dataset/MOT17_trainval_test/trainval_box_onefuture/past7_depth1_near5/merge_val.txt
```

And make the code structure like this:

```
|-- MoTT  
`-- |-- dataset
    `-- |-- MOT17_trainval_test
        `-- |-- gt_pedescsv
            |    |-- 02gt_pedes.csv
            |    |-- 04gt_pedes.csv
            |    .......
            `-- trainval_box_onefuture
                `-- past7_depth1_near5
                |    |-- 02gt_pedes_train.txt
                |    |-- 02gt_pedes_val.txt
                |    ... ...
                |    |-- merge_train.txt
                |    |-- merge_val.txt
        `-- MOT17Labels
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
We provide a MOT17 pretrained model at this [link](https://drive.google.com/drive/folders/1L65UeBmr0UHyOavk-Geyfsx24OEyIBLx?usp=sharing), and the detection results of MOT17 test movies by YOLOX detector at this [link](https://drive.google.com/drive/folders/1QlnTsRG7V2LHfrm43RCrychfIYlGkx2W?usp=sharing), and we followed the YOLOX training instruction of ByteTrack.

1. Train.
```
python train.py \
--trainfilename='MOT17_trainvalbox' \
--train_path='dataset/MOT17_trainval_test/trainval_box_onefuture/past7_depth1_near5/merge_train.txt' \
--val_path='dataset/MOT17_trainval_test/trainval_box_onefuture/past7_depth1_near5/merge_val.txt' \
--use_tb \
--ckpt_save_root='./checkpoint'
```

2. Track single sequence or multiple sequences.
```
python tracking.py \
--test_path='dataset/yolox_det_all/MOT17-01-FRCNN.csv' \
--model_ckpt_path='./pretrained_model/MOT17_trainval/20221127_13_59_43.chkpt' \
--eval_save_path='./prediction/'
```

```
python tracking_mulseq.py \
--test_path='dataset/yolox_det_all' \
--model_ckpt_path='./pretrained_model/MOT17_trainval/20221127_13_59_43.chkpt' \
--eval_save_path='./prediction/'
```

3. Evaluate tracking results of MOT17 training sequences.
We use the official [evaluation tools](https://github.com/JonathonLuiten/TrackEval.git) to calculate metrics on tracking results of MOT17 training data. We upload [a version](https://github.com/imzhangyd/TrackEval-pedes.git) for quick test pedestrain tracking. Refer its 'Readme.md' for details.

```
#1 Postprocessing and preparing for evaluation.
python result2bbox_auginput17 \
--resultfolder='./prediction/20240321_17_12_34' \
--interpolation \
--detcsvfolder='/ldap_shared/home/s_zyd/MoTT/dataset/yolox_det_all'

#2 Link the results to evaluation path.
name='mott-test'
mkdir "/data/ldap_shared/home/s_zyd/TrackEval-pedes/data/trackers/mot_challenge/MOT17-train/${name}"
ln -s "/ldap_shared/home/s_zyd/MoTT/prediction/20240321_17_12_34/data" "/ldap_shared/home/s_zyd/TrackEval-pedes/data/trackers/mot_challenge/MOT17-train/${name}/data"

#3 Eval
python scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL $name --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --NUM_PARALLEL_CORES 1
```
If you want to submit results of test sequences for online evaluation, you can zip the result 'data' folder and upload it to the [website](https://motchallenge.net/).

4. Tracking with visulizing process.
If you want to see what happened during the matching process,
```
python tracking.py \
--test_path='dataset/yolox_det_all/MOT17-02-FRCNN.csv' \
--model_ckpt_path='./pretrained_model/MOT17_trainval/20221127_13_59_43.chkpt' \
--eval_save_path='./prediction/' \
--vis \
--imageroot='./dataset/MOT17/train
```


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


