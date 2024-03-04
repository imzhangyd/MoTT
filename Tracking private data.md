# Detection and tracking pipeline

Assume you have a image sequence to track particles in it. First detect particles on all frames and then track using their positions. 

We use [DL_Particle_Detection](https://github.com/imzhangyd/DL_Particle_Detection.git) to detect, and [MoTT](https://github.com/imzhangyd/MoTT.git) to track.

## Example 1: Track an image sequence without detection labels and track labels.
### 1. Image data preparation
As introduced in [MoTT
/DATASET.md](https://github.com/imzhangyd/MoTT/blob/main/DATASET.md), we download the challenge image. 

### 2. Detection code preparation
Install DL_Particle_Detection according to the [Installation](https://github.com/imzhangyd/DL_Particle_Detection/blob/main/README.md). Download the pretrained deepBlink model checkpoints at [link](https://drive.google.com/drive/folders/1W93aOc_rCUnCS4D1ZFBSN4YJa-va-8hN).

### 3. Detect particles
```
python infer_one_thre_onlypred.py \
--test_datapath='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/challenge/MICROTUBULE snr 7 density low/' \
--ckpt_path='./pretrained_model/MICROTUBULE_SNR7/checkpoints_113.pth' \
--exp_name='MICROTUBULE_SNR7_densitylow_deepBlink'
```

### 4. Detection result preparation for track
```
python utils/detcsv2xml.py \
--det_folder='./Log/20240301_11_12_50_MICROTUBULE_SNR7_deepBlink_eval/prediction_0.5' \
--detfor_track='./detfor_track'
```

### 5. Tracking code preparation
Install MoTT according to the [Installation](https://github.com/imzhangyd/MoTT/blob/main/README.md). Download the pretrained MoTT model checkpoint at [link](https://drive.google.com/drive/folders/1-0mp2tQ3YXu4wHK3GbboI-ombeWpYdzC?usp=sharing).

Make the directory like this:
```
|-- ${rootdir}
`-- |-- DL_Particle_Detection
    `-- MoTT
```

### 6. Tracking
```
# cd MoTT
CUDA_VISIBLE_DEVICES=2 python tracking.py \
--test_path='../DL_Particle_Detection/detfor_track/MICROTUBULE snr 7 density low.xml' \
--model_ckpt_path='./pretrained_model/MICROTUBULE_snr_1247_density_low/20220406_11_18_51.chkpt' \
--eval_save_path='./prediction/'
```

### 7. Visualize tracks
```
python vis_tracks.py \
--imgfolder='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/challenge/MICROTUBULE snr 7 density low' \
--trackcsvpath='./prediction/20240301_15_25_56/track_result.csv' \
--vis_save='./prediction/20240301_15_25_56/track_vis'
```

## Example 2: Train detector and trackor and evaluate.
### 1. Image data preparation
As introduced in [MoTT
/DATASET.md](https://github.com/imzhangyd/MoTT/blob/main/DATASET.md), we download the training image with .xml lable file and challenge image with its .xml label file.

### 2. Detection code preparation
Install DL_Particle_Detection according to the [Installation](https://github.com/imzhangyd/DL_Particle_Detection/blob/main/README.md).

### 3. Detect particles
(1) Convert the track lable file (.xml) to detection label file for training and evaluation.

```
# training data
python utils/xml2csv.py \
--imgfolder='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/training/MICROTUBULE snr 7 density low' \
--track_xmlpath='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/training/MICROTUBULE snr 7 density low/MICROTUBULE snr 7 density low.xml'
# challenge data
python utils/xml2csv.py \
--imgfolder='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/challenge/MICROTUBULE snr 7 density low' \
--track_xmlpath='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/challenge/MICROTUBULE snr 7 density low.xml'
```
(2) Split train val data
```
python utils/split_trainval.py \
--img_folder='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/training/MICROTUBULE snr 7 density low' \
--save_path='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/training/MICROTUBULE snr 7 density low' \
--split_p=0.8
```

(3) Train detection model
```
python trainval.py \
--train_datapath='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/training/MICROTUBULE snr 7 density low_train/' \
--val_datapath='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/training/MICROTUBULE snr 7 density low_val/' \
--exp_name='MICROTUBULE_SNR7_low_density_deepBlink'
```

(4) Evaluation on challenge data
```
python infer_determine_thre.py \
--val_datapath='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/training/MICROTUBULE snr 7 density low_val/' \
--test_datapath='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/challenge/MICROTUBULE snr 7 density low/' \
--ckpt_path='Log/20240301_17_18_13_MICROTUBULE_SNR7_low_density_deepBlink_trainval/checkpoints/checkpoints_138.pth' \
--exp_name='MICROTUBULE_SNR7_densitylow_deepBlink'
```

### 4. Detection result preparation for track
```
python utils/detcsv2xml.py \
--det_folder='./Log/20240304_09_08_01_MICROTUBULE_SNR7_densitylow_deepBlink_eval/prediction_0.4' \
--detfor_track='./detfor_track'
```

### 5. Tracking code preparation
Install MoTT according to the [Installation](https://github.com/imzhangyd/MoTT/blob/main/README.md).

Make the directory like this:
```
|-- ${rootdir}
`-- |-- DL_Particle_Detection
    `-- MoTT
```

### 6. Tracking and evaluation
(1) Make the train val file for training from original .xml label file of training data.
```
# cd MoTT
python generate_trainvaldata.py \
--trackxmlpath='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/training/MICROTUBULE snr 7 density low/MICROTUBULE snr 7 density low.xml' \
--savefolder='./dataset/ISBI_trainval'
```

(2) Train model
```
CUDA_VISIBLE_DEVICES=1 python train.py \
--trainfilename='MICROTUBULE snr 7 density low' \
--train_path='./dataset/ISBI_trainval/past7_depth2_near5/MICROTUBULE snr 7 density low_train.txt' \
--val_path='./dataset/ISBI_trainval/past7_depth2_near5/MICROTUBULE snr 7 density low_val.txt'
```

(3) Tracking
```
CUDA_VISIBLE_DEVICES=1 python tracking.py \
--test_path='../DL_Particle_Detection/detfor_track/MICROTUBULE snr 7 density low.xml' \
--model_ckpt_path='./checkpoint/20240304_09_38_32_MICROTUBULE_snr_7_density_low_ckpt/20240304_09_38_53.chkpt' \
--eval_save_path='./prediction/'
```

(4) Evaluation
```
python eval.py \
--GTxmlpath='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/challenge/MICROTUBULE snr 7 density low.xml' \
--pred_xmlpath='./prediction/20240304_10_08_22/track_result.xml'
```
### 7. Visualize tracks
```
python vis_tracks.py \
--imgfolder='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/challenge/MICROTUBULE snr 7 density low' \
--trackcsvpath='./prediction/20240304_10_08_22/track_result.csv' \
--vis_save='./prediction/20240304_10_08_22/track_vis'
```


