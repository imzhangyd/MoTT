# Detection and tracking pipeline

Assume you have a image sequence to track particles in it. First detect particles on all frames and then track using their positions. 

We use [DL_Particle_Detection](https://github.com/imzhangyd/DL_Particle_Detection.git) to detect, and [MoTT](https://github.com/imzhangyd/MoTT.git) to track.

## Example 1: An image sequence without detection labels and track labels.
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

## Example 2: An image sequence with evaluation.



### Data preparation
You can use https://github.com/imzhangyd/DL_Particle_Detection/blob/main/utils/xml2csv.py to convert the xml track labels to .csv detection labels.







## Tracking

### 
