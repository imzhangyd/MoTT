# Dataset Preparation

## ISBI Particle Tracking Dataset
You can download ISBI particle tracking dataset at http://bioimageanalysis.org/track/, or you can download below.

- Training images and ground truth tracks: [Microtubules](http://bioimageanalysis.org/track/bench/microtubule.zip), [Receptors](http://bioimageanalysis.org/track/bench/receptor.zip), [Vesicles](http://bioimageanalysis.org/track/bench/vesicle.zip), [Viruses](http://bioimageanalysis.org/track/bench/virus.zip).

- Challenge image data: [Microtubules](http://bioimageanalysis.org/track/challenge/microtubule.zip), [Receptors](http://bioimageanalysis.org/track/challenge/receptor.zip), [Vesicles](http://bioimageanalysis.org/track/challenge/vesicle.zip), [Viruses](http://bioimageanalysis.org/track/challenge/virus.zip). [Challenge ground truth for all scenarios](http://bioimageanalysis.org/track/challenge/ground_truth.zip).

## Trainval file generation
Each training ground truth tracks .xml file can be converted to training and validation .txt file for training models. For example,
```
python generate_trainvaldata.py \
--trackxmlpath='./dataset/ground_truth/MICROTUBULE snr 7 density low.xml' \
--savefolder='./dataset/ISBI_trainval'
```
