'''
This script handles the tracking.
'''
import glob
import argparse
import os
import pandas as pd
import time
import subprocess
from engine.trainval import trainval
from engine.inference import tracking
from utils import resultcsv_2xml
import shutil

__author__ = "Yudong Zhang"


def save_args_to_file(args, path):
    with open(path, 'a+') as file:
        for arg, value in vars(args).items():
            if isinstance(value, list):
                value = ', '.join(map(str, value))
            file.write(f"{arg}: {value}\n")
        file.write('--------------------------')


def parse_args_():
    parser = argparse.ArgumentParser()
    
    # data params
    parser.add_argument('--len_established',type=int,default=7) 
    parser.add_argument('--len_future',type=int,default=1) 
    parser.add_argument('--near',type=int,default=5)

    parser.add_argument('--traindatamean', nargs='+', type=int, default=[-3.8447242, -7.2373133, -2.3897965, -5.299531, -3.6751747, -10.802332, -2.909537, -7.12589, 803.7549, 403.8249, 774.43823, 836.82025, 314.45084, 494.85004,  62.86108, 179.01665])
    parser.add_argument('--traindatastd',nargs='+', type=int, default=[110.6746 ,67.4468, 112.93308, 112.18339, 72.53153, 95.47296, 41.04576, 102.74431, 455.76172, 218.71327, 454.32388, 459.0247, 217.7225, 238.48856, 49.815372, 130.16364])

    # device
    parser.add_argument('--no_cuda', default=False, action='store_true')
    
    # data path
    parser.add_argument('--test_path', type=str, default='dataset/yolox_det_all')
    # model path
    parser.add_argument('--model_ckpt_path', type=str, default='./pretrained_model/MOT17_trainval/20221127_13_59_43.chkpt')
    # save path
    parser.add_argument('--eval_save_path', type=str, default='./prediction/')

    parser.add_argument('--det_keep_rate', type=float, default=1.0)

    # vis process
    parser.add_argument('--vis',default=False, action='store_true')

    # track threshold
    parser.add_argument('--track_high_thresh',type=float,default=0.6)
    parser.add_argument('--track_low_thresh',type=float,default=0.1)
    parser.add_argument('--new_track_thresh',type=float,default=0.6)

    parser.add_argument('--track_buffer',type=float,default=30)

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':

    opt = parse_args_()

    # data param
    past = opt.len_established
    cand = opt.len_future
    near = opt.near
    
    now = int(round(time.time()*1000))
    nowname = time.strftime('%Y%m%d_%H_%M_%S',time.localtime(now/1000))

    # test_det_pa = opt.test_path
    model_p = opt.model_ckpt_path
    # output_csv_pa = os.path.join(opt.eval_save_path, nowname, 'track_result.csv')
    os.makedirs(os.path.join(opt.eval_save_path, nowname))
    save_args_to_file(opt, os.path.join(opt.eval_save_path, nowname, 'param.txt'))
    pyfilepath = os.path.abspath(__file__)
    shutil.copy(pyfilepath,os.path.join(opt.eval_save_path,nowname,os.path.split(pyfilepath)[-1]))
    shutil.copy(os.path.join(os.path.split(pyfilepath)[0],'engine/inference.py'), os.path.join(opt.eval_save_path,nowname, 'inference.py'))


    # threshold param
    track_buffer = opt.track_buffer
    track_high_thresh = opt.track_high_thresh
    new_track_thresh = opt.new_track_thresh

    for test_det_pa in glob.glob(os.path.join(opt.test_path,'**.csv')):
        seq = os.path.split(test_det_pa)[1].split('.')[0]
        output_csv_pa = os.path.join(opt.eval_save_path, nowname, seq+'_link.csv')

        if '05' in seq or '06' in seq:
            track_buffer = 14
        elif '13' in seq or '14' in seq:
            track_buffer = 25
        else:
            track_buffer = 30

        if '01' in seq:
            track_high_thresh = 0.65
        elif '06' in seq:
            track_high_thresh = 0.65
        elif '12' in seq:
            track_high_thresh = 0.7
        elif '14' in seq:
            track_high_thresh = 0.67

        if opt.vis:
            if not os.path.exists(os.path.join(opt.eval_save_path, nowname, seq)):
                os.makedirs(os.path.join(opt.eval_save_path, nowname, seq))

        keep_track = tracking(
            input_detfile=test_det_pa,
            output_trackcsv=output_csv_pa,
            model_path=model_p,
            fract=opt.det_keep_rate,
            Past=past,
            Cand=cand,
            Near=near,
            track_buffer=opt.track_buffer,
            new_track_thresh=opt.new_track_thresh,
            track_high_thresh=opt.track_high_thresh,
            track_low_thresh=opt.track_low_thresh,
            mean_=opt.traindatamean,
            std_=opt.traindatastd,
            vis=opt.vis,
            no_cuda=opt.no_cuda
            )
        
        # xmlfilepath = output_csv_pa.replace('.csv','.xml')
        # resultcsv_2xml(xmlfilepath, output_csv_pa)

