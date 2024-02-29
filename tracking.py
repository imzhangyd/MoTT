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
    parser.add_argument('--len_future',type=int,default=2) 
    parser.add_argument('--near',type=int,default=5)

    # device
    parser.add_argument('--no_cuda', default = False)
    
    # data path
    parser.add_argument('--test_path', type=str, default='dataset/deepblink_det/MICROTUBULE snr 7 density low.xml')
    # model path
    parser.add_argument('--model_ckpt_path', type=str, default='./pretrained_model/MICROTUBULE_snr_1247_density_low/20220406_11_18_51.chkpt')
    # save path
    parser.add_argument('--eval_save_path', type=str, default='./prediction/')

    parser.add_argument('--det_keep_rate', type=float, default=1.0)

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

    test_det_pa = opt.test_path
    model_p = opt.model_ckpt_path
    output_csv_pa = os.path.join(opt.eval_save_path, nowname, 'track_result.csv')

    save_args_to_file(opt, os.path.join(opt.eval_save_path, nowname, 'param.txt'))

    keep_track = tracking(
        input_detxml=test_det_pa,
        output_trackcsv=output_csv_pa,
        model_path=model_p,
        fract=opt.det_keep_rate,
        Past=past,
        Cand=cand,
        Near=near,
        no_cuda=opt.no_cuda
        )
