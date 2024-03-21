'''
unfinished
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
from utils import load_model
from engine.trainval import eval
from Dataset import func_getdataloader

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
    
    # train params
    # data params
    parser.add_argument('--val_path', default='dataset/MOT17_trainval_test/trainval_box_onefuture/past7_depth1_near5/merge_val.txt')    

    parser.add_argument('--traindatamean', nargs='+', type=int, default=[-3.8447242, -7.2373133, -2.3897965, -5.299531, -3.6751747, -10.802332, -2.909537, -7.12589, 803.7549, 403.8249, 774.43823, 836.82025, 314.45084, 494.85004,  62.86108, 179.01665])
    parser.add_argument('--traindatastd',nargs='+', type=int, default=[110.6746 ,67.4468, 112.93308, 112.18339, 72.53153, 95.47296, 41.04576, 102.74431, 455.76172, 218.71327, 454.32388, 459.0247, 217.7225, 238.48856, 49.815372, 130.16364])


    # model path
    parser.add_argument('--model_ckpt_path', type=str, default='./pretrained_model/MOT17_trainval/20221127_13_59_43.chkpt')
    parser.add_argument('--batch_size', type=int, default=64)

    # device
    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('--output_pa', type=str, default='./prediction')
    
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':

    opt = parse_args_()

    loadmodelpa = {}
    loadmodelpa['model'] = opt.model_ckpt_path

    if not opt.no_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    now = int(round(time.time()*1000))
    nowname = time.strftime('%Y%m%d_%H_%M_%S',time.localtime(now/1000))
    opt.output_dir = os.path.join(opt.output_pa, nowname)
    os.makedirs(opt.output_dir, exist_ok=True)

    transformer_ins = load_model(loadmodelpa,device)

    ins_loader_val,valdata = func_getdataloader(
        txtfile=opt.val_path, batch_size=opt.batch_size, 
        shuffle=True, num_workers=16,mean=opt.traindatamean,std=opt.traindatastd)
    
    eval(transformer_ins, ins_loader_val,len(valdata), device, opt,valdata.mean,valdata.std)

# unfinished