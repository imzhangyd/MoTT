'''
This script handles the training, tracking and evaluation.
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
    
    # train params
    # data params
    parser.add_argument('--trainfilename', type=str, default='MICROTUBULE snr 1247 density low')
    parser.add_argument('--train_path', default='dataset/ISBI_mergesnr_trainval_data/MICROTUBULE snr 1247 density low_train.txt')   
    parser.add_argument('--val_path', default='dataset/ISBI_mergesnr_trainval_data/MICROTUBULE snr 1247 density low_val.txt')    

    parser.add_argument('--len_established',type=int,default=7) 
    parser.add_argument('--len_future',type=int,default=2) 
    parser.add_argument('--near',type=int,default=5)

    # network params
    parser.add_argument('--n_layers', type=int, default=1) 
    parser.add_argument('--d_k', type=int, default=96) 
    parser.add_argument('--d_v', type=int, default=96) 
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=96*6)
    parser.add_argument('--d_inner_hid', type=int, default=96*6*2)
    parser.add_argument('--n_position',type=int,default=5000)

    # training params
     
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_warmup_steps', type=int, default=1000)
    parser.add_argument('--lr_mul', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label_smoothing', default = True)

    # output and record
    # parser.add_argument('--output_dir', type=str, default=outputmodel_path)
    parser.add_argument('--use_tb', default=False, action='store_true')
    parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('--ckpt_save_root', type=str, default='./checkpoint')

    # device
    parser.add_argument('--no_cuda', default=False, action='store_true')
    
    # test params
    # data params
    parser.add_argument('--test_path', type=str, default='dataset')
    parser.add_argument('--testsnr_list', nargs='+', type=int, choices=[1,2,4,7], default=[4,7], help='example: --testsnr_list 1 2 7')
    parser.add_argument('--detection_list', nargs='+', default=['ground_truth', 'deepblink_det'],
                    help='List of data types. Example: --detection_list ground_truth deepblink_det')
    # GT path
    parser.add_argument('--GTfolder', type=str, default='dataset/ground_truth')

    parser.add_argument('--det_keep_rate', type=float, default=1.0)
    # save 
    parser.add_argument('--eval_save_path', type=str, default='./prediction/')


    # choose process
    parser.add_argument('--train', default=False, action='store_true')

    parser.add_argument('--tracking', default=False, action='store_true')
    parser.add_argument('--model_ckpt_path', type=str, default=None)
    parser.add_argument('--eval', default=False, action='store_true')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':

    opt = parse_args_()

    print(opt.train)
    print(opt.tracking)
    trainfilename = opt.trainfilename

    assert trainfilename in opt.train_path and trainfilename in opt.val_path

    # data param
    past = opt.len_established
    cand = opt.len_future
    near = opt.near
    opt.num_cand = near**cand
    

    #==================train====================
    now = int(round(time.time()*1000))
    nowname = time.strftime('%Y%m%d_%H_%M_%S',time.localtime(now/1000))
    expname = nowname+'_'+trainfilename.replace(' ','_')+'_ckpt'
    
    if opt.train:
        # make save ckpt folder
        outputmodel_path = os.path.join(opt.ckpt_save_root,expname)
        if not os.path.exists(outputmodel_path):
            os.makedirs(outputmodel_path)
        opt.output_dir = outputmodel_path
        # save params
        save_args_to_file(opt, os.path.join(outputmodel_path,'param.txt'))
        # train
        trainval(opt)

    #==================test=====================
    if opt.tracking and not opt.train:
        assert opt.model_ckpt_path is not None
    #     outputmodel_path = opt.model_ckpt_path
    # print('outputmodel_path')
    # print(outputmodel_path)
    if opt.tracking:
        detection_list = opt.detection_list
        for datatype in detection_list:
            testfilename_list = [trainfilename.replace('1247',str(i)) for i in opt.testsnr_list]
            for testfilename in testfilename_list:
                #==================tracking=====================
                savefolder = os.path.join(nowname+'_'+datatype, testfilename.replace(' ','_'))
                output_path = os.path.join(opt.eval_save_path,savefolder)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                # prepare model ckpt
                if opt.tracking and not opt.train:
                    model_p = opt.model_ckpt_path
                else:
                    model_p = glob.glob(outputmodel_path+'/**.chkpt')[-1].replace('\\','/')
                print(model_p)
                # prepare data path
                test_det_pa = os.path.join(opt.test_path, datatype, testfilename+'.xml')
                # output csv path
                output_csv_pa = os.path.join(output_path, testfilename.replace(' ','_')+'.csv')
                # save params
                save_args_to_file(opt, os.path.join(output_path, testfilename.replace(' ','_')+'_param.txt'))
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

                #==================evaluation=====================
                if opt.eval:
                    # write csv result to xml file for eval
                    xmlfilepath = os.path.join(output_path, testfilename.replace(' ','_')+'.xml')
                    resultcsv_2xml(xmlfilepath, output_csv_pa, testfilename)

                    # prepare gt path, result path, and output path
                    ref = os.path.join(opt.GTfolder,testfilename+'.xml')
                    can = xmlfilepath
                    out = can.replace('xml','txt')

                    subprocess.call(
                        ['java', '-jar', 'trackingPerformanceEvaluation.jar', 
                        '-r', ref, '-c', can,'-o',out])

                    print('[Info] Finish evaluating')
                    print(f'Save file:{out}')