'''
This script handles the training, tracking and evaluation.
'''
import glob
import argparse

import numpy as np
import random
import os

import torch
import torch.optim as optim

from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim


from Dataset import func_getdataloader
import pandas as pd
import numpy as np
import torch

import numpy as np
import pandas as pd
import subprocess
from engine.trainval import train
from engine.inference import tracking


__author__ = "Yudong Zhang"


def trainval(opt):

    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # For reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
        
    device = torch.device('cuda')

    print('==>init dataLoader')
    batch_size = opt.batch_size
    ins_loader_train,traindata = func_getdataloader(txtfile=opt.train_path, batch_size=batch_size, shuffle=True, num_workers=16)
    ins_loader_val,valdata = func_getdataloader(txtfile=opt.val_path, batch_size=batch_size, shuffle=True, num_workers=16)
    print('\tTraining data number:{}'.format(len(traindata)))
    print('\tValidation data number:{}'.format(len(valdata)))

    print('==>init transformer')
    transformer = Transformer(
        n_passed = opt.len_established,
        n_future = opt.len_future,
        n_candi = opt.num_cand,
        n_position = opt.n_position,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout)
    transformer_ins = transformer.to(device)

    print('==>init optimizer')
    optimizer = ScheduledOptim(
        optim.Adam(transformer_ins.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)
    
    print('==>start train')
    train(transformer_ins, ins_loader_train,len(traindata), ins_loader_val,len(valdata), optimizer, device, opt)


if __name__ == '__main__':

    trainfilename = 'VESICLE snr 1247 density low'
    print(trainfilename)
    # data param
    past = 7
    cand=2
    near = 5
    # network param
    n_layer_ = 1
    n_head_ = 6
    d_kv_ = 96
    d_model_ = n_head_*d_kv_
    ffn_ = 2*d_model_
    # optim param
    warmup_ = 1000
    batch_ = 64
    # datapath outputpath
    traindata_path = 'dataset/20220406_exp_mergesnr_trainTFT/'+trainfilename+'_train.txt'.format(past,cand,near)
    valdata_path = 'dataset/20220406_exp_mergesnr_trainTFT/'+trainfilename+'_val.txt'.format(past,cand,near)
    outputmodel_path = './outputmodel_obtainresult/'+trainfilename.replace(' ','_')
    if not os.path.exists(outputmodel_path):
        os.makedirs(outputmodel_path)
    # train function
        
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_path', default=traindata_path)   
    parser.add_argument('-val_path', default=valdata_path)     
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-b', '--batch_size', type=int, default=batch_)
    parser.add_argument('-d_model', type=int, default=d_model_) 
    parser.add_argument('-n_position',type=int,default=5000) 
    parser.add_argument('-len_established',type=int,default=past) 
    parser.add_argument('-len_future',type=int,default=cand) 
    parser.add_argument('-num_cand',type=int,default=near**cand) 
    parser.add_argument('-d_inner_hid', type=int, default=ffn_) 
    parser.add_argument('-d_k', type=int, default=d_kv_) 
    parser.add_argument('-d_v', type=int, default=d_kv_) 
    parser.add_argument('-n_head', type=int, default=n_head_) 
    parser.add_argument('-n_layers', type=int, default=n_layer_) 
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=warmup_)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight',default = True)
    parser.add_argument('-proj_share_weight', default = True)
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')
    parser.add_argument('-output_dir', type=str, default=outputmodel_path)
    parser.add_argument('-use_tb', default=True)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', default = False)
    parser.add_argument('-label_smoothing', default = True)

    opt = parser.parse_args()

    trainval(opt)

    
    #==================test=====================
    #  test link
    datatype_list = ['ground_truth','deepblink_det']
    # datatype_list = ['ground_truth']
    for datatype in datatype_list:
        testfilename_list = [trainfilename.replace('1247',str(i)) for i in [7,1]]
        for testfilename in testfilename_list:
            print(testfilename)
            output_path = './230213debug1outresult_'+datatype+'/'+testfilename.replace(' ','_')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            fract_ = 1.0
            model_p = glob.glob(outputmodel_path+'/**.chkpt')[-1].replace('\\','/')
            # model_p = './trainonSNR7model/20220307_11_23_12.chkpt'
            test_det_pa = 'dataset/'+datatype+'/'+testfilename+'.xml'
            output_csv_pa = output_path+'/'+testfilename.replace(' ','_')+'.csv'
            keep_track = tracking(
                input_detxml=test_det_pa,
                output_trackcsv=output_csv_pa,
                model_path = model_p,
                fract=fract_,
                Past = past,
                Cand=cand,
                Near=near
                )


            snr = testfilename.split(' ')[2]
            dens = testfilename.split(' ')[-1]
            scenario = testfilename.split(' ')[0]
            method= '_TFT'
            thrs = 0

            filepath = output_path+'/'+testfilename.replace(' ','_')+'.xml'
            track_csv = output_path+'/'+testfilename.replace(' ','_')+'.csv'
            result_csv = pd.read_csv(track_csv)
            t_trackid = list(set(result_csv['trackid']))

            # csv to xml
            with open(filepath, "w+") as output:
                output.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
                output.write('<root>\n')
                output.write('<TrackContestISBI2012 SNR="' + str(
                    snr) + '" density="' + dens + '" scenario="' + scenario + \
                            '" ' + method + '="' + str(thrs) + '">\n')
                
                for trackid in t_trackid:
                    thistrack = result_csv[result_csv['trackid']==trackid]
                    if len(thistrack) > 1:
                        thistrack.sort_values("frame",inplace=True)
                        thistrack_np = thistrack.values

                        output.write('<particle>\n')
                        for pos in thistrack_np:
                            output.write('<detection t="' + str(int(pos[-1])) +
                                        '" x="' + str(pos[2]) +
                                        '" y="' + str(pos[3]) + '" z="0"/>\n')
                        output.write('</particle>\n')
                output.write('</TrackContestISBI2012>\n')
                output.write('</root>\n')
                output.close()


            ref = 'dataset/ground_truth/'+testfilename+'.xml'
            can = filepath
            o = can.replace('xml','txt')

            subprocess.call(
                ['java', '-jar', 'trackingPerformanceEvaluation.jar', 
                '-r', ref, '-c', can,'-o', o])

            print(testfilename)
            print(fract_)
            print(model_p)
            print(filepath)