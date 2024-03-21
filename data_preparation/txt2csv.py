'''
Convert the Mot17 original .txt to .csv files.

'''
import glob
import pandas as pd
import argparse
import os

__author__ = "Yudong Zhang"


def parse_args_():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txtpath_format', type=str, default='./dataset/MOT17Labels/train/**FRCNN/gt/gt.txt')
    parser.add_argument('--outputfolder',type=str, default='./dataset/MOT17_trainval_test/gt_pedescsv')
    opt = parser.parse_args()
    return opt


def onetxt2csv(gttxtpa):
    this_txt = pd.read_csv(gttxtpa,header=None)
    this_txt.columns = ['frame','id','bb_left','bb_top','bb_width','bb_height','conf','cls','vis']
    this_txt_pedestrain = this_txt[this_txt['cls']==1]
    this_txt_pedestrain.loc[:, 'pos_x'] = this_txt_pedestrain.loc[:,'bb_left'] + this_txt_pedestrain.loc[:,'bb_width'] / 2
    this_txt_pedestrain.loc[:, 'pos_y'] = this_txt_pedestrain.loc[:,'bb_top']  + this_txt_pedestrain.loc[:,'bb_height'] / 2

    this_txt_pedestrain = this_txt_pedestrain.reset_index(drop=True)
    return this_txt_pedestrain


if __name__ == '__main__':

    opt = parse_args_()
    os.makedirs(opt.outputfolder, exist_ok=True)

    gttxt_pathlist = glob.glob(opt.txtpath_format)

    for gttxtpa in gttxt_pathlist:
        print(f'[Info]Process:{gttxtpa}')

        os.path.split(gttxtpa)[-1]
        this_txt_pedestrain = onetxt2csv(gttxtpa)
        num = gttxtpa.split('/')[-3].split('-')[1]
        savepa = os.path.join(opt.outputfolder, num+'gt_pedes.csv')
        this_txt_pedestrain.to_csv(savepa)
        
    print('Finish')
        
    

