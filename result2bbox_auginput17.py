'''
Postprocess of tracking results.

'''


__author__ = 'Yudong Zhang'


import glob
import pandas as pd
import numpy as np
import shutil
import os
import argparse

def interpolation_convert(det_result_pathlist,interpolation,detcsvfolder):
    if interpolation:
        for predcsvpa in det_result_pathlist:
        # predcsvpa = det_result_pathlist[0]
            print(predcsvpa)
            this_filename = os.path.split(predcsvpa)[-1].split('_link.csv')[0]
            # this_filename
            detcsvpa = os.path.join(detcsvfolder,this_filename+'.csv')
            # detori
            pred_csv = pd.read_csv(predcsvpa)
            det_csv = pd.read_csv(detcsvpa)
            pred_csv.columns = ['det_id','trackid','pos_x','pos_y',\
                'bb_left','bb_right','bb_top','bb_bottom','frame']
            pred_csv = pred_csv[['det_id','trackid','pos_x','pos_y','frame']]
            det_csv.columns = ['det_id']+list(det_csv.columns)[1:]
            pred_csv['bb_left'] = 0
            pred_csv['bb_top'] = 0
            pred_csv['bb_width'] = 0
            pred_csv['bb_height'] = 0
            # print(pred_csv.head())
            # print(det_csv.head())
            for tt_id in list(set(pred_csv['trackid'].values.tolist())):
                one_track = pred_csv[pred_csv['trackid']==tt_id].sort_values('frame')
                
                for i in range(len(one_track)):
                    # one_track = pred_csv[pred_csv['trackid']==tt_id].sort_values('frame')
                    det_d = one_track.iloc[i,0]
                    frame_ = one_track.iloc[i,4]
                    ind = int(one_track.iloc[i].name)

                    if not (det_d ==0 and frame_ > 1):
                        pred_csv.iloc[ind,-4:] = det_csv[det_csv['det_id'] == det_d].iloc[0,3:7].values.tolist()

                one_track = pred_csv[pred_csv['trackid']==tt_id].sort_values('frame')
                tofill = one_track[(one_track['det_id']==0)&(one_track['frame']>1)]
                if len(tofill) != 0:
                    tofillframe = tofill['frame'].values
                    dif_tofillframe = tofillframe[1:] - tofillframe[:-1]
                    dif_tofillframe = np.array([2]+list(dif_tofillframe)+[2])
                    sec_num = (dif_tofillframe>1).sum()
                    indtofill = np.where(dif_tofillframe>1)[0]
                    start_end_sec = []
                    # for ss in indtofill:
                    #     gap = dif_tofillframe[ss]
                    #     assert gap > 1
                    #     startframe = tofillframe[ss]
                    #     endframe = tofillframe[ss]+gap
                    #     start_end_sec.append([startframe,endframe])

                    for ss in range(len(indtofill)-1):
                        start = indtofill[ss]
                        end = indtofill[ss+1]-1
                        framestart = tofillframe[start]
                        frameend = tofillframe[end]
                        start_end_sec.append([framestart,frameend])

                    for sec in start_end_sec:
                        left_frame = sec[0]-1
                        right_frame = sec[1]+1
                        if right_frame < one_track['frame'].values.max():
                            left_bbox = one_track[one_track['frame'] == left_frame].iloc[0,-4:].values
                            right_bbox = one_track[one_track['frame'] == right_frame].iloc[0,-4:].values
                            
                            for curr_frame in range(int(left_frame+1),int(right_frame)):
                                one_track_oneframe = one_track[one_track['frame'] == curr_frame]
                                ind = int(one_track_oneframe.iloc[0].name)
                                curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / \
                                                        (right_frame - left_frame) + left_bbox
                                pred_csv.iloc[ind,-4:] = curr_bbox.tolist()
                        else:
                            left_bbox = one_track[one_track['frame'] == left_frame].iloc[0,-4:].values
                            
                            for curr_frame in range(int(left_frame+1),int(right_frame)):
                                one_track_oneframe = one_track[one_track['frame'] == curr_frame]
                                ind = int(one_track_oneframe.iloc[0].name)
                                pred_csv.iloc[ind,-4:] = left_bbox.tolist()


            pred_csv['conf'] = 1
            pred_csv['x'] = -1
            pred_csv['y'] = -1
            pred_csv['z'] = -1

            tobe_out = pred_csv.loc[:, \
                ['frame','trackid','bb_left','bb_top','bb_width','bb_height','conf','x','y','z'] \
                    ].sort_values('frame')
            tobe_out[['frame']] = tobe_out[['frame']].astype(int)
            
            # delete tracks with length=1
            for theid in list(set(tobe_out['trackid'])):
                tempdf = tobe_out[tobe_out['trackid'] == theid]
                if len(tempdf) == 1:
                    tobe_out = tobe_out.drop(tobe_out[tobe_out['trackid']==theid].index)
            outputpath_ = predcsvpa.replace('_link.csv','.txt')
            outputfolder = os.path.join(os.path.split(outputpath_)[0],'data')
            os.makedirs(outputfolder,exist_ok=True)
            outpath_ = os.path.join(outputfolder,os.path.split(outputpath_)[1])
            tobe_out.to_csv(outpath_,header = None, index = None)
            
        re_filelist = glob.glob(outputfolder+'/**.txt')
        for file in re_filelist:
            shutil.copy(file,file.replace('-FRCNN','-DPM'))
            shutil.copy(file,file.replace('-FRCNN','-SDP'))

    else:
        for predcsvpa in det_result_pathlist:
            print(predcsvpa)
            this_filename = os.path.split(predcsvpa)[-1].split('_link.csv')[0]
            # this_filename
            # detcsvpa = 'D:/research/tracking/20220401_data_MOT/result_MOT17/YOLOX_det/'+ \
            #     this_filename+'.csv'
            detcsvpa = os.path.join(detcsvfolder,this_filename+'.csv')
            # detori
            pred_csv = pd.read_csv(predcsvpa)
            det_csv = pd.read_csv(detcsvpa)
            pred_csv.columns = ['det_id','trackid','pos_x','pos_y',\
                'bb_left','bb_right','bb_top','bb_bottom','frame']

            pred_csv['bb_width']  = pred_csv['bb_right'] - pred_csv['bb_left']
            pred_csv['bb_height'] = pred_csv['bb_bottom'] - pred_csv['bb_top']
            # print(pred_csv.head())
            # print(det_csv.head())

            pred_csv['conf'] = 1
            pred_csv['x'] = -1
            pred_csv['y'] = -1
            pred_csv['z'] = -1

            tobe_out = pred_csv.loc[:, \
                ['frame','trackid','bb_left','bb_top','bb_width','bb_height','conf','x','y','z'] \
                    ].sort_values('frame')
            tobe_out[['frame']] = tobe_out[['frame']].astype(int)
            
            # delete tracks with length=1
            for theid in list(set(tobe_out['trackid'])):
                tempdf = tobe_out[tobe_out['trackid'] == theid]
                if len(tempdf) == 1:
                    tobe_out = tobe_out.drop(tobe_out[tobe_out['trackid']==theid].index)
            outputpath_ = predcsvpa.replace('_link.csv','.txt')
            outputfolder = os.path.join(os.path.split(outputpath_)[0],'data')
            os.makedirs(outputfolder,exist_ok=True)
            outpath_ = os.path.join(outputfolder,os.path.split(outputpath_)[1])
            tobe_out.to_csv(outpath_,header = None, index = None)
            
        re_filelist = glob.glob(outputfolder+'/**.txt')
        for file in re_filelist:
            shutil.copy(file,file.replace('-FRCNN','-DPM'))
            shutil.copy(file,file.replace('-FRCNN','-SDP'))


def parse_args_():
    parser = argparse.ArgumentParser()
    
    # data params
    parser.add_argument('--resultfolder',type=str,default='./prediction/20240321_17_12_34') 
    parser.add_argument('--interpolation',default=False,action='store_true') 
    parser.add_argument('--detcsvfolder',type=str,default='/ldap_shared/home/s_zyd/MoTT/dataset/yolox_det_all')

    opt = parser.parse_args()
    return opt



if __name__ == '__main__':

    opt = parse_args_()
    resultfolder= opt.resultfolder
    det_result_pathlist = glob.glob(os.path.join(resultfolder, '**FRCNN_link.csv'))
    assert len(det_result_pathlist) > 0
    interpolation = opt.interpolation
    detcsvfolder = opt.detcsvfolder

    interpolation_convert(det_result_pathlist,interpolation,detcsvfolder)

