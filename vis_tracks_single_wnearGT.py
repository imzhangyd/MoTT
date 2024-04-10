'''
visulize randomly several single tracks
'''
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import skimage.io as io

import seaborn as sns
import random
import argparse
from utils import readXML, find_near
import numpy as np

__author__ = "Yudong Zhang"


palette = sns.color_palette('hls', 30)
def get_color(seed):
    random.seed(seed)
    # random color
    bbox_color = random.choice(palette)
    bbox_color = [int(255 * c) for c in bbox_color][::-1]
    cl='#'+hex(bbox_color[0])[-2:]+hex(bbox_color[1])[-2:]+hex(bbox_color[2])[-2:]
    cl = cl.upper()
    return cl


def xml2df(xmlfilepath):
    poslist = readXML(xmlfilepath) # x, y, t, z, float(p)
    P = [np.array(_) for _ in poslist]
    M = np.vstack(P)
    detection_total = pd.DataFrame(M[:,[0,1,2,4]])
    detection_total.columns=['pos_x','pos_y','frame','trackid']
    return detection_total



def parse_args_():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--imgfolder', type=str, default='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/challenge/MICROTUBULE snr 7 density low')
    parser.add_argument('--trackcsvpath', type=str, default='./prediction/20240301_15_25_56/track_result.csv')
    parser.add_argument('--vis_save', type=str, default='./prediction/20240301_15_25_56/track_vis')   
    parser.add_argument('--img_fmt', type=str, default='**t{:03d}**.tif')
    parser.add_argument('--vis_dot', default=False, action='store_true' )
    parser.add_argument('--vistrack_number', default=50, type=int)
    parser.add_argument('--vistrack_length', type=int, default=2)

    parser.add_argument('--GTtrackxmlpath', type=str, default='./dataset/tracks10/GTxml/test_2024_04_08__14_44_25.xml')
    parser.add_argument('--visGT_near_num',type=int, default=5)


    parser.add_argument('--vispast_length',type=int, default=None)

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    opt = parse_args_()

    result_pa = opt.trackcsvpath
    imgfolder = opt.imgfolder
    savefolder = opt.vis_save
    os.makedirs(savefolder, exist_ok=True)

    result = pd.read_csv(result_pa,header=0)

    filename = result_pa.split('/')[-1].replace('.csv','')
    
    print('[Info] Start')

    GT_df = xml2df(opt.GTtrackxmlpath)
    alltracks_idx = list(set(result['trackid']))
    alltracks_idx.sort()
    viscount = 0
    for idx_, the_id in enumerate(alltracks_idx):

        print(f"[Info] Processing {idx_}/{len(alltracks_idx)}")
        this_idtrack = result[result['trackid'] == the_id]
        if len(this_idtrack) >= opt.vistrack_length:
            viscount += 1
            print(f"[Info] Finish {viscount}/{opt.vistrack_number}")
            print(f"[Info] Processing {idx_}/{len(alltracks_idx)}")
            if viscount >= opt.vistrack_number:
                break

            this_idtrack = this_idtrack.sort_values('frame')
            for fr in range(int(this_idtrack['frame'].values.min()), int(this_idtrack['frame'].values.max()+1)):
                
                imgpath = glob.glob(os.path.join(imgfolder,opt.img_fmt.format(fr)))
                assert len(imgpath) == 1
                img = io.imread(imgpath[0])
                H,W = img.shape
                plt.figure()
                plt.imshow(img,'gray')
                plt.axis('off')
                # ID_color = get_color(the_id)
                ID_color = 'r'
                # this_iddet = result[result['trackid']==the_id].sort_values('frame')
                this_iddet_near = this_idtrack[(this_idtrack['frame']<=fr)] #&(this_iddet['frame']>fr-10)
                this_iddet_near = this_iddet_near.sort_values(by='frame')
                xlist = [max(min(x, W-2),1) for x in this_iddet_near['pos_x']]
                ylist = [max(min(y, H-2),1) for y in this_iddet_near['pos_y']]
                plt.plot(xlist,ylist,linewidth=0.5,color=ID_color)
                if opt.vis_dot:
                    plt.scatter([xlist[-1]],[ylist[-1]],color=ID_color, marker='o', edgecolors=ID_color, s=1,linewidths=1)
                
                # vis near GT
                thisframe_gtdf = GT_df[GT_df['frame'] == fr]
                nearp = find_near(thisframe_gtdf, x = xlist[-1],y = ylist[-1])
                numnear = min(opt.visGT_near_num, nearp.shape[0])
                for i in range(numnear):
                    onenearid = nearp[i,-2]
                    ID_color = get_color(onenearid) if onenearid != the_id else get_color(onenearid+1)
                    its_tracklet = GT_df[GT_df['trackid'] == onenearid]
                    past_tracklet = its_tracklet[its_tracklet['frame'] <= fr]
                    past_tracklet = past_tracklet.sort_values(by='frame')
                    xlist = [max(min(x, W-2),1) for x in past_tracklet['pos_x']]
                    ylist = [max(min(y, H-2),1) for y in past_tracklet['pos_y']]
                    if opt.vispast_length:
                        if len(xlist)>opt.vispast_length:
                            xlist = xlist[-opt.vispast_length:]
                            ylist = ylist[-opt.vispast_length:]

                    plt.plot(xlist,ylist,'--',linewidth=0.5,color=ID_color)
                        
                    if opt.vis_dot:
                        plt.scatter([xlist[-1]],[ylist[-1]],color=ID_color, marker='x', edgecolors=ID_color, s=0.7,linewidths=0.7)
                

                plt.savefig(os.path.join(savefolder, 'track%d_%03d.jpg'%(the_id,fr)), bbox_inches='tight',dpi=300,pad_inches=0.0)
                plt.close()
        else:
            # print(f'track{the_id} has only one length')
            pass
    #     break
    print('[Info] Success!')