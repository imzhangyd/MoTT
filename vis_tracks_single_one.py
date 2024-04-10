'''
visulize one single track
'''
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import skimage.io as io

import seaborn as sns
import random
import argparse


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


def parse_args_():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--imgfolder', type=str, default='/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/challenge/MICROTUBULE snr 7 density low')
    parser.add_argument('--trackcsvpath', type=str, default='./prediction/20240301_15_25_56/track_result.csv')
    parser.add_argument('--vis_save', type=str, default='./prediction/20240301_15_25_56/track_vis')   
    parser.add_argument('--img_fmt', type=str, default='**t{:03d}**.tif')
    parser.add_argument('--vis_dot', default=False, action='store_true' )
    parser.add_argument('--vistrack_id', type=int, nargs='+', default=[])
    # parser.add_argument('--vistrack_length', type=int, default=2)

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
    # alltracks_idx = list(set(result['trackid']))
    # alltracks_idx.sort()
    alltracks_idx = opt.vistrack_id
    # viscount = 0
    for idx_, the_id in enumerate(alltracks_idx):

        print(f"[Info] Processing {idx_}/{len(alltracks_idx)}")
        this_idtrack = result[result['trackid'] == the_id]
        assert len(this_idtrack) > 0
        # if len(this_idtrack) >= opt.vistrack_length:
            # viscount += 1
            # print(f"[Info] Finish {viscount}/{opt.vistrack_number}")
            # print(f"[Info] Processing {idx_}/{len(alltracks_idx)}")
            # if viscount >= opt.vistrack_number:
            #     break
        this_idtrack = this_idtrack.sort_values('frame')
        for fr in range(int(this_idtrack['frame'].values.min()), int(this_idtrack['frame'].values.max()+1)):

            imgpath = glob.glob(os.path.join(imgfolder,opt.img_fmt.format(fr)))
            assert len(imgpath) == 1

            img = io.imread(imgpath[0])
            H,W = img.shape
            plt.figure()
            plt.imshow(img,'gray')
            plt.axis('off')
            ID_color = 'r' #get_color(the_id)
            # this_iddet = result[result['trackid']==the_id].sort_values('frame')
            this_iddet_near = this_idtrack[(this_idtrack['frame']<=fr)] #&(this_iddet['frame']>fr-10)
            xlist = [max(min(x, W-2),1) for x in this_iddet_near['pos_x']]
            ylist = [max(min(y, H-2),1) for y in this_iddet_near['pos_y']]
            plt.plot(xlist,ylist,linewidth=0.5,color=ID_color)
            if opt.vis_dot:
                plt.scatter([xlist[-1]],[ylist[-1]],color=ID_color, marker='o', edgecolors=ID_color, s=1,linewidths=1)

            plt.savefig(os.path.join(savefolder, 'track%d_%03d.jpg'%(the_id,fr)), bbox_inches='tight',dpi=300,pad_inches=0.0)
            plt.close()
        # else:
            # print(f'track{the_id} has only one length')
            # pass
    #     break
    print('[Info] Success!')