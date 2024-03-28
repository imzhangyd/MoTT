'''
When you have only patial labels of tracks, and you have all the detections labels,
you can use this file to generate train and val format files for training model. 

'''

from asyncio import tasks
import numpy as np
import random
import glob
from treelib import Node, Tree
import argparse
import os
import pandas as pd

__author__ = "Yudong Zhang"


def readXML(file):
    with open(file) as f:
        lines = f.readlines()
    f.close()
    poslist = []
    p = 0
    maxframe = 0
    for i in range(len(lines)):
        if '<particle>' in lines[i]:
            posi = []
        elif '<detection t=' in lines[i]:
            ind1 = lines[i].find('"')
            ind2 = lines[i].find('"', ind1 + 1)
            t = float(lines[i][ind1 + 1:ind2])
            ind1 = lines[i].find('"', ind2 + 1)
            ind2 = lines[i].find('"', ind1 + 1)
            x = float(lines[i][ind1 + 1:ind2])
            ind1 = lines[i].find('"', ind2 + 1)
            ind2 = lines[i].find('"', ind1 + 1)
            y = float(lines[i][ind1 + 1:ind2])
            ind1 = lines[i].find('"', ind2 + 1)
            ind2 = lines[i].find('"', ind1 + 1)
            z = float(lines[i][ind1 + 1:ind2])
            if t > maxframe: maxframe = t
            posi.append([x, y, t, z, float(p)])
        elif '</particle>' in lines[i]:
            p += 1
            poslist.append(posi)
    return poslist, maxframe



def read_trackmatecsv(file):
    thcsv = pd.read_csv(file, header=0,index_col=None)
    maxframe = thcsv['FRAME'].values.max()
    trackid = list(set(thcsv['TRACK_ID']))
    poslist = []
    for t_id in trackid:
        posi = []
        t_df = thcsv[thcsv['TRACK_ID']==t_id]
        t_df = t_df.sort_values('FRAME')
        for i in range(len(t_df)):
            x = t_df.iloc[i].loc['POSITION_X']
            y = t_df.iloc[i].loc['POSITION_Y']
            t = t_df.iloc[i].loc['FRAME']
            z = 0.0
            p = t_df.iloc[i].loc['TRACK_ID']
            posi.append([x, y, t, z, float(p)])
        poslist.append(posi)
    return poslist, maxframe


def read_detnormalcsv(file):
    thcsv = pd.read_csv(file, header=0,index_col=None)
    maxframe = thcsv['frame'].values.max()
    # trackid = list(set(thcsv['id']))
    poslist = []
    # for t_id in trackid:
    posi = []
    # t_df = thcsv[thcsv['TRACK_ID']==t_id]
    thcsv = thcsv.sort_values('frame')
    for i in range(len(thcsv)):
        x = thcsv.iloc[i].loc['pos_x']
        y = thcsv.iloc[i].loc['pos_y']
        t = thcsv.iloc[i].loc['frame']
        z = 0.0
        p = 0.0
        posi.append([x, y, t, z, float(p)])
    poslist.append(posi)
    return poslist, maxframe


def find_near(n,detlabelcontent,x,y):
    frame_ind  = n
    all_posi = []

    for panum,paticle in enumerate(detlabelcontent):
        np_parpos = np.array(paticle)
        frampos = np.where(np_parpos[:,2]== frame_ind)
        if len(frampos[0])>0:
            for i in range(len(frampos[0])):
                all_posi.append([panum,frampos[0][i],np_parpos[frampos[0][i],0], np_parpos[frampos[0][i],1]])
    
    dis_all_posi = []
    for thisframepos in all_posi:
        dis = (thisframepos[2]-x)**2 +(thisframepos[3]-y)**2
        dis_all_posi.append(thisframepos+[dis])
    dis_all_posi_np = np.array(dis_all_posi)
    a_arg = np.argsort(dis_all_posi_np[:,-1])
    sortnp = dis_all_posi_np[a_arg.tolist()]

    return sortnp


def make_data(tracklabelcontent, detlabelcontent, SIG, outputfolder, name, Past, Cand, n_near, frameend):
    
    print('==>Process:{}'.format(name))
    txtoutputname = os.path.join(outputfolder, name)

    for pa_ind,paticle_poslist in enumerate(tracklabelcontent): # each track
        print('Particle number:{}/{}, with length:{}'.format(pa_ind,len(tracklabelcontent),len(paticle_poslist)))

        if len(paticle_poslist) >= 1+Cand:
            print('It will generate {} samples.'.format(len(paticle_poslist)))

            # ===========padding=============
            first_frame = paticle_poslist[0][2]
            last_frame = paticle_poslist[-1][2]
            p_ind = paticle_poslist[0][-1]
            padding_before = []
            for a in range(Past-1,0,-1):
                padding_before.append([-1,-1,first_frame-a,-1,p_ind])
            padding_after = []
            for b in range(Cand):
                padding_after.append([-1,-1,last_frame+b+1,-1,p_ind])
            pad_paticle_poslist = padding_before+paticle_poslist+padding_after

            p_poslist_len = len(paticle_poslist)

            
            
            line_ind = -1
            for line_ind in range(p_poslist_len): # shifting windows
                pastposlist = []
                # =============past==============
                for i in range(Past):
                    pastposlist.append([pad_paticle_poslist[line_ind+i][0],pad_paticle_poslist[line_ind+i][1],pad_paticle_poslist[line_ind+i][3]])
                
                # ===========GT candidate===========
                tree = Tree()
                tree.create_node(tag='ext', identifier='ext', data=pastposlist[-1])
                for j in range(Cand):
                    nodename = 'ext'+'1'*(j+1)
                    tree.create_node(tag=nodename, identifier=nodename, parent='ext'+'1'*(j), data=[pad_paticle_poslist[line_ind+Past+j][0],pad_paticle_poslist[line_ind+Past+j][1],pad_paticle_poslist[line_ind+Past+j][3]])
                
                # ===========other candidate===========
                frame_ind = pad_paticle_poslist[line_ind+Past][2]
                frame_indnext_list = [frame_ind+t for t in range(Cand)]

                # avoid search over bounding frame
                if frame_indnext_list[-1]>=frameend:
                    continue

                nodenamelist = ['ext']
                for frame_ind__ in frame_indnext_list:
                    newnamelist = []
                    for tobe_extlabel in nodenamelist:

                        # confirm the current position
                        thisnodedata = tree.get_node(tobe_extlabel).data
                        if thisnodedata[-1] == -1:
                            parentnodelabel = tobe_extlabel
                            for _ in range(Cand):
                                parentnodelabel = tree.parent(parentnodelabel).tag
                                parentnode = tree.get_node(parentnodelabel)
                                if parentnode.data[-1] != -1:
                                    near_objpos = parentnode.data.copy()
                                    break
                        else:
                            near_objpos = thisnodedata.copy()

                        # find near position next
                        np_re = find_near(n=frame_ind__,detlabelcontent=detlabelcontent,x=near_objpos[0],y=near_objpos[1])

                        # children number of this node
                        if tree.children(tobe_extlabel) == []: # null
                            numb = 0
                            neednull = 1
                            notequalGT = 0
                        else: # has a GT children
                            nodename = tobe_extlabel+str(1)
                            newnamelist.append(nodename)
                            numb = 1
                            # Is the GT children real or dumn detection
                            if tree.children(tobe_extlabel)[0].data[-1] == -1:
                                neednull = 0
                                notequalGT = 0
                            else:
                                neednull = 1
                                notequalGT = 1
                        
                        # extend node using near position
                        for ppos in np_re:
                            rand_i = int(ppos[0])
                            fra_num = int(ppos[1])
                            # judge whether same as GT，if yes then skip it
                            if notequalGT:
                                GTlabel = tobe_extlabel+str(1)
                                if 9 > (detlabelcontent[rand_i][fra_num][0] - tree.get_node(GTlabel).data[0])**2 + (detlabelcontent[rand_i][fra_num][1] - tree.get_node(GTlabel).data[1])**2:
                                    continue
                            numb += 1
                            nodename = tobe_extlabel+str(numb)
                            newnamelist.append(nodename)
                            tree.create_node(
                                tag=nodename, 
                                identifier=nodename, 
                                parent=tobe_extlabel, 
                                data=[detlabelcontent[rand_i][fra_num][0],detlabelcontent[rand_i][fra_num][1],detlabelcontent[rand_i][fra_num][3]])
                            if numb == n_near-neednull: # fill the defined number
                                break

                        # if need dumn detections, add them
                        if numb < n_near-neednull:
                            neednull = n_near-numb
                        
                        for _ in range(neednull):
                            numb += 1
                            nodename = tobe_extlabel+str(numb)
                            newnamelist.append(nodename)
                            tree.create_node(
                                tag=nodename, 
                                identifier=nodename, 
                                parent=tobe_extlabel, 
                                data=[-1,-1,-1])

                    # all node of one depth
                    nodenamelist = newnamelist.copy()
                # convert to list
                all_candidate = []
                paths_leaves = [path_[1:] for path_ in tree.paths_to_leaves()]
                for onepath in paths_leaves:
                    onepath_data = []
                    for onepos in onepath:
                        onepath_data.append(tree.get_node(onepos).data)
                    all_candidate.append(onepath_data)
                
                # Check all items are different
                str_candlist = [str(_) for _ in all_candidate]
                assert len(str_candlist) == len(set(str_candlist)),print('Warning: Generated candidate duplicates!{}-{}'.format(paticle_poslist,line_ind))

                # Shuffle
                n_bat = n_near**(Cand-1)
                total_choice= [all_candidate[i*n_bat:(i+1)*n_bat] for i in range(n_near)]

                assert len(total_choice) == n_near, print(len(total_choice))
                total_choice_rand = []
                GT1 = 0
                GT2 = 0
                gt_random = False
                for it in total_choice:
                    indexlist = range(len(total_choice))
                    randomindexlist = random.sample(indexlist,len(indexlist))
                    if not gt_random:
                        # print(randomindexlist)
                        GT2 = np.where(np.array(randomindexlist) == 0)[0].item()
                        gt_random = True
                    temp = [it[hh] for hh in randomindexlist]
                    total_choice_rand.append(temp)

                indexlist = range(len(total_choice))
                randomindexlist = random.sample(indexlist,len(indexlist))
                GT1 = np.where(np.array(randomindexlist) == 0)[0].item()
                final_total = [total_choice_rand[kk] for kk in randomindexlist]

                GT_num = GT1*len(total_choice)+GT2

                # Write to file
                f = open(txtoutputname+'_{}.txt'.format(SIG),'a+')
                f.write(str(pastposlist)+'s')
                for key in final_total:
                    f.write(str(key)+'s')
                # f.write(str(dist25out)+'s')
                f.write(str(int(GT_num)))
                f.write('s'+str(frame_ind)+'\n')
                f.close()


def parse_args_():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--past_length', type=int, default=7)
    parser.add_argument('--tree_depth', type=int, default=2)
    parser.add_argument('--node_number', type=int, default=5)


    parser.add_argument('--tracklabelpath', type=str, default='/mnt/data1/ZYDdata/lwj/200609 lamp-1mch int2s 013all.csv')
    parser.add_argument('--detlabelpath', type=str, default='/ldap_shared/home/s_zyd/dl_particle_detection/test/DL_Particle_Detection/detfor_track/lamp_vesicel4_0.99/200609 lamp-1mch int2s 013.csv')
    parser.add_argument('--track_label_file_type', type=str, default='trackmatecsv', choices=['xml','trackmatecsv'])
    parser.add_argument('--det_label_file_type', type=str, default='normalcsv', choices=['xml','trackmatecsv','normalcsv'])


    parser.add_argument('--savefolder', type=str, default='./dataset/lamp_trainval')

    parser.add_argument('--trainval_splitrate', type=float, default=0.7)


    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    opt = parse_args_()

    tracklabelpath = opt.tracklabelpath
    savefolder = opt.savefolder

    name = os.path.split(tracklabelpath)[-1].replace('.xml','').replace('.csv','')
    outputfolder = os.path.join(savefolder, f'past{opt.past_length}_depth{opt.tree_depth}_near{opt.node_number}')
    os.makedirs(outputfolder, exist_ok=True)

    # read xml file 
    if opt.track_label_file_type == 'xml':
        tracklabelcontent,maxframe  = readXML(tracklabelpath)
    elif opt.track_label_file_type == 'trackmatecsv':
        tracklabelcontent,maxframe = read_trackmatecsv(tracklabelpath)
    else:
        raise KeyError
    
    trainval_splitframe = int(maxframe * opt.trainval_splitrate)

    # split train val
    tracklabel_content_train = []
    tracklabel_content_val = []
    for ite in tracklabelcontent:
        train_ite = []
        val_ite = []
        for line in ite:
            if line[2]<trainval_splitframe:
                train_ite.append(line)
            else:
                val_ite.append(line)
        if len(train_ite) > 1:
            tracklabel_content_train.append(train_ite)
        if len(val_ite) > 1:
            tracklabel_content_val.append(val_ite)



    # detection labels
    if opt.det_label_file_type == 'normalcsv':
        detlabelcontent,_ = read_detnormalcsv(opt.detlabelpath)


    make_data(
        tracklabel_content_train, detlabelcontent, 'train', outputfolder, name,
        opt.past_length, opt.tree_depth, opt.node_number, 
        trainval_splitframe)
    
    make_data(
        tracklabel_content_val, detlabelcontent, 'val', outputfolder, name,
        opt.past_length, opt.tree_depth, opt.node_number, 
        maxframe)
    
