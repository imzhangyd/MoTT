import numpy as np
import random
import glob
from treelib import Node, Tree
import pandas as pd
import argparse
import os

__author__ = "Yudong Zhang"


def readCSV(file):
    thcsv = pd.read_csv(file)
    maxframe = thcsv['frame'].values.max()
    trackid = list(set(thcsv['id']))
    poslist = []
    for t_id in trackid:
        posi = []
        t_df = thcsv[thcsv['id']==t_id]
        t_df = t_df.sort_values('frame')
        for i in range(len(t_df)):
            x = t_df.iloc[i,-2]
            y = t_df.iloc[i,-1]
            t = t_df.iloc[i,1]
            z = 0.0
            p = t_df.iloc[i,2]
            left = t_df.iloc[i,3]
            right = left + t_df.iloc[i,5]
            top = t_df.iloc[i,4]
            bottom = top + t_df.iloc[i,6]
            posi.append([x, y, t,left,right,top,bottom,z, float(p)])
        poslist.append(posi)
    return poslist, maxframe



def find_near(n,csvcontent,x,y):
    frame_ind  = n
    all_posi = []

    for panum,paticle in enumerate(csvcontent):
        np_parpos = np.array(paticle)
        frampos = np.where(np_parpos[:,2]== frame_ind)
        if len(frampos[0])>0:
            all_posi.append([panum,frampos[0].item(),np_parpos[frampos[0].item(),0], np_parpos[frampos[0].item(),1]])

    dis_all_posi = []
    for thisframepos in all_posi:
        dis = (thisframepos[2]-x)**2 +(thisframepos[3]-y)**2
        dis_all_posi.append(thisframepos+[dis])
    dis_all_posi_np = np.array(dis_all_posi)
    a_arg = np.argsort(dis_all_posi_np[:,-1])
    sortnp = dis_all_posi_np[a_arg.tolist()]

    return sortnp


def make_data(csvcontent, SIG, outputfolder, name, Past, Cand, n_near, frameend):
    
    print('==>Process:{}'.format(name))
    txtoutputname = os.path.join(outputfolder, name)

    for pa_ind, paticle_poslist in enumerate(csvcontent):
        print('Particle number:{}/{}, with length:{}'.format(pa_ind,len(csvcontent),len(paticle_poslist)))
        if len(paticle_poslist) >= 1+Cand:
            print('It will generate {} samples.'.format(len(paticle_poslist)))

            # ===========padding=============
            first_frame = paticle_poslist[0][2]
            last_frame = paticle_poslist[-1][2]
            p_ind = paticle_poslist[0][-1]
            padding_before = []
            for a in range(Past-1,0,-1):
                padding_before.append([-1,-1,first_frame-a]+[-1]*5+[p_ind])  #-1*5 -> left right top bottom z
            padding_after = []
            for b in range(Cand):
                padding_after.append([-1,-1,last_frame+b+1]+[-1]*5+[p_ind])
            pad_paticle_poslist = padding_before+paticle_poslist+padding_after

            p_poslist_len = len(paticle_poslist)

            
            
            line_ind = -1
            for line_ind in range(p_poslist_len): # shifting windows
                pastposlist = []
                # =============past==============
                for i in range(Past):
                    pastposlist.append(
                        [pad_paticle_poslist[line_ind+i][0], # x
                        pad_paticle_poslist[line_ind+i][1], # y
                        pad_paticle_poslist[line_ind+i][3], # left
                        pad_paticle_poslist[line_ind+i][4], # right
                        pad_paticle_poslist[line_ind+i][5], # top
                        pad_paticle_poslist[line_ind+i][6], # bottom
                        pad_paticle_poslist[line_ind+i][7], # z
                        ])
                # ===========GT candidate===========
                tree = Tree()
                tree.create_node(tag='ext', identifier='ext', data=pastposlist[-1])
                for j in range(Cand):
                    nodename = 'ext'+'1'*(j+1)
                    tree.create_node(
                        tag=nodename, 
                        identifier=nodename, 
                        parent='ext'+'1'*(j), 
                        data=[
                            pad_paticle_poslist[line_ind+Past+j][0],# x
                            pad_paticle_poslist[line_ind+Past+j][1],# y
                            pad_paticle_poslist[line_ind+Past+j][3],# left
                            pad_paticle_poslist[line_ind+Past+j][4],# right
                            pad_paticle_poslist[line_ind+Past+j][5],# top
                            pad_paticle_poslist[line_ind+Past+j][6],# bottom
                            pad_paticle_poslist[line_ind+Past+j][7],# z
                            ])

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
                        np_re = find_near(
                            n=frame_ind__,
                            csvcontent=csvcontent,
                            x=near_objpos[0],
                            y=near_objpos[1]
                            )

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
                                if csvcontent[rand_i][fra_num][0] == tree.get_node(GTlabel).data[0] and csvcontent[rand_i][fra_num][1] == tree.get_node(GTlabel).data[1]:
                                    continue
                            numb += 1
                            nodename = tobe_extlabel+str(numb)
                            newnamelist.append(nodename)
                            tree.create_node(
                                tag=nodename, 
                                identifier=nodename, 
                                parent=tobe_extlabel, 
                                data=[
                                    csvcontent[rand_i][fra_num][0],# x
                                    csvcontent[rand_i][fra_num][1],# y
                                    csvcontent[rand_i][fra_num][3],# left
                                    csvcontent[rand_i][fra_num][4],# right
                                    csvcontent[rand_i][fra_num][5],# top
                                    csvcontent[rand_i][fra_num][6],# bottom
                                    csvcontent[rand_i][fra_num][7],# z
                                    ])
                            if numb == n_near-neednull: # fill the defined number
                                break

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
                                data=[-1]*7)

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
                if len(str_candlist) != len(set(str_candlist)):
                    print('Skip! Generated candidate duplicates!{}-{}'.format(paticle_poslist,line_ind))
                    continue
                
                # Shuffle all
                indexlist = range(len(all_candidate))
                randomindexlist = random.sample(indexlist,len(indexlist))
                GT_num = np.where(np.array(randomindexlist) == 0)[0].item()
                random_all_cand = [all_candidate[hh] for hh in randomindexlist]

                f = open(txtoutputname+'_{}.txt'.format(SIG),'a+')
                f.write(str(pastposlist)+'s')
                for key in random_all_cand:
                    f.write(str(key)+'s')
                # f.write(str(dist25out)+'s')
                f.write(str(int(GT_num)))
                f.write('s'+str(frame_ind)+'\n')
                f.close()


def parse_args_():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--past_length', type=int, default=7)
    parser.add_argument('--tree_depth', type=int, default=1)
    parser.add_argument('--node_number', type=int, default=5)


    parser.add_argument('--csvpath', type=str, default='./dataset/MOT17_trainval_test/gt_pedescsv/13gt_pedes.csv')
    parser.add_argument('--savefolder', type=str, default='./dataset/MOT17_trainval_test/trainval_box_onefuture')

    parser.add_argument('--trainvalsplit_ratio', type=int, default=0.7)


    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    opt = parse_args_()

    csvpath = opt.csvpath
    trainval_ratio = opt.trainvalsplit_ratio

    savefolder = opt.savefolder
    name = os.path.split(csvpath)[-1].replace('.csv','')
    outputfolder = os.path.join(savefolder, f'past{opt.past_length}_depth{opt.tree_depth}_near{opt.node_number}')
    os.makedirs(outputfolder, exist_ok=True)

    
    
    # read csv file 
    csvcontent, maxframe = readCSV(csvpath)
    # split train val
    split_threshold = int(maxframe*trainval_ratio)
    movielenth = maxframe
    csvcontent_train = []
    csvcontent_val = []
    for ite in csvcontent:
        train_ite = []
        val_ite = []
        for line in ite:
            if line[2]<split_threshold:
                train_ite.append(line)
            else:
                val_ite.append(line)
        if len(train_ite) > 0:
            csvcontent_train.append(train_ite)
        if len(val_ite) > 0:
            csvcontent_val.append(val_ite)


    make_data(
        csvcontent_train, 'train', outputfolder, name,
        opt.past_length, opt.tree_depth, opt.node_number, 
        split_threshold)
    
    make_data(
        csvcontent_val, 'val', outputfolder, name,
        opt.past_length, opt.tree_depth, opt.node_number, 
        maxframe)
    
