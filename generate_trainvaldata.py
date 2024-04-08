from asyncio import tasks
import numpy as np
import random
import glob
from treelib import Node, Tree
import argparse
import os


__author__ = "Yudong Zhang"


def readXML(file):
    with open(file) as f:
        lines = f.readlines()
    f.close()
    poslist = []
    p = 0
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
            posi.append([x, y, t, z, float(p)])
        elif '</particle>' in lines[i]:
            p += 1
            poslist.append(posi)
    return poslist


def find_near(n,xmlcontent,x,y):
    frame_ind  = n
    all_posi = []

    for panum,paticle in enumerate(xmlcontent):
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


def make_data(xmlcontent, SIG, outputfolder, name, Past, Cand, n_near, frameend):
    
    print('==>Process:{}'.format(name))
    txtoutputname = os.path.join(outputfolder, name)

    for pa_ind,paticle_poslist in enumerate(xmlcontent): # each track
        print('Particle number:{}/{}, with length:{}'.format(pa_ind,len(xmlcontent),len(paticle_poslist)))

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
                    pastposlist.append(
                        [pad_paticle_poslist[line_ind+i][0], # x
                         pad_paticle_poslist[line_ind+i][1], # y
                         pad_paticle_poslist[line_ind+i][3]]) # z
                
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
                            pad_paticle_poslist[line_ind+Past+j][0],
                            pad_paticle_poslist[line_ind+Past+j][1],
                            pad_paticle_poslist[line_ind+Past+j][3]
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
                            xmlcontent=xmlcontent,
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
                                if xmlcontent[rand_i][fra_num][0] == tree.get_node(GTlabel).data[0] and xmlcontent[rand_i][fra_num][1] == tree.get_node(GTlabel).data[1]:
                                    continue
                            numb += 1
                            nodename = tobe_extlabel+str(numb)
                            newnamelist.append(nodename)
                            tree.create_node(
                                tag=nodename, 
                                identifier=nodename, 
                                parent=tobe_extlabel, 
                                data=[
                                    xmlcontent[rand_i][fra_num][0],
                                    xmlcontent[rand_i][fra_num][1],
                                    xmlcontent[rand_i][fra_num][3]
                                    ])
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


    parser.add_argument('--trackxmlpath', type=str, default='./dataset/ground_truth/MICROTUBULE snr 7 density low.xml')
    parser.add_argument('--savefolder', type=str, default='./dataset/ISBI_trainval')

    parser.add_argument('--total_frame', type=int, default=100)
    parser.add_argument('--trainval_splitframe', type=int, default=70)


    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    opt = parse_args_()

    xmlfilepath = opt.trackxmlpath
    savefolder = opt.savefolder

    name = os.path.split(xmlfilepath)[-1].replace('.xml','')
    outputfolder = os.path.join(savefolder, f'past{opt.past_length}_depth{opt.tree_depth}_near{opt.node_number}')
    os.makedirs(outputfolder, exist_ok=True)

    # read xml file 
    xmlcontent = readXML(xmlfilepath)
    # split train val
    xmlcontent_train = []
    xmlcontent_val = []
    for ite in xmlcontent:
        train_ite = []
        val_ite = []
        for line in ite:
            if line[2]<opt.trainval_splitframe:
                train_ite.append(line)
            else:
                val_ite.append(line)
        if len(train_ite) > 0:
            xmlcontent_train.append(train_ite)
        if len(val_ite) > 0:
            xmlcontent_val.append(val_ite)

    if len(xmlcontent_train) > 0:
        make_data(
            xmlcontent_train, 'train', outputfolder, name,
            opt.past_length, opt.tree_depth, opt.node_number, 
            opt.trainval_splitframe)
    else:
        print('No training data was generated!')

    if len(xmlcontent_val) > 0:
        make_data(
            xmlcontent_val, 'val', outputfolder, name,
            opt.past_length, opt.tree_depth, opt.node_number, 
            opt.total_frame)
    else:
        print('No validation data was generated!')
    
