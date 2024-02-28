'''
This script handles the tracking process.
'''
import time

import numpy as np
import torch.nn as nn
import time

import pandas as pd
import numpy as np
from Dataset_match import func_getdataloader_match
from torch import nn
import gurobipy as grb
import time
from treelib import Tree

import numpy as np
import pandas as pd
from utils import load_model, readXML, find_near


__author__ = "Yudong Zhang"


def probability_maximization_solver(costs, trackid_list, detid_list):
    '''
    costs: a numpy array (n * 3), look like [[cost0, trackid_0, detid_0], [cost1, trackid_1, detid_1],...]
    '''
    # print('===>>>start construct model')
    # Create a new model
    model = grb.Model("mip1")
    model.Params.outputflag = 0

    # set variables
    ro = [] 
    for i in range(len(costs)):
        ro.append(model.addVar(0.0, 1.0, 0.0, grb.GRB.BINARY,'z_' + str(costs[i][1]) + '_' + str(costs[i][2]))) 
    model.update()
    
    # set objective
    expr = grb.LinExpr()
    for i in range(len(ro)):
        expr.add(ro[i], costs[i][0]) 
    model.setObjective(expr, grb.GRB.MAXIMIZE) 

    # set constraints
    nrConstraint = 0
    exprcs = []
    for j in trackid_list: 
        exprc = grb.LinExpr()
        flag = False
        for cc in range(len(costs)): 
            if costs[cc][1] == j: 
                exprc.add(ro[cc], 1.0)
                flag = True
        nrConstraint += 1 
        exprcs.append(exprc)
        if flag: 
            model.addConstr(exprc, grb.GRB.EQUAL, 1.0, "c_" + str(nrConstraint)) 

    for j in detid_list:
        exprc = grb.LinExpr()
        flag = False
        for cc in range(len(costs)):
            if costs[cc][2] == j:
                exprc.add(ro[cc], 1.0)
                flag = True
        nrConstraint += 1
        exprcs.append(exprc)
        if flag:
            model.addConstr(exprc,grb.GRB.LESS_EQUAL,1.0, "c_" + str(nrConstraint))

    # print('===>>>Finish construct model')
    model.optimize()
    # print('===>>>Finish optimize')
    assert model.status == grb.GRB.Status.OPTIMAL

    solutionIndex = []
    solutions = []
    for i in range(len(ro)):
        if ro[i].Xn > 0.5: # 0 or 1, get 1 links
            solutionIndex.append(i)
            solutions.append(ro[i].VarName)
    
    return solutions


def make_tracklets(established_track, detection_total, this_frame, end_frame, Past, Near, Cand):
    '''
    This function is used to generate candidate tracklets for each live tracklet.

    established_track: DataFrame, the live tracklet set
    detection_total: DataFrame, detections in all frames

    this_frame: int, this frame number
    end_frame: int, the last frame number

    Past: int, the livetracklet length
    Near: int, the detection number to select at each depth, equals (node number - 1) at each depth
    Cand: int, the depth of tree
    '''

    t_trackid = set(established_track['trackid'])
    one_frame_match_list = []
    for one_trackid in t_trackid:
        thistrack_dic = {}
        one_track = established_track[established_track['trackid']==one_trackid]
        one_track_length = len(one_track)
        one_track_list = one_track.values.tolist()
        p_ind = one_track_list[0][0] 
        padding_before = []
        if one_track_length < Past:
            for _ in range(Past-one_track_length):
                padding_before.append([p_ind,-1,-1,-1,-1])
        convert_one_track = []
        for b in one_track_list:
            convert_one_track.append(b+[0])
        pad_paticle_poslist = padding_before+convert_one_track

        if convert_one_track[-1][3] < end_frame:
            pastposlist = []
            for i in range(-Past,0): 
                pastposlist.append([
                    pad_paticle_poslist[i][1],
                    pad_paticle_poslist[i][2],
                    pad_paticle_poslist[i][4]
                    ])

            tree = Tree()
            tree.create_node(tag='ext', identifier='ext', data=pastposlist[-1])
            
            frame_ind = this_frame+1 
            frame_indnext_list = [frame_ind+t for t in range(Cand)]

            nodenamelist = ['ext']
            SIG = True
            for frame_ind__ in frame_indnext_list:
                newnamelist = []
                for tobe_extlabel in nodenamelist: 
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
                    
                    if frame_ind__ > end_frame: 
                        np_re = []
                    else:
                        np_re = find_near(pdcontent=detection_total[detection_total['frame']==frame_ind__],x=near_objpos[0],y=near_objpos[1])
                    
                    numb = 0
                    neednull = 1

                    det_id_4cand = []
                    for ppos in np_re: 
                        det_id_4cand.append(int(ppos[-2]))
                        numb += 1
                        nodename = tobe_extlabel+str(numb)
                        newnamelist.append(nodename)
                        tree.create_node(
                            tag=nodename, 
                            identifier=nodename, 
                            parent=tobe_extlabel, 
                            data=[ppos[0],ppos[1],0]
                            )
                        if numb == Near-neednull:
                            break
                    if numb < Near-neednull:
                        neednull = Near-numb

                    for _ in range(neednull):
                        det_id_4cand.append(-1)
                        numb += 1
                        nodename = tobe_extlabel+str(numb)
                        newnamelist.append(nodename)
                        tree.create_node(
                            tag=nodename, 
                            identifier=nodename, 
                            parent=tobe_extlabel, 
                            data=[-1,-1,-1])
                    if SIG:
                        det_id_4cand_reserve = det_id_4cand.copy()
                        SIG = False
                nodenamelist = newnamelist.copy()
            all_candidate = []
            paths_leaves = [path_[1:] for path_ in tree.paths_to_leaves()]
            for onepath in paths_leaves:
                onepath_data = []
                for onepos in onepath:
                    onepath_data.append(tree.get_node(onepos).data)
                all_candidate.append(onepath_data)
            
            if convert_one_track[-1][3] < (end_frame-(Cand-1)):
                str_candlist = [str(_) for _ in all_candidate]
                assert len(str_candlist) == len(set(str_candlist)), print(f'Warning: Generated candidate duplicates! \
                        \n last livetrack frame:{convert_one_track[-1][3]}, Cand:{Cand}, end_frame-(Cand-1):{end_frame-(Cand-1)} \
                        \n generated cand list: {all_candidate}')

        thistrack_dic['trackid'] = one_trackid
        thistrack_dic['cand5_id'] = det_id_4cand_reserve
        thistrack_dic['pastpos'] = pastposlist
        thistrack_dic['cand25'] = all_candidate
        
        one_frame_match_list.append(thistrack_dic)

    return one_frame_match_list


def tracking(input_detxml,output_trackcsv,model_path,fract,Past,Cand,Near):

    print('[Info] Start tracking {}'.format(time.strftime('%Y%m%d_%H_%M_%S',time.localtime(int(round(time.time()*1000))/1000))))

    # prepare model
    opt = {}
    opt['model'] = model_path
    # device = 'cpu'
    device = 'cuda'
    transformer = load_model(opt, device)
    transformer.eval()

    # prepare detection data
    test_track_xml = input_detxml
    pos_list_all = readXML(test_track_xml) 

    P = [np.array(_) for _ in pos_list_all]
    M = np.vstack(P)
    detection_total = pd.DataFrame(M[:,:3])
    detection_total.columns=['pos_x','pos_y','frame']

    detection_total = detection_total.sample(frac=fract,replace=False,random_state=1,axis=0)
    detection_total.reset_index(drop=True,inplace=True)
    detection_total['det_id'] = detection_total.index

    start_frame = min(list(detection_total['frame']))
    end_frame = max(list(detection_total['frame']))
    print('[Info] Tracking range: frame {}----frame {}'.format(start_frame, end_frame))
    # Initialization the live tracklet set and keep set
    keep_track = pd.DataFrame(columns=['trackid','pos_x','pos_y','frame'])
    print('[Info] Finish prepare total det')


    this_frame = start_frame
    while(this_frame<end_frame):
        print('[Info] Process Frame {}-{}'.format(this_frame,this_frame+1))

        this_det = detection_total[detection_total['frame'] == this_frame]
        next_det = detection_total[detection_total['frame'] == this_frame + 1]

        # 0. Initialization livetracklet set at frist frame
        if this_frame == start_frame:
            # print('===>>> Step0: Initialization')
            established_track = this_det[['det_id','pos_x','pos_y','frame']]
            established_track = established_track.rename(columns={'det_id':'trackid'})
            # make a hold record set
            temp = np.zeros([len(this_det),2])
            temp[:,0] = this_det['det_id']
            established_track_HOLDnum = pd.DataFrame(temp)
            established_track_HOLDnum.columns = ['trackid','HOLDnum']

        # 1. For each livetracklet, make candidate live trackelts by constructing hypothesis tree
        # print('===>>> Step1: make tracklets')
        one_frame_match_list = make_tracklets(established_track, detection_total, this_frame, end_frame, Past, Near, Cand)

        # 2. predict matching probabilities between livetracklet and candidate tracklets, and predict future state(next position and existence probability)
        # print('===>>> Step2: predict matching probilities')
        this_frame_dataloader,this_frame_data = func_getdataloader_match(
            one_frame_match_list, 
            batch_size=len(one_frame_match_list), # prediction all at once
            shuffle=False, 
            num_workers=1)
        
        for batch in this_frame_dataloader:
            src_seq = batch[0].float().to(device)
            trg_seq = batch[1].float().to(device)
            trackid_batch = batch[2].tolist()
            cand5id_batch = batch[3].tolist()
            pred_shift,pred_prob = transformer(src_seq, trg_seq)

        shrink = nn.MaxPool1d(kernel_size=Near**(Cand-1), stride=Near**(Cand-1))
        soft = nn.Softmax(dim=-1)
        pred_prob5 = shrink(pred_prob.unsqueeze(0)).squeeze(0)
        soft_pred_prob5 = soft(pred_prob5).detach().cpu().numpy().tolist()

        # record the future state prediction
        pred_shift_next1 = pred_shift[:,0,:-1].detach().cpu().numpy()*this_frame_data.std +this_frame_data.mean
        pred_shift_exist_flag = pred_shift[:,0,-1].detach().cpu().numpy().reshape(-1,1)
        pred_shift_id_np = np.concatenate([np.array(trackid_batch).reshape(-1,1),pred_shift_next1,pred_shift_exist_flag],-1)
        pred_shift_id_pd = pd.DataFrame(pred_shift_id_np)
        pred_shift_id_pd.columns = ['trackid','shift_x','shift_y','exist_flag']

        # 3. construct a discrecte optimazation problem, and solve it to obtain the optimal matching
        # print('===>>> Step3: find the overall best matching')
        costlist = []
        for it in range(len(one_frame_match_list)):
            for m in range(Near):
                costlist.append([soft_pred_prob5[it][m], trackid_batch[it], cand5id_batch[it][m]])

        solutions = probability_maximization_solver(costlist, trackid_batch, list(next_det['det_id']))
       

        # 4. process the solution and manage tracklet set
        # real tracklet -- real det : link
        # real tracklet -- dumy det
        #       if hold num < hold_thre and existence > existence_thre：
        #                link
        #       else: stop and save to keep set
        # null -- real det: Initialize new live tracklet
        # print('===>>> Step4: track management')
        linked_det_id = []
        for so in solutions:
            link_track_id = int(so.split('_')[1])
            link_cand_id = int(so.split('_')[2])
            linked_det_id.append(link_cand_id)
            
            if link_cand_id != -1: 
                ext = next_det[next_det['det_id'] == link_cand_id]
                ext = ext[['det_id','pos_x','pos_y','frame']]
                ext = ext.rename(columns={'det_id':'trackid'})
                ext['trackid'] = link_track_id
                established_track = established_track.append(ext)
                established_track_HOLDnum.loc[established_track_HOLDnum['trackid'] == link_track_id, 'HOLDnum'] = 0
            elif link_cand_id == -1: 
                thisid_HOLDnum = established_track_HOLDnum[established_track_HOLDnum.trackid == link_track_id].iloc[0,1]
                thisid_pred_shift = pred_shift_id_pd[pred_shift_id_pd.trackid == link_track_id]
                
                if (thisid_HOLDnum <10) and (this_frame < end_frame-1) and (thisid_pred_shift.iloc[0,3]>pred_shift_id_pd['exist_flag'].mean()): 
                    established_track_HOLDnum.loc[established_track_HOLDnum['trackid'] == link_track_id, 'HOLDnum'] = thisid_HOLDnum+1
                    thisid_track = established_track[established_track.trackid==link_track_id]
                    last_frame = thisid_track.iloc[-1,-1]
                    last_x = thisid_track.iloc[-1,1]
                    last_y = thisid_track.iloc[-1,2]
                    shift_x = thisid_pred_shift.iloc[0,1]
                    shift_y = thisid_pred_shift.iloc[0,2]
                    temp_dic = {'trackid':[link_track_id],'pos_x':[last_x+shift_x],'pos_y':[last_y+shift_y],'frame':[last_frame+1]}
                    ext  = pd.DataFrame(temp_dic)
                    established_track = established_track.append(ext)
                else:
                    thisholdnum = thisid_HOLDnum
                    if thisholdnum > 0:
                        tobeapp = established_track[established_track['trackid']==link_track_id].iloc[:-int(thisholdnum),:]
                    else: 
                        tobeapp = established_track[established_track['trackid']==link_track_id]
                    keep_track = keep_track.append(tobeapp)
                    # established_track.reset_index(drop=True,inplace=True)
                    # established_track = established_track.drop(established_track[established_track['trackid']==link_track_id].index)
                    established_track = established_track[established_track['trackid']!=link_track_id]


        for to_belinkid in list(next_det['det_id']): 
            if to_belinkid not in linked_det_id:
                ext = next_det[next_det['det_id'] == to_belinkid]
                ext = ext[['det_id','pos_x','pos_y','frame']]
                ext = ext.rename(columns={'det_id':'trackid'})
                established_track = established_track.append(ext)

                temp_dic = {'trackid':[ext.iloc[0,0]],'HOLDnum':[0]}  
                temp_pd = pd.DataFrame(temp_dic)
                established_track_HOLDnum = established_track_HOLDnum.append(temp_pd)


        print('[Info] Success {}'.format(time.strftime('%Y%m%d_%H_%M_%S',time.localtime(int(round(time.time()*1000))/1000))))

        this_frame += 1

    # last frame
    keep_track = keep_track.append(established_track)
    keep_track.to_csv(output_trackcsv)

    print('[Info] Finish!') 
