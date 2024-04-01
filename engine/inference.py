'''
This script handles the tracking process.
'''
import time
import torch
import numpy as np
import torch.nn as nn
import time

import pandas as pd
import numpy as np
# from Dataset_match import func_getdataloader_match
from Dataset_oridif_match import func_getdataloader_oridif_match
from engine.visualization import vis_netpred, vis_matchres
from torch import nn
import gurobipy as grb
import time
from treelib import Tree

import numpy as np
import pandas as pd
from utils import load_model, readXML, find_near


__author__ = "Yudong Zhang"


# pulp solver
def probability_maximization_pulp_solver(costs,trackid_list,detid_list):
    '''
    costs: a numpy array (n * 3), look like [[cost0, trackid_0, detid_0], [cost1, trackid_1, detid_1],...]
    trackid_list: trackid_0,trackid_1,...
    detid_list: detid_0,detid_1,...
    '''
    # set variables 
    varis=[]
    for i in range(len(costs)):
        if costs[i][2]==-1:
            vari= str(costs[i][1]) + "_" + 'x'
        else:
            vari= str(costs[i][1]) + "_" + str(costs[i][2])
        varis.append(vari)
    mottVaris=[pulp.LpVariable(f'z_{i}',lowBound=0,upBound=1,cat=pulp.LpBinary) for i in varis]


    # set objective
    mottProDp=pulp.LpProblem("mottProDp",sense=pulp.const.LpMaximize)
    ss=[]
    for varIndex in range(len(mottVaris)):
        s=mottVaris[varIndex] * costs[varIndex][0]
        ss.append(s)
    mottProDp += pulp.lpSum(ss), "Objective Function"

    # set constraints
    for j in trackid_list:
        flag = False
        exprc=[]
        for cc in range(len(costs)): 
             if costs[cc][1] == j: 
                 flag = True
                 exprc.append(mottVaris[cc])
        if flag:
            mottProDp += (pulp.lpSum(exprc)==1)


    for j in detid_list:
        exprcj=[]
        flag = False
        for cc in range(len(costs)): 
             if costs[cc][2] == j: 
                 flag = True
                 exprcj.append(mottVaris[cc])
        if flag:
            mottProDp += pulp.lpSum(exprcj)<=1
    
    # set solver and optimize
    pulp_dir = os.path.dirname(pulp.__file__)
    solver = pulp.COIN_CMD(path=os.path.join(pulp_dir, 'solverdir/cbc/linux/64/cbc',msg=False))
    mottProDp.solve(solver)

    # analyze solution
    solutions = []
    for v in mottProDp.variables():
        if v.varValue>0.5:
            tempv=str(v)
            sp1=tempv.split('_')[1]
            sp2=tempv.split('_')[2]
            if sp2=='x':
                tempv='z_'+sp1+'_-1'
                solutions.append(tempv)
            else:
                solutions.append(str(v))
    return solutions


# gurobi solver
def probability_maximization_gurobi_solver(costs, trackid_list, detid_list):
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
    Near: int, the detection number to select at each depth, node number at each depth
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
                padding_before.append([p_ind]+[-1]*len(one_track_list[0]))
        convert_one_track = []
        for b in one_track_list:
            convert_one_track.append(b+[0])
        pad_paticle_poslist = padding_before+convert_one_track

        if convert_one_track[-1][-2] < end_frame:
            pastposlist = []
            for i in range(-Past,0):
                pastposlist.append([
                    pad_paticle_poslist[i][1],
                    pad_paticle_poslist[i][2],
                    pad_paticle_poslist[i][3],
                    pad_paticle_poslist[i][4],
                    pad_paticle_poslist[i][5],
                    pad_paticle_poslist[i][6],
                    pad_paticle_poslist[i][8]
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
                    if frame_ind__ > end_frame  or len(detection_total[detection_total['frame']==frame_ind__]) == 0:
                        np_re = []
                    else:
                        np_re = find_near(  #pos_x, pos_y, left, right, top, bottom, frame, det_id, dis
                            pdcontent=detection_total[detection_total['frame']==frame_ind__],
                            x=near_objpos[0],
                            y=near_objpos[1])
                    
                    numb = 0
                    neednull = 1

                    # extend node using near position
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
                            data=[ppos[0],ppos[1],ppos[2],ppos[3],ppos[4],ppos[5],0])
                        if numb == Near-neednull:
                            break
                    
                    # if need dumn detections, add them
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
                            data=[-1]*len(pastposlist[0]))
                    if SIG:
                        det_id_4cand_reserve = det_id_4cand.copy()
                        SIG = False
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
            
            # # Check all items are different
            # if convert_one_track[-1][3] < (end_frame-(Cand-1)):
            #     str_candlist = [str(_) for _ in all_candidate]
            #     assert len(str_candlist) == len(set(str_candlist)), print(f'Warning: Generated candidate duplicates! \
            #             \n last livetrack frame:{convert_one_track[-1][3]}, Cand:{Cand}, end_frame-(Cand-1):{end_frame-(Cand-1)} \
            #             \n generated cand list: {all_candidate}')

        thistrack_dic['trackid'] = one_trackid
        thistrack_dic['cand5_id'] = det_id_4cand_reserve
        thistrack_dic['pastpos'] = pastposlist
        thistrack_dic['cand25'] = all_candidate
        
        one_frame_match_list.append(thistrack_dic)

    return one_frame_match_list


def tracking(input_detfile,output_trackcsv,model_path,fract,Past,Cand,Near,
             track_buffer=10,new_track_thresh=0.6, track_high_thresh=0.6, track_low_thresh=0.1,
             mean_=None,std_=None,vis=False,no_cuda=False,imagefolder='./dataset/MOT17/train',solver_name='pulp'):

    print('[Info] Start tracking {}'.format(time.strftime('%Y%m%d_%H_%M_%S',time.localtime(int(round(time.time()*1000))/1000))))

    # prepare model
    opt = {}
    opt['model'] = model_path
    if not no_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    transformer = load_model(opt, device)
    transformer.eval()

    # prepare detection data
    detection_total_ori = pd.read_csv(input_detfile,index_col=0) #.iloc[:,[6,7,8,2,3,4,5,0]]
    detection_total = detection_total_ori[detection_total_ori['conf']>track_high_thresh]
    detection_total = detection_total[[
        'pos_x','pos_y','bb_left','bb_top','bb_width','bb_height','frame']]
    detection_total['bb_right'] = detection_total['bb_left'] + detection_total['bb_width']
    detection_total['bb_bottom'] = detection_total['bb_top'] + detection_total['bb_height']
    detection_total = detection_total[['pos_x','pos_y','bb_left','bb_right','bb_top','bb_bottom','frame']]
    
    imgx = detection_total['pos_x'].max()
    imgy = detection_total['pos_y'].max()

    # detection_total = detection_total.sample(frac=fract,replace=False,random_state=1,axis=0)
    # detection_total.reset_index(drop=True,inplace=True)
    detection_total['det_id'] = detection_total.index

    start_frame = min(list(detection_total['frame']))
    end_frame = max(list(detection_total['frame']))
    print('[Info] Tracking range: frame {}----frame {}'.format(start_frame, end_frame))
    # Initialization the live tracklet set and keep set
    established_track = pd.DataFrame(columns=['trackid','pos_x','pos_y','bb_left','bb_right','bb_top','bb_bottom','frame'])   
    keep_track = pd.DataFrame(columns=['trackid','pos_x','pos_y','bb_left','bb_right','bb_top','bb_bottom','frame'])
    print('[Info] Finish prepare total det')


    this_frame = start_frame
    while(this_frame<end_frame):
        if this_frame%20 == 0:
            print('[Info] Process Frame {}-{}'.format(this_frame,this_frame+1))

        this_det = detection_total[detection_total['frame'] == this_frame]
        next_det = detection_total[detection_total['frame'] == this_frame + 1]

        # 0. Initialization livetracklet set at frist frame
        if this_frame == start_frame or (len(established_track) == 0 and len(this_det)>0):
            # print('===>>> Step0: Initialization')
            established_track = this_det[['det_id','pos_x','pos_y','bb_left','bb_right','bb_top','bb_bottom','frame']]
            established_track = established_track.rename(columns={'det_id':'trackid'})
            # make a hold record set
            temp = np.zeros([len(this_det),2])
            temp[:,0] = this_det['det_id']
            established_track_HOLDnum = pd.DataFrame(temp)
            established_track_HOLDnum.columns = ['trackid','HOLDnum']

        if len(established_track) == 0:
            this_frame += 1
            continue
        
        # 1. For each livetracklet, make candidate live trackelts by constructing hypothesis tree
        # print('===>>> Step1: make tracklets')
        one_frame_match_list = make_tracklets(established_track, detection_total, this_frame, end_frame, Past, Near, Cand)

        # 2. predict matching probabilities between livetracklet and candidate tracklets, and predict future state(next position and existence probability)
        # print('===>>> Step2: predict matching probilities')
        this_frame_dataloader,this_frame_data = func_getdataloader_oridif_match(
            one_frame_match_list,
            batch_size=len(one_frame_match_list), # prediction all at once
            shuffle=False, 
            num_workers=1,
            mean=mean_,
            std=std_)
        
        for batch in this_frame_dataloader:
            src_seq = batch[0].float().to(device)
            trg_seq = batch[1].float().to(device)
            trackid_batch = batch[2].tolist()
            cand5id_batch = batch[3].tolist()
            pred_shift,pred_prob = transformer(src_seq, trg_seq)
            src_seq_abpos = batch[-1].float()

        lastab = src_seq_abpos[:,-1,:2].numpy()
        shrink = nn.MaxPool1d(kernel_size=Near**(Cand-1), stride=Near**(Cand-1))
        soft = nn.Softmax(dim=-1)
        pred_prob5 = shrink(pred_prob.unsqueeze(0)).squeeze(0)
        soft_pred_prob5 = soft(pred_prob5).detach().cpu().numpy().tolist()
        norm_pred_prob5 = (pred_prob5-pred_prob5.min())+0.0001
        norm_pred_prob5 = (norm_pred_prob5/norm_pred_prob5.max()).detach().cpu().numpy().tolist()
        # record the classfication number
        pred_value,pred_clsnum = torch.max(pred_prob,1)
        pred_nextone_detid  = torch.Tensor(cand5id_batch)[
            [range(len(cand5id_batch))],pred_clsnum // Near**(Cand-1)][0].detach().cpu().numpy().reshape(-1,1)
        
        # record the future state prediction
        pred_shift_next1 = pred_shift[:,0,:-1].detach().cpu().numpy()*this_frame_data.std +this_frame_data.mean
        pred_shift_exist_flag = pred_shift[:,0,-1].detach().cpu().numpy().reshape(-1,1)
        pred_shift_id_np = np.concatenate([
            np.array(trackid_batch).reshape(-1,1),
            pred_shift_next1,pred_shift_exist_flag,
            pred_nextone_detid
            ],-1)
        pred_shift_id_pd = pd.DataFrame(pred_shift_id_np)
        pred_shift_id_pd.columns = [
            'trackid',
            'shift_x',
            'shift_y',
            'shift_left',
            'shift_right',
            'shift_top',
            'shift_bottom',
            'shift_width',
            'shift_height',
            'abs_x',
            'abs_y',
            'abs_left',
            'abs_right',
            'abs_top',
            'abs_botoom',
            'abs_width',
            'abs_height',
            'exist_flag',
            'p_detid']

        if vis:
            thisframeimage = vis_netpred(input_detfile,
                this_frame,
                trackid_batch,
                src_seq_abpos,
                pred_nextone_detid,
                trg_seq,
                Near,
                Cand,
                this_frame_data,
                lastab,
                cand5id_batch,
                pred_prob5,
                output_trackcsv,
                imagefolder
                )
        # 3. construct a discrecte optimazation problem, and solve it to obtain the optimal matching
        # print('===>>> Step3: find the overall best matching')
        costlist = []
        for it in range(len(one_frame_match_list)):
            for m in range(Near):
                costlist.append([norm_pred_prob5[it][m], trackid_batch[it], cand5id_batch[it][m]]) #

        # solutions = probability_maximization_solver(costlist, trackid_batch, list(next_det['det_id']))
        if solver_name == 'gurobi':
            solutions = probability_maximization_gurobi_solver(costlist, trackid_batch, list(next_det['det_id']))
        else:
            solutions = probability_maximization_pulp_solver(costlist, trackid_batch, list(next_det['det_id']))
       

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
            # linked_det_id.append(link_cand_id)
            
            p_detid = pred_shift_id_pd[pred_shift_id_pd['trackid'] == link_track_id]['p_detid'].values[0]
            if link_cand_id != p_detid:
                dif_cls_gurobi_changeto_1 = True
            # vis 

            use_guribimatchdet = False
            use_fillup = False
            dist_changeto_1 = False
            dif_cls_gurobi_changeto_1 = False
            temp_dic = {}
            
            if link_cand_id != -1:
                # if Warning
                thisid_track = established_track[established_track.trackid==link_track_id]
                last_x = thisid_track.iloc[-1,1]
                last_y = thisid_track.iloc[-1,2]

                ext = next_det[next_det['det_id'] == link_cand_id]
                next_x = ext.iloc[0,0]
                next_y = ext.iloc[0,1]
                dist = np.sqrt((next_x-last_x)**2 + (next_y-last_y)**2)
                # ext = ext[['det_id','pos_x','pos_y','frame']]
                ext = ext[['det_id','pos_x','pos_y','bb_left','bb_right','bb_top','bb_bottom','frame']]
                ext = ext.rename(columns={'det_id':'trackid'})
                ext['trackid'] = link_track_id
                established_track = established_track.append(ext)
                linked_det_id.append(link_cand_id)
                established_track_HOLDnum.loc[established_track_HOLDnum['trackid'] == link_track_id, 'HOLDnum'] = 0
                use_guribimatchdet = True
            
            stop_4_outview = False
            stop_4_overhold = False
            stop_4_predstop = False
            stop_4_moviestop = False
            
            if link_cand_id == -1: 
                thisid_HOLDnum = established_track_HOLDnum[established_track_HOLDnum.trackid == link_track_id].iloc[0,1]
                thisid_pred_shift = pred_shift_id_pd[pred_shift_id_pd.trackid == link_track_id]
                thisid_track = established_track[established_track.trackid==link_track_id]
                if (thisid_HOLDnum < track_buffer) and (this_frame < end_frame-1): # and (thisid_pred_shift.iloc[0,3]>pred_shift_id_pd['exist_flag'].mean()): 
                    established_track_HOLDnum.loc[established_track_HOLDnum['trackid'] == link_track_id, 'HOLDnum'] = thisid_HOLDnum+1
                    
                    last_frame = thisid_track.iloc[-1,-1]
                    last_x = thisid_track.iloc[-1,1]
                    last_y = thisid_track.iloc[-1,2]
                    last_left = thisid_track.iloc[-1,3]
                    last_right = thisid_track.iloc[-1,4]
                    last_top = thisid_track.iloc[-1,5]
                    last_bottom = thisid_track.iloc[-1,6]

                    if len(thisid_track) < 2:
                        last_shift_x = 0
                        last_shift_y = 0
                        last_shift_left   = 0
                        last_shift_right  = 0
                        last_shift_top    = 0
                        last_shift_bottom = 0
                    else:
                        last_shift_x = thisid_pred_shift.iloc[0,1]
                        last_shift_y = thisid_pred_shift.iloc[0,2]
                        last_shift_left   = thisid_pred_shift.iloc[0,3]
                        last_shift_right  = thisid_pred_shift.iloc[0,4]
                        last_shift_top    = thisid_pred_shift.iloc[0,5]
                        last_shift_bottom = thisid_pred_shift.iloc[0,6]

                    shift_x = last_shift_x
                    shift_y = last_shift_y
                    shift_left  =last_shift_left  
                    shift_right =last_shift_right 
                    shift_top   =last_shift_top   
                    shift_bottom=last_shift_bottom
                    
                    if last_x+shift_x < imgx and last_y+shift_y < imgy and last_x+shift_x >0 and last_y+shift_y>0:
                        temp_dic = {
                            'trackid'  :[link_track_id],
                            'pos_x'    :[last_x+shift_x],
                            'pos_y'    :[last_y+shift_y],
                            'bb_left'  :[last_left  +shift_left  ],
                            'bb_right' :[last_right +shift_right ],
                            'bb_top'   :[last_top   +shift_top   ],
                            'bb_bottom':[last_bottom+shift_bottom],
                            'frame':[last_frame+1]
                            }
                        ext  = pd.DataFrame(temp_dic)
                        established_track = established_track.append(ext)
                        established_track_HOLDnum.loc[established_track_HOLDnum['trackid'] == link_track_id, 'HOLDnum'] = thisid_HOLDnum+1
                        use_fillup = True
                    else: # outer the image, delete
                        tobeapp = established_track[established_track['trackid']==link_track_id]
                        keep_track = keep_track.append(tobeapp)
                        established_track = established_track[established_track['trackid'] != link_track_id] # 这样就相当于去掉了那个
                        stop_4_outview = True
                        
                else:
                    stop_4_overhold = True if thisid_HOLDnum >= track_buffer else False
                    stop_4_moviestop = True if this_frame >= (end_frame-1) else False

                    thisholdnum = thisid_HOLDnum
                    if thisholdnum > 0:
                        tobeapp = established_track[
                            established_track['trackid']==link_track_id].iloc[:-int(thisholdnum),:]
                    else:
                        tobeapp = established_track[
                            established_track['trackid']==link_track_id]

                    keep_track = keep_track.append(tobeapp)
                    established_track = established_track[established_track['trackid'] != link_track_id]

            # vis
            if vis:
                vis_matchres(this_frame, link_track_id, output_trackcsv, link_cand_id, p_detid, thisframeimage,next_det,
                    use_guribimatchdet, dif_cls_gurobi_changeto_1, dist_changeto_1, use_fillup, stop_4_moviestop,
                    stop_4_outview, stop_4_predstop, stop_4_overhold,temp_dic)
                    
        for to_belinkid in list(next_det['det_id']): 
            if to_belinkid not in linked_det_id:
                itsconf = detection_total_ori.iloc[to_belinkid]['conf']
                if itsconf > new_track_thresh:
                    ext = next_det[next_det['det_id'] == to_belinkid]
                    ext = ext[['det_id','pos_x','pos_y','bb_left','bb_right','bb_top','bb_bottom','frame']]
                    ext = ext.rename(columns={'det_id':'trackid'})
                    established_track = established_track.append(ext)

                    temp_dic = {'trackid':[ext.iloc[0,0]],'HOLDnum':[0]}  
                    temp_pd = pd.DataFrame(temp_dic)
                    established_track_HOLDnum = established_track_HOLDnum.append(temp_pd)

        if this_frame%20 == 0:
            print('[Info] Success {}'.format(time.strftime('%Y%m%d_%H_%M_%S',time.localtime(int(round(time.time()*1000))/1000))))

        this_frame += 1

    # last frame
    keep_track = keep_track.append(established_track)
    keep_track.to_csv(output_trackcsv)

    print('[Info] Finish!') 
