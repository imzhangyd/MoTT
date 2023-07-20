'''
This script handles the training process.
'''
import glob
import argparse
import math
import time
from turtle import up

from tqdm import tqdm
import numpy as np
import random
import os
import torch.nn as nn
import time

import torch
import torch.optim as optim

from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim


from Dataset import func_getdataloader
import pandas as pd
import numpy as np
import torch
from transformer.Models import Transformer
from Dataset_match import func_getdataloader_match
from torch import nn
import gurobipy as grb
import time
from treelib import Node, Tree

import numpy as np
import pandas as pd
import subprocess



def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, total_loss_prob,total_loss_dist = 0, 0, 0
    total_accuracy = 0
    total_accuracy_5 = 0

    desc = '  - (Training)   '
    Loss_func = nn.CrossEntropyLoss()
    Loss_func_dist = nn.MSELoss()

    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):
        
        # prepare data
        src_seq = batch[0].float().to(device)
        trg_seq = batch[1].float().to(device)
        label_shift = batch[2].float().to(device)
        label_prob = batch[3].to(device)

        # forward
        optimizer.zero_grad()
        pred_shift,pred_prob = model(src_seq, trg_seq) 
        loss_prob = Loss_func(pred_prob,label_prob)
        loss_dist = Loss_func_dist(pred_shift,label_shift)
        loss = loss_prob+loss_dist

        loss.backward()
        optimizer.step_and_update_lr()

        total_loss_prob += loss_prob.item()*pred_shift.shape[0]
        total_loss_dist += loss_dist.item()*pred_shift.shape[0]
        total_loss += loss.item()*pred_shift.shape[0]
        # calculate accuracy
        pred_num = torch.argmax(pred_prob,1)
        accracy = np.sum((pred_num==label_prob).cpu().numpy())
        total_accuracy += accracy

        up = (label_prob - label_prob%5)
        down = up+5
        accuracy5 = np.sum((((pred_num-up)>=0)&(((pred_num-down) <0))).cpu().numpy())
        total_accuracy_5 += accuracy5

    return total_loss_prob,total_loss_dist,total_loss,total_accuracy,total_accuracy_5


def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, total_loss_prob,total_loss_dist = 0, 0, 0
    total_accuracy = 0
    total_accuracy_5 = 0

    desc = '  - (Validation) '
    Loss_func = nn.CrossEntropyLoss()
    Loss_func_dist = nn.MSELoss()

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = batch[0].float().to(device)
            trg_seq = batch[1].float().to(device)
            label_shift = batch[2].float().to(device)
            label_prob = batch[3].to(device)

            # forward
            pred_shift,pred_prob = model(src_seq, trg_seq)            
            loss_prob = Loss_func(pred_prob,label_prob)
            loss_dist = Loss_func_dist(pred_shift,label_shift)
            loss = loss_prob+loss_dist


            total_loss_prob += loss_prob.item()*pred_shift.shape[0]
            total_loss_dist += loss_dist.item()*pred_shift.shape[0]
            total_loss += loss.item()*pred_shift.shape[0]
            # calculate accuracy
            pred_num = torch.argmax(pred_prob,1)
            accracy = np.sum((pred_num==label_prob).cpu().numpy())
            total_accuracy += accracy

            up = (label_prob - label_prob%5)
            down = up+5
            accuracy5 = np.sum((((pred_num-up)>=0)&(((pred_num-down) <0))).cpu().numpy())
            total_accuracy_5 += accuracy5

    return total_loss_prob,total_loss_dist,total_loss,total_accuracy,total_accuracy_5

def  train(model, training_data,traindata_len, validation_data,valdata_len, optimizer, device, opt):
    ''' Start training '''

    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    if opt.use_tb:
        print("[Info] Use Tensorboard")

        now = int(round(time.time()*1000))
        nowname = time.strftime('%Y%m%d_%H_%M_%S',time.localtime(now/1000))
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard/'+nowname))

        log_param = os.path.join(opt.output_dir, nowname+'.log')
        thisfold = open(log_param,'a')
        thisfold.write('n_layer:{}\n'.format(opt.n_layers))
        thisfold.write('n_head:{}\n'.format(opt.n_head))
        thisfold.write('d_k/v:{}\n'.format(opt.d_k))
        thisfold.write('ffn_inner_d:{}\n'.format(opt.d_inner_hid))
        thisfold.write('warmup:{}\n'.format(opt.n_warmup_steps))
        thisfold.write('batchsize:{}\n'.format(opt.batch_size))
        thisfold.close()


    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,accuracy\n')
        log_vf.write('epoch,loss,accuracy\n')

    def print_performances(header, loss,accu, start_time, lr):
        print('  - {header:12} loss: {loss:3.4f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", loss=loss,
                  accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))

    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss_prob,train_loss_dist,train_loss,train_accuracy,train_accuracy_5 = train_epoch(
            model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)

        train_loss /= traindata_len
        train_loss_prob /= traindata_len
        train_loss_dist /= traindata_len

        train_accuracy_5 /= traindata_len
        train_acc = train_accuracy/traindata_len
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        # print('Loss:{}'.format(train_loss))
        print_performances('Training', train_loss, train_acc, start, lr)

        start = time.time()
        valid_loss_prob,valid_loss_dist,valid_loss,valid_accuracy,valid_accuracy_5 = eval_epoch(model, validation_data, device, opt)
        valid_loss_prob /= valdata_len
        valid_loss_dist /= valdata_len
        valid_loss /= valdata_len
        valid_accuracy_5 /= valdata_len
        valid_acc = valid_accuracy/valdata_len
        print_performances('Validation', valid_loss, valid_acc, start, lr)
        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*0)
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            model_name = nowname+'.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                accu=100*train_acc))
            log_vf.write('{epoch},{loss: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                accu=100*valid_acc))

        if opt.use_tb:
            tb_writer.add_scalar('loss/train',train_loss, epoch_i)
            tb_writer.add_scalar('loss/val',valid_loss, epoch_i)
            tb_writer.add_scalar('loss/train_prob',train_loss_prob, epoch_i)
            tb_writer.add_scalar('loss/val_prob',valid_loss_prob, epoch_i)
            tb_writer.add_scalar('loss/train_dist',train_loss_dist, epoch_i)
            tb_writer.add_scalar('loss/val_dist',valid_loss_dist, epoch_i)

            tb_writer.add_scalar('accuracy/train',train_acc, epoch_i)
            tb_writer.add_scalar('accuracy/val',valid_acc, epoch_i)
            tb_writer.add_scalar('accuracy/train_5',train_accuracy_5, epoch_i)
            tb_writer.add_scalar('accuracy/val_5',valid_accuracy_5, epoch_i)

            tb_writer.add_scalar('learning_rate', lr, epoch_i)


def main1(n_layer_, n_head_, d_kv_, d_modle_, ffn_, warmup_, batch_,train_path,val_path,output_path,past,cand,near):
    '''
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-train_path', default=train_path)   
    parser.add_argument('-val_path', default=val_path)     
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-b', '--batch_size', type=int, default=batch_)
    parser.add_argument('-d_model', type=int, default=d_modle_) 
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
    parser.add_argument('-output_dir', type=str, default=output_path)
    parser.add_argument('-use_tb', default=True)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', default = False)
    parser.add_argument('-label_smoothing', default = True)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # https://pytorch.org/docs/stable/notes/randomness.html
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

    #========= Loading Dataset =========#
    print('==>init dataLoader')
    batch_size = opt.batch_size
    ins_loader_train,traindata = func_getdataloader(txtfile=opt.train_path, batch_size=batch_size, shuffle=True, num_workers=16)
    ins_loader_val,valdata = func_getdataloader(txtfile=opt.val_path, batch_size=batch_size, shuffle=True, num_workers=16)

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


def find_near(pdcontent,x,y):

    pdcontent = pdcontent.drop_duplicates(subset=['pos_x','pos_y'])
    all_posi = pdcontent.values.tolist()
    dis_all_posi = []
    for thisframepos in all_posi:
        dis = (thisframepos[0]-x)**2 +(thisframepos[1]-y)**2
        dis_all_posi.append(thisframepos+[dis])
    dis_all_posi_np = np.array(dis_all_posi)
    a_arg = np.argsort(dis_all_posi_np[:,-1]) 
    sortnp = dis_all_posi_np[a_arg.tolist()]

    return sortnp 


def load_model(g_opt, device):

    checkpoint = torch.load(g_opt['model']) 
    opt = checkpoint['settings']
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
        dropout=opt.dropout).to(device)
    transformer.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return transformer 



def main2(input_detxml,output_trackcsv,model_path,fract,Past,Cand,Near):

    print('===>>>START {}'.format(time.strftime('%Y%m%d_%H_%M_%S',time.localtime(int(round(time.time()*1000))/1000))))

    opt = {}
    opt['model'] = model_path
    # device = 'cpu'
    device = 'cuda'
    transformer = load_model(opt, device)
    transformer.eval()

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

    established_track = pd.DataFrame(columns=['trackid','pos_x','pos_y','frame']) 
    keep_track = pd.DataFrame(columns=['trackid','pos_x','pos_y','frame'])

    print('===>>>Finish prepare total det')
    this_frame = start_frame
    while(this_frame<end_frame):
        print('===>>>Process Frame {}-{}'.format(this_frame,this_frame+1))
        this_det = detection_total[detection_total['frame'] == this_frame]
        next_det = detection_total[detection_total['frame']==this_frame+1]

        if this_frame == start_frame:
            established_track = this_det[['det_id','pos_x','pos_y','frame']]
            established_track = established_track.rename(columns={'det_id':'trackid'})
            temp = np.zeros([len(this_det),2])
            temp[:,0] = this_det['det_id']
            established_track_HOLDnum = pd.DataFrame(temp)
            established_track_HOLDnum.columns = ['trackid','HOLDnum']
        t_trackid = set(established_track['trackid'])


        n_near = Near
        
        one_frame_match_list = []
        for one_trackid in t_trackid:
            thistrack_dic = {}
            one_track = established_track[established_track['trackid']==one_trackid]

            one_track_length = len(one_track)
            one_track_list = one_track.values.tolist()
            p_ind = one_track_list[0][0] 
            padding_before = []
            if one_track_length < Past:
                for a in range(Past-one_track_length):
                    padding_before.append([p_ind,-1,-1,-1,-1])
            convert_one_track = []
            for b in one_track_list:
                convert_one_track.append(b+[0])
            pad_paticle_poslist = padding_before+convert_one_track

            if convert_one_track[-1][3]<end_frame:
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
                        notequalGT = 0

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
                            if numb == n_near-neednull:
                                break
                        if numb < n_near-neednull:
                            neednull = n_near-numb

                        for _i in range(neednull):
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
                
                if convert_one_track[-1][3]<end_frame-(cand-1):
                    str_candlist = [str(_) for _ in all_candidate]
                    assert len(str_candlist) == len(set(str_candlist)),print('生成重复候选！！{}'.format(str_candlist))

            thistrack_dic['trackid'] = one_trackid
            thistrack_dic['cand5_id'] = det_id_4cand_reserve
            thistrack_dic['pastpos'] = pastposlist
            thistrack_dic['cand25'] = all_candidate
            
            one_frame_match_list.append(thistrack_dic)

        print('===>>>Finish construct samples')


        this_frame_dataloader,this_frame_data = func_getdataloader_match(one_frame_match_list,batch_size=len(one_frame_match_list), shuffle=False, num_workers=1)
 
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

        
        pred_shift_next1 = pred_shift[:,0,:-1].detach().cpu().numpy()*this_frame_data.std +this_frame_data.mean
        pred_shift_exist_flag = pred_shift[:,0,-1].detach().cpu().numpy().reshape(-1,1)
        pred_shift_id_np = np.concatenate([np.array(trackid_batch).reshape(-1,1),pred_shift_next1,pred_shift_exist_flag],-1)
        pred_shift_id_pd = pd.DataFrame(pred_shift_id_np)
        pred_shift_id_pd.columns = ['trackid','shift_x','shift_y','exist_flag']
        print('===>>>Finish Predicting probability')


        costlist = []
        for it in range(len(one_frame_match_list)):
            for m in range(Near):
                costlist.append([soft_pred_prob5[it][m], trackid_batch[it], cand5id_batch[it][m]])


        print('===>>>start construct model')
        costs = costlist.copy()
        # Create a new model
        model = grb.Model("mip1")
        model.Params.outputflag = 0

        ro = [] 
        for i in range(len(costs)):
            ro.append(model.addVar(0.0, 1.0, 0.0, grb.GRB.BINARY,'z_' + str(costs[i][1]) + '_' + str(costs[i][2]))) 
        model.update()

        expr = grb.LinExpr()
        for i in range(len(ro)):
            expr.add(ro[i], costs[i][0]) 
        model.setObjective(expr, grb.GRB.MAXIMIZE) 

        nrConstraint = 0
        exprcs = []
        for j in trackid_batch: 
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

        for j in list(next_det['det_id']): 
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
        
        print('===>>>Finish construct model {}')
        model.optimize()
        print('===>>>Finish optimize {}')
        assert model.status == grb.GRB.Status.OPTIMAL

        solutionIndex = []
        solutions = []
        for i in range(len(ro)):
            if ro[i].Xn > 0.5:
                solutionIndex.append(i)
                solutions.append(ro[i].VarName)
        


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
                    established_track.reset_index(drop=True,inplace=True)
                    established_track = established_track.drop(established_track[established_track['trackid']==link_track_id].index)
        
        for to_belinkid in list(next_det['det_id']): 
            if to_belinkid not in linked_det_id:
                ext = next_det[next_det['det_id'] == to_belinkid]
                ext = ext[['det_id','pos_x','pos_y','frame']]
                ext = ext.rename(columns={'det_id':'trackid'})
                established_track = established_track.append(ext)

                temp_dic = {'trackid':[ext.iloc[0,0]],'HOLDnum':[0]}  
                temp_pd = pd.DataFrame(temp_dic)
                established_track_HOLDnum = established_track_HOLDnum.append(temp_pd)


        print('===>>>Finish process!!! {}'.format(time.strftime('%Y%m%d_%H_%M_%S',time.localtime(int(round(time.time()*1000))/1000))))

        this_frame += 1


    keep_track = keep_track.append(established_track)
    keep_track.to_csv(output_trackcsv)




if __name__ == '__main__':

    # =================train=========================
    trainfilename = 'MICROTUBULE snr 1247 density low'
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
    main1(n_layer_, n_head_, d_kv_, d_model_, ffn_, warmup_, batch_,traindata_path,valdata_path,outputmodel_path,past,cand,near)

    
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
            keep_track = main2(
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