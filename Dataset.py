# from cv2 import norm
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
import math
import ipdb


__author__ = "Yudong Zhang"

def normlization(tensor,mean,std):
    tensor_ = tensor
    for num,line in enumerate(tensor):
        if line[-1] == 1:
            tensor_[num,:-1] = (line[:-1]-mean)/std
        elif line[-1] == 0:
            tensor_[num,:] = 0
    return tensor_


class cls_Dataset(Dataset):
    def __init__(self,txtfile,opt,datamean=None,datastd=None):
        super(cls_Dataset,self).__init__()

        with open(txtfile) as t_file:
            lines = t_file.readlines()
        datapathlist = []
        set_candnum = opt.near**opt.len_future
        num_cand = lines[0].count('s')-2

        for line in lines:
            try:
                line = line.strip('\n')
                line = line.rstrip()
                passed = eval(line.split('s')[0])

                t_future = []
                if set_candnum == num_cand: # single tracklet
                    this_cand = line.count('s')-2
                    if this_cand == num_cand:
                        for m in range(num_cand):
                            t_future.append(eval(line.split('s')[m+1]))
                    else:
                        for m in range(this_cand):
                            t_future.append(eval(line.split('s')[m+1]))
                        for m in range(int(num_cand-this_cand)):
                            t_future.append(
                                [[-1]*len(passed[0]) for _ in range(opt.len_future)]
                                )
                else: # a group of tracklets with one same next one
                    assert num_cand == opt.near
                    for m in range(num_cand):
                        thisfu = eval(line.split('s')[m+1])
                        this_cand = len(thisfu)
                        if this_cand == num_cand:
                            for n in range(num_cand):
                                t_future.append(thisfu[n])
                        else:
                            for n in range(this_cand):
                                t_future.append(thisfu[n])
                            for n in range(int(num_cand-this_cand)):
                                t_future.append(
                                    [[-1]*len(passed[0]) for _ in range(opt.len_future)]
                                    )


                passed_np = np.array(passed) # x y size intensity flag(-1 or 0)

                # shift of passed
                passed_shift = passed_np[1:,:-1] - passed_np[:-1,:-1] # s_x, s_y, s_size, s_inten
                start_ = 0
                if -1 in set(passed_np[:,-1]):
                    start_ = np.where(passed_np[:,-1] == -1)[0][-1]+1
                    passed_shift[:start_,:] = 0 # not real shift
                # flag
                flag_list = [0]*(start_) + [1]*(len(passed_shift)-(start_))
                flag_np = np.array(flag_list).reshape(-1,1)
                # ori pos of passed
                passed_pre = passed_np[1:,:].copy()
                if -1 in set(passed_pre[:,-1]):
                    passed_pre[:start_-1,:] = 0 # not real pos
                
                # add abs shift x， abs shift y， abs dist
                abs_shift = np.abs(passed_shift[:,:2])
                abs_dist = np.sqrt(abs_shift[:,0]**2+abs_shift[:,1]**2).reshape([-1,1])

                # concate shift and ori pos and abs shift dist and flag
                passed_shift = np.concatenate([
                    passed_shift, passed_pre[:,:-1],
                    abs_shift, abs_dist, flag_np],-1) # s_x, s_y, s_size, s_inten,x, y, size, inten, abs shiftx, abs shift y, abs dist,flag(0 or 1)

                score = int(eval(line.split('s')[-2]))
                
                # shift of future
                future_shift = []
                for kk in t_future:
                    # shift future
                    temp = np.array([passed[-1]]+kk)
                    this_shift = np.zeros([len(kk),len(kk[0])-1])
                    where_exist = np.where(temp[:,-1] == 0)
                    # average multi frame for null
                    gap_fra = (where_exist[0][1:] - where_exist[0][:-1]).reshape(-1,1)
                    gap_frames = np.concatenate([gap_fra]*(temp.shape[1]-1),1)
                    this_shift[where_exist[0][1:]-1] = (temp[where_exist[0][1:],:-1]-temp[where_exist[0][:-1],:-1]) / gap_frames
                    # flag
                    flag_here = np.array(kk)[:,-1].reshape(-1,1) + 1
                    # ori pos of passed
                    thisfu = np.array(kk)
                    thisfu[np.where(thisfu[:,-1] < -0.5),:] = 0

                    # abs shift
                    abs_shift = np.abs(this_shift[:,:2])
                    abs_dist = np.sqrt(abs_shift[:,0]**2+abs_shift[:,1]**2).reshape([-1,1])
                    # concatenate
                    this_sft = np.concatenate([
                        this_shift, thisfu[:,:-1],
                        abs_shift, abs_dist, flag_here],-1) ## s_x, s_y, s_size, s_inten,x, y, size, inten, abs shiftx, abs shift y, abs dist,flag(0 or 1)
                    future_shift.append(this_sft.tolist())

                future_shift_np = np.array(future_shift)
                framenum = eval(line.split('s')[-1])
                gold_shift = future_shift[score] # list ：n_length* inoutdim
                # index of right nextone
                scores = np.intersect1d(np.where(future_shift_np[:,0,4] == gold_shift[0][4])[0], np.where(future_shift_np[:,0,5] == gold_shift[0][5])[0])
                datapathlist.append([passed_shift,future_shift_np,gold_shift,score,scores,framenum,passed])
            except TypeError:
                print(line)
        if datamean is None:
            reshapelen = len(datapathlist[0][0][0])
            t_passed_ = np.stack(np.array(datapathlist)[:,0]).reshape([-1,reshapelen])
            t_passed_1 = t_passed_[t_passed_[:,-1] == 1]
            t_future_ = np.stack(np.array(datapathlist)[:,1]).reshape([-1,reshapelen])
            t_future_1 = t_future_[t_future_[:,-1] == 1]
            t_shift = np.concatenate([t_passed_1,t_future_1],0)

            t_mean = t_shift.astype(np.float32).mean(0)
            t_std = t_shift.astype(np.float32).std(0)

            self.mean = t_mean[:-1]
            self.std = t_std[:-1]

        else:
            self.mean = np.array(datamean)
            self.std = np.array(datastd)
        print(self.mean)
        print(self.std)
        self.datapathlist = datapathlist


    def __getitem__(self, index: int):
        # get path
        passed_shift_ = np.array(self.datapathlist[index][0])
        future_shift_ = self.datapathlist[index][1]
        gold_shift_ = np.array(self.datapathlist[index][2])
        score_ = np.array(self.datapathlist[index][3])
        scores_ = np.array(self.datapathlist[index][4])

        framenum_ = np.array(self.datapathlist[index][-2])
        passed_ = np.array(self.datapathlist[index][-1])

        # numpy->torch
        
        passed_shift_t = torch.from_numpy(passed_shift_)
        future_shift_t = torch.from_numpy(future_shift_)
        gold_shift_t = torch.from_numpy(gold_shift_)
        score_t = torch.from_numpy(score_).long()
        scores_t = torch.from_numpy(scores_).long()

        framenum_t = torch.from_numpy(framenum_)
        passed_t = torch.from_numpy(passed_)
        # s_x, s_y, s_size, s_inten,x, y, size, inten, abs shiftx, abs shift y, abs dist,flag(0 or 1)
        # normalization
        passed_shift_t_norm = normlization(passed_shift_t,torch.from_numpy(self.mean),torch.from_numpy(self.std))
        future_shift_t_norm = future_shift_t
        for num,fu in enumerate(future_shift_t_norm):
            fu_norm = normlization(fu,torch.from_numpy(self.mean),torch.from_numpy(self.std))
            future_shift_t_norm[num] = fu_norm
        gold_shift_t_norm = normlization(gold_shift_t,torch.from_numpy(self.mean),torch.from_numpy(self.std))
        # cumsum
        # gold_sft_flag = gold_shift_t_norm[:,-1:]
        # gold_sft_cs = gold_shift_t_norm[:,:-1] #.cumsum(dim=-2)
        # gold_final = torch.cat([gold_sft_cs,gold_sft_flag],-1)
        gold_final = gold_shift_t_norm

        ip_lb = (passed_shift_t_norm,future_shift_t_norm,gold_final,score_t,scores_t,framenum_t,passed_t)
        return (ip_lb)

    def __len__(self):
        return len(self.datapathlist)

    

def func_getdataloader(txtfile, batch_size, shuffle, num_workers, opt, mean=None,std=None):
    dtst_ins = cls_Dataset(txtfile,opt,mean,std)
    loads_ins = DataLoader(dataset = dtst_ins, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return loads_ins,dtst_ins