from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
# import math
# from dataset.dataprocess import func_normlize

def normlization(tensor,mean,std):
    tensor_ = tensor
    for num,line in enumerate(tensor):
        if line[-1] == 1:
            tensor_[num,:-1] = (line[:-1]-mean)/std
        elif line[-1] == 0:
            tensor_[num,:] = 0
    return tensor_


class cls_Dataset_oridif_match(Dataset):
    def __init__(self,one_frame_match_list,mean_=None,std_=None):
        super(cls_Dataset_oridif_match,self).__init__()


        datapathlist = []
        datapathlist = []
        for line in one_frame_match_list:
            passed = line['pastpos']
            t_future = line['cand25']
            trackid = line['trackid']
            cand5_id = line['cand5_id']


            # add bbox height and width
            passed_np = np.array(passed)
            aug_passednp = -np.ones([passed_np.shape[0],passed_np.shape[1]+2])
            aug_passednp[:,-1]  = passed_np[:,-1]
            aug_passednp[:,:-3] = passed_np[:,:-1]
            fline = np.where(aug_passednp[:,-1] > -0.5)
            aug_passednp[fline,-3] = aug_passednp[fline,3] - aug_passednp[fline,2]
            aug_passednp[fline,-2] = aug_passednp[fline,5] - aug_passednp[fline,4]
            passed = aug_passednp.tolist()

            future_np = np.array(t_future)
            futureshape = future_np.shape
            future_np = future_np.reshape(-1,futureshape[-1])
            aug_futurenp = -np.ones([future_np.shape[0],future_np.shape[1]+2])
            aug_futurenp[:,-1]  = future_np[:,-1]
            aug_futurenp[:,:-3] = future_np[:,:-1]
            fline = np.where(aug_futurenp[:,-1] > -0.5)
            aug_futurenp[fline,-3] = aug_futurenp[fline,3] - aug_futurenp[fline,2]
            aug_futurenp[fline,-2] = aug_futurenp[fline,5] - aug_futurenp[fline,4]
            future_np = aug_futurenp.reshape([futureshape[0],futureshape[1],-1])
            t_future = future_np.tolist()

            # shift of passed
            passed_shift = aug_passednp[1:,:-1] - aug_passednp[:-1,:-1]
            start_ = 0
            if -1 in set(aug_passednp[:,-1]):
                start_ = np.where(aug_passednp[:,-1] == -1)[0][-1]+1
                passed_shift[start_-1,:] =0
            flag_list = [0]*(start_) + [1]*(len(passed_shift)-(start_))
            flag_np = np.array(flag_list).reshape(-1,1)
            augpassed_pre = aug_passednp[1:,:].copy()
            augpassed_pre[np.where(augpassed_pre[:,-1] < -0.5),:] = 0
            passed_shift = np.concatenate([passed_shift,augpassed_pre[:,:-1],flag_np],-1)
            
                
            # shift of future
            future_shift = []
            for kk in t_future:
                thisfu = np.array(kk)
                thisfu[np.where(thisfu[:,-1] < -0.5),:] = 0

                temp = np.array([passed[-1]]+kk)
                this_shift = np.zeros([len(kk),len(kk[0])-1])
                where_exist = np.where(temp[:,-1] == 0)
                # for null average shift multi frames
                gap_fra = (where_exist[0][1:] - where_exist[0][:-1]).reshape(-1,1)
                gap_frames = np.concatenate([gap_fra]*(temp.shape[1]-1),1)
                this_shift[where_exist[0][1:]-1] = \
                    (temp[where_exist[0][1:],:-1]-temp[where_exist[0][:-1],:-1]) \
                        / gap_frames
                flag_here = np.array(kk)[:,-1].reshape(-1,1) + 1
                this_sft = np.concatenate([this_shift,thisfu[:,:-1],flag_here],-1)
                future_shift.append(this_sft.tolist())

            future_shift_np = np.array(future_shift)

            # framenum = eval(line.split('s')[-1])
            # gold_shift = future_shift[score]

            datapathlist.append([passed_shift,future_shift_np,trackid,cand5_id,passed])
                   

        if mean_ is None:
            reshapelen = len(datapathlist[0][0][0])
            t_passed_ = np.stack(np.array(datapathlist)[:,0]).reshape([-1,reshapelen])
            t_passed_1 = t_passed_[t_passed_[:,-1] == 1]
            t_future_ = np.stack(np.array(datapathlist)[:,1]).reshape([-1,reshapelen])
            t_future_1 = t_future_[t_future_[:,-1] == 1]
            t_shift = np.concatenate([t_passed_1,t_future_1],0)

            t_mean = t_shift.astype(np.float32).mean(0)
            t_std = t_shift.astype(np.float32).std(0)


            self.mean = np.array(t_mean[:-1])
            self.std  = np.array(t_std[:-1])

        else:
            self.mean = np.array(mean_)
            self.std = np.array(std_)

        self.datapathlist = datapathlist
        


    def __getitem__(self, index: int):
        # get path
        passed_shift_ = np.array(self.datapathlist[index][0])
        future_shift_ = self.datapathlist[index][1]
        trackid_ = np.array(self.datapathlist[index][2])
        cand5_id_ = np.array(self.datapathlist[index][3])

        passed_ = np.array(self.datapathlist[index][-1])

        # numpy->torch
        
        passed_shift_t = torch.from_numpy(passed_shift_)
        future_shift_t = torch.from_numpy(future_shift_)

        passed_t = torch.from_numpy(passed_)
        
        # normalization
        passed_shift_t_norm = normlization(passed_shift_t,torch.from_numpy(self.mean),torch.from_numpy(self.std))
        future_shift_t_norm = future_shift_t
        for num,fu in enumerate(future_shift_t):
            fu_norm = normlization(fu,torch.from_numpy(self.mean),torch.from_numpy(self.std))
            future_shift_t_norm[num] = fu_norm
        
        ip_lb = (passed_shift_t_norm,future_shift_t_norm,trackid_,cand5_id_,passed_t)
        return (ip_lb)

    def __len__(self):
        return len(self.datapathlist)

    



def func_getdataloader_oridif_match(txtfile, batch_size, shuffle, num_workers,mean=None,std=None):
    dtst_ins = cls_Dataset_oridif_match(txtfile,mean,std)
    loads_ins = DataLoader(dataset = dtst_ins, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return loads_ins,dtst_ins