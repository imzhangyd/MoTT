from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader


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
    def __init__(self,txtfile):
        super(cls_Dataset,self).__init__()

        with open(txtfile) as t_file:
            lines = t_file.readlines()
        datapathlist = []
        num_cand = lines[0].count('s')-2
        for line in lines:
            line = line.strip('\n')
            line = line.rstrip()
            passed = eval(line.split('s')[0])

            t_future = []
            this_cand = line.count('s')-2
            if this_cand == num_cand:
                for m in range(num_cand):
                    t_future.append(eval(line.split('s')[m+1]))
            else:
                for m in range(this_cand):
                    t_future.append(eval(line.split('s')[m+1]))
                for m in range(int(num_cand-this_cand)):
                    t_future.append(
                        [[-1]*len(passed[0]) for _ in range(len(t_future[0]))]
                        )


            passed_np = np.array(passed)
            passed_shift = passed_np[1:,:-1] - passed_np[:-1,:-1]
            start_ = 0
            if -1 in set(passed_np[:,-1]):
                start_ = np.where(passed_np[:,-1] == -1)[0][-1]+1
                passed_shift[start_-1,:] =0
            flag_list = [0]*(start_) + [1]*(len(passed_shift)-(start_))
            flag_np = np.array(flag_list).reshape(-1,1)
            passed_shift = np.concatenate([passed_shift,flag_np],-1)
            score = int(eval(line.split('s')[-2]))
            

            
            future_shift = []
            for kk in t_future:
                temp = np.array([passed[-1]]+kk)
                this_shift = np.zeros([len(kk),len(kk[0])-1])
                where_exist = np.where(temp[:,-1] == 0)
                this_shift[where_exist[0][1:]-1] = temp[where_exist[0][1:],:-1]-temp[where_exist[0][:-1],:-1]
                flag_here = np.array(kk)[:,-1].reshape(-1,1) + 1
                this_sft = np.concatenate([this_shift,flag_here],-1)
                future_shift.append(this_sft.tolist())

            future_shift_np = np.array(future_shift)
            framenum = eval(line.split('s')[-1])
            gold_shift = future_shift[score]
            datapathlist.append([passed_shift,future_shift_np,gold_shift,score,framenum,passed])


        t_passed_ = np.stack(np.array(datapathlist)[:,0]).reshape([-1,3])
        t_passed_1 = t_passed_[t_passed_[:,-1] == 1]
        t_future_ = np.stack(np.array(datapathlist)[:,1]).reshape([-1,3])
        t_future_1 = t_future_[t_future_[:,-1] == 1]
        t_shift = np.concatenate([t_passed_1,t_future_1],0)

        
        t_mean = t_shift.mean(0)
        t_std = t_shift.std(0)

        self.mean = t_mean[:-1]
        self.std = t_std[:-1]


        self.datapathlist = datapathlist


    def __getitem__(self, index: int):
        # get path
        passed_shift_ = np.array(self.datapathlist[index][0])
        future_shift_ = self.datapathlist[index][1]
        gold_shift_ = np.array(self.datapathlist[index][2])
        score_ = np.array(self.datapathlist[index][3])

        framenum_ = np.array(self.datapathlist[index][-2])
        passed_ = np.array(self.datapathlist[index][-1])

        # numpy->torch
        
        passed_shift_t = torch.from_numpy(passed_shift_)
        future_shift_t = torch.from_numpy(future_shift_)
        gold_shift_t = torch.from_numpy(gold_shift_)
        score_t = torch.from_numpy(score_).long()

        framenum_t = torch.from_numpy(framenum_)
        passed_t = torch.from_numpy(passed_)

        # normalization
        passed_shift_t_norm = normlization(passed_shift_t,torch.from_numpy(self.mean),torch.from_numpy(self.std))
        future_shift_t_norm = future_shift_t
        for num,fu in enumerate(future_shift_t):
            fu_norm = normlization(fu,torch.from_numpy(self.mean),torch.from_numpy(self.std))
            future_shift_t_norm[num] = fu_norm
        gold_shift_t_norm = normlization(gold_shift_t,torch.from_numpy(self.mean),torch.from_numpy(self.std))
        # cumsum
        gold_sft_flag = gold_shift_t_norm[:,-1:]
        gold_sft_cs = gold_shift_t_norm[:,:-1].cumsum(dim=-2)
        gold_final = torch.cat([gold_sft_cs,gold_sft_flag],-1)

        ip_lb = (passed_shift_t_norm,future_shift_t_norm,gold_final,score_t,framenum_t,passed_t)
        return (ip_lb)

    def __len__(self):
        return len(self.datapathlist)

    

def func_getdataloader(txtfile, batch_size, shuffle, num_workers):
    dtst_ins = cls_Dataset(txtfile)
    loads_ins = DataLoader(dataset = dtst_ins, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return loads_ins,dtst_ins