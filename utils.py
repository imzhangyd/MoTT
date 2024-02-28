import numpy as np
import torch
from transformer.Models import Transformer


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