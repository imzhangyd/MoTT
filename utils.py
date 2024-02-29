import numpy as np
import torch
from transformer.Models import Transformer
import pandas as pd

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


def resultcsv_2xml(xmlfilepath, output_csv_pa, testfilename):

    result_csv = pd.read_csv(output_csv_pa)

    snr = testfilename.split(' ')[2]
    dens = testfilename.split(' ')[-1]
    scenario = testfilename.split(' ')[0]
    method= '_MoTT'
    thrs = 0
    
    t_trackid = list(set(result_csv['trackid']))
    # csv to xml
    with open(xmlfilepath, "w+") as output:
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