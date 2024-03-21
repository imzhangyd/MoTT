import cv2
import numpy as np
import os


__author__ = "Yudong Zhang"


def vis_netpred(input_detfile,
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
         imagefolder=None):
    assert imagefolder
    thisframeimage = cv2.imread(os.path.join(
        imagefolder,
        os.path.split(input_detfile)[-1].split('.')[0],
        'img1/{:06d}.jpg'.format(int(this_frame))))
    for indd,tkid in enumerate(trackid_batch):
        onetracklet = src_seq_abpos[indd]
        predthisdetid = pred_nextone_detid[indd]
        realtraj = onetracklet[onetracklet[:,-1] != -1][:,:2] \
            .reshape(-1,2).unsqueeze(0).numpy()

        thisfrmtkid_image = np.zeros(
            [thisframeimage.shape[0]+200,thisframeimage.shape[1],thisframeimage.shape[2]])
        thisfrmtkid_image[:-200,:,:] = thisframeimage
        
        

        # draw the candidate position and matching possibility
        nextindhere = range(0,trg_seq.shape[1],Near**(Cand-1))
        nextindhere = list(nextindhere)
        # tm = range(trg_seq.shape[0])
        # tm = list(tm)
        nextoneshifts = trg_seq[indd][nextindhere][:,0].detach().cpu().numpy()
        nullflag = np.sum(nextoneshifts == 0,1)<3
        nextoneabsshifts = nextoneshifts[:,:2]*this_frame_data.std[:2] +this_frame_data.mean[:2]
        nextonepos = lastab[indd].reshape(-1,2) + nextoneabsshifts

        for jh in range(len(nextonepos)):
            if predthisdetid == cand5id_batch[indd][jh]:
                colortuple = (255,0,255) # write and circle the candidate with highest posssibility in purple
            else:
                colortuple = (0,255,0) # write and circle other candidate in green
            if nullflag[jh]:
                thisfrmtkid_image = cv2.polylines(
                    thisfrmtkid_image,
                    np.concatenate(
                        [lastab[indd,:][np.newaxis,:],nextonepos[jh,:][np.newaxis,:]],0)[np.newaxis,:].astype(int), 
                    False, colortuple, 1
                )
                thisfrmtkid_image = cv2.putText(
                    img       = thisfrmtkid_image, 
                    text      = str(cand5id_batch[indd][jh]), 
                    org       = (int(nextonepos[jh,0]),int(nextonepos[jh,1])), 
                    fontFace  = cv2.FONT_HERSHEY_PLAIN,  
                    fontScale = 1,
                    color     = colortuple,
                    thickness = 1)

                thisfrmtkid_image = cv2.putText(
                    img       = thisfrmtkid_image, 
                    text      = '[Detection]'+str(cand5id_batch[indd][jh])+'_{:.1f}'.format(pred_prob5[indd][jh]), 
                    org       = (50,thisframeimage.shape[0]+40+jh*40), 
                    fontFace  = cv2.FONT_HERSHEY_PLAIN,  
                    fontScale = 2,
                    color     = colortuple,
                    thickness = 2)

                x_ = int(nextonepos[jh,0])
                y_ = int(nextonepos[jh,1])
                pt = (x_,y_)
                thisfrmtkid_image = cv2.circle(
                    thisfrmtkid_image, pt, 2, colortuple, 2)
            else:
                thisfrmtkid_image = cv2.putText(
                    img       = thisfrmtkid_image, 
                    text      = '[Detection]'+str(cand5id_batch[indd][jh])+'_{:.1f}'.format(pred_prob5[indd][jh]), 
                    org       = (50,thisframeimage.shape[0]+40+jh*40), 
                    fontFace  = cv2.FONT_HERSHEY_PLAIN,  
                    fontScale = 2,
                    color     = colortuple,
                    thickness = 2)

        # draw the history of single tracklet with white color
        thisfrmtkid_image = cv2.polylines(
            thisfrmtkid_image,
            realtraj.astype(int), False, (255,255,255), 1)
        for i in range(len(realtraj[0])):
            x_ = int(realtraj[0,i,0])
            y_= int(realtraj[0,i,1])
            pt = (x_,y_)
            thisfrmtkid_image = cv2.circle(
                thisfrmtkid_image, pt, 1, (255,255,255), 1)

        cv2.imwrite(
            os.path.join(output_trackcsv.replace('_link.csv',''),"frame{:06d}_track{:06d}_.jpg".format(int(this_frame), tkid)),
            thisfrmtkid_image
        )
    return thisframeimage



def vis_matchres(this_frame, link_track_id, output_trackcsv, link_cand_id, p_detid, thisframeimage,next_det,
                 use_guribimatchdet, dif_cls_gurobi_changeto_1, dist_changeto_1, use_fillup, stop_4_moviestop,
                 stop_4_outview, stop_4_predstop, stop_4_overhold,temp_dic):
    
    gurobiimage = cv2.imread(os.path.join(
        output_trackcsv.replace('_link.csv',''), 
        "frame{:06d}_track{:06d}_.jpg".format(int(this_frame),link_track_id)))
    if link_cand_id == p_detid:
        colortuple = (255,0,255) # If the gurobi match is same as classification, write gurobi match result in purple
    else:
        colortuple = (10,215,255) # if different, write gurobi match result in golden, and circle the gurobi match result.
    gurobiimage = cv2.putText(
                img       = gurobiimage, 
                text      = '[Gurobi]'+str(link_cand_id), 
                org       = (450,thisframeimage.shape[0]+40), 
                fontFace  = cv2.FONT_HERSHEY_PLAIN,  
                fontScale = 2,
                color     = colortuple,
                thickness = 2)
    if link_cand_id != p_detid and link_cand_id > 0:
        pt = (int(next_det[next_det['det_id'] == link_cand_id].iloc[0,0]), int(next_det[next_det['det_id'] == link_cand_id].iloc[0,1]))
        gurobiimage = cv2.circle(
            gurobiimage, pt, 5, colortuple, 2)

    # cv2.imwrite(
    #     output_trackcsv.replace('_link.csv','/')+"frame{:06d}_track{:06d}_gurobi.jpg".format(int(this_frame), link_track_id),
    #     gurobiimage
    # )
    # cv2.imwrite(
    #     output_trackcsv.replace('_link.csv','/')+"track{:06d}_frame{:06d}_gurobi.jpg".format(link_track_id, int(this_frame) ),
    #     gurobiimage
    # )
        
    conclusion = []
    if use_guribimatchdet:
        conclusion.append('Use gurobi matching result')
        if not dif_cls_gurobi_changeto_1:
            conclusion.append('Same as cls pred')
        else:
            conclusion.append('Different from cls pred')
    if dist_changeto_1:
        conclusion.append('Gurobi det is too far, not link it')
    
    if use_fillup:
        conclusion.append('Fill up a prediction pos')
    if stop_4_moviestop:
        conclusion.append('Stop for movie end')
    if stop_4_outview:
        conclusion.append('Stop for out of image')
    if stop_4_predstop:
        conclusion.append('Stop for cls of strong stop')
    if stop_4_overhold:
        conclusion.append('Stop for hold long without re-link')

    if not use_fillup:
        colortuple = (255,255,255) # if not fill, write conclusion in white
    else:
        colortuple = (255,255,0) # if fill, write conclusion in bright blue, and circle the position of fill up.

    for m in range(len(conclusion)):
        gurobiimage = cv2.putText(
            img       = gurobiimage, 
            text      = conclusion[m], 
            org       = (700,thisframeimage.shape[0]+50+m*50), 
            fontFace  = cv2.FONT_HERSHEY_PLAIN,  
            fontScale = 3,
            color     = colortuple,
            thickness = 2)

    if use_fillup:
        assert temp_dic
        pt = (int(temp_dic['pos_x'][0]),int(temp_dic['pos_y'][0]))
        gurobiimage = cv2.circle(
            gurobiimage, pt, 5, colortuple, 2)

    cv2.imwrite(os.path.join(
        output_trackcsv.replace('_link.csv',''),
        "frame{:06d}_track{:06d}_mangement.jpg".format(int(this_frame), link_track_id)),
        gurobiimage
    )
    cv2.imwrite(os.path.join(
        output_trackcsv.replace('_link.csv',''),
        "track{:06d}_frame{:06d}_mangement.jpg".format(link_track_id, int(this_frame))),
        gurobiimage
    )