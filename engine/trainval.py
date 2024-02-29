from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import time
import os
import torch
import torch.optim as optim
import random

from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim


from Dataset import func_getdataloader


__author__ = "Yudong Zhang"


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


def train(model, training_data, traindata_len, validation_data,valdata_len, optimizer, device, opt):
    ''' Start training '''

    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    if opt.use_tb:
        print("[Info] Use Tensorboard")

        now = int(round(time.time()*1000))
        nowname = time.strftime('%Y%m%d_%H_%M_%S',time.localtime(now/1000))
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard/'+nowname))


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



def trainval(opt):

    opt.d_word_vec = opt.d_model

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
    
    if not opt.no_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('[Info] Initialize dataLoader')
    batch_size = opt.batch_size
    ins_loader_train,traindata = func_getdataloader(txtfile=opt.train_path, batch_size=batch_size, shuffle=True, num_workers=16)
    ins_loader_val,valdata = func_getdataloader(txtfile=opt.val_path, batch_size=batch_size, shuffle=True, num_workers=16)
    print('\tTraining data number:{}'.format(len(traindata)))
    print('\tValidation data number:{}'.format(len(valdata)))

    print('[Info] Initialize transformer')
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

    print('[Info] Initialize optimizer')
    optimizer = ScheduledOptim(
        optim.Adam(transformer_ins.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)
    
    print('[Info] Start train')
    train(transformer_ins, ins_loader_train,len(traindata), ins_loader_val,len(valdata), optimizer, device, opt)