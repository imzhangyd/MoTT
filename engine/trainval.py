from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import time
import os


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


