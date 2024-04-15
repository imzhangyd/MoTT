"""
This script handles the training, tracking and evaluation.
"""

import glob
import argparse
import os
import pandas as pd
import time
import subprocess
from engine.trainval import trainval
from engine.inference import tracking
from utils import resultcsv_2xml
import shutil

__author__ = "Yudong Zhang"


def save_args_to_file(args, path):
    with open(path, "a+") as file:
        for arg, value in vars(args).items():
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            file.write(f"{arg}: {value}\n")
        file.write("--------------------------\n")


def parse_args_():
    parser = argparse.ArgumentParser()

    # train params
    # data params
    parser.add_argument("--trainfilename", type=str, default="tracks10")
    parser.add_argument(
        "--train_path",
        default="/mnt/data1/ZYDdata/code/MoTT_particle_notsame/MoTT/dataset/tracks10/trainval_data/add_size/past7_depth2_near5/debugdata.txt",
    )
    parser.add_argument(
        "--val_path",
        default="/mnt/data1/ZYDdata/code/MoTT_particle_notsame/MoTT/dataset/tracks10/trainval_data/add_size/past7_depth2_near5/debugdata.txt",
    )
    parser.add_argument("--traindatamean", nargs="+", type=int, default=None)
    parser.add_argument("--traindatastd", nargs="+", type=int, default=None)
    parser.add_argument("--len_established", type=int, default=7)
    parser.add_argument("--len_future", type=int, default=2)
    parser.add_argument("--near", type=int, default=5)

    # network params
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--d_k", type=int, default=96)
    parser.add_argument("--d_v", type=int, default=96)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=96 * 6)
    parser.add_argument("--d_inner_hid", type=int, default=96 * 6 * 2)
    parser.add_argument("--n_position", type=int, default=5000)
    parser.add_argument("--inoutdim", type=int, default=12)
    # s_x, s_y, s_size, s_inten,    x, y, size, inten,   abs shiftx, abs shift y, abs dist,flag(0 or 1)

    # training params

    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_warmup_steps", type=int, default=1000)
    parser.add_argument("--lr_mul", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", default=True)

    # output and record
    # parser.add_argument('--output_dir', type=str, default=outputmodel_path)
    parser.add_argument("--use_tb", default=False, action="store_true")
    parser.add_argument(
        "--save_mode", type=str, choices=["all", "best"], default="best"
    )
    parser.add_argument("--ckpt_save_root", type=str, default="./checkpoint")

    # device
    parser.add_argument("--no_cuda", default=False, action="store_true")

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":

    opt = parse_args_()

    trainfilename = opt.trainfilename

    # assert trainfilename in opt.train_path and trainfilename in opt.val_path

    # data param
    past = opt.len_established
    cand = opt.len_future
    near = opt.near
    opt.num_cand = near**cand

    # ==================train====================
    now = int(round(time.time() * 1000))
    nowname = time.strftime("%Y%m%d_%H_%M_%S", time.localtime(now / 1000))
    expname = nowname + "_" + trainfilename.replace(" ", "_") + "_ckpt"

    # make save ckpt folder
    outputmodel_path = os.path.join(opt.ckpt_save_root, expname)
    if not os.path.exists(outputmodel_path):
        os.makedirs(outputmodel_path)
    opt.output_dir = outputmodel_path
    # save params
    save_args_to_file(opt, os.path.join(outputmodel_path, "param.txt"))
    pyfilepath = os.path.abspath(__file__)
    shutil.copy(
        pyfilepath, os.path.join(outputmodel_path, os.path.split(pyfilepath)[-1])
    )
    shutil.copy(
        os.path.join(os.path.split(pyfilepath)[0], "engine/trainval.py"),
        os.path.join(outputmodel_path, "trainval.py"),
    )
    shutil.copytree(
        os.path.join(os.path.split(pyfilepath)[0], "transformer"),
        os.path.join(outputmodel_path, "transformer"),
    )
    # train
    traindatamean, traindatastd = trainval(opt)

    with open(os.path.join(outputmodel_path, "param.txt"), "a") as f:
        f.write(str(traindatamean))
        f.write(str(traindatastd))
    f.close()
