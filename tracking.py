"""
This script handles the tracking.
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

    # data params
    parser.add_argument("--len_established", type=int, default=7)
    parser.add_argument("--len_future", type=int, default=2)
    parser.add_argument("--near", type=int, default=5)

    parser.add_argument(
        "--traindatamean",
        nargs="+",
        type=int,
        default=[
            -9.1594001e-03,
            -1.2082328e-02,
            9.5779408e-04,
            4.7907796e-02,
            2.5584569e02,
            2.5631042e02,
            1.4279901e01,
            6.6001752e02,
            6.7447734e00,
            6.7296052e00,
            1.0614292e01,
        ],
    )
    parser.add_argument(
        "--traindatastd",
        nargs="+",
        type=int,
        default=[
            8.982184,
            8.968291,
            2.0443368,
            102.18194,
            140.57402,
            140.01491,
            1.8774419,
            93.93305,
            5.929493,
            5.9249625,
            7.0021815,
        ],
    )

    # device
    parser.add_argument("--no_cuda", default=False, action="store_true")

    # data path
    parser.add_argument(
        "--test_path",
        type=str,
        default="dataset/tracks10/origin/test_2024_04_08__14_46_12.csv",
    )
    # model path
    parser.add_argument(
        "--model_ckpt_path",
        type=str,
        default="./checkpoint/20240415_12_54_44_tracks10_ckpt_add12feat_decoup/20240415_13_07_16.chkpt",
    )

    parser.add_argument("--holdnum", type=int, default=10)
    # save path
    parser.add_argument("--eval_save_path", type=str, default="./prediction/")

    parser.add_argument("--det_keep_rate", type=float, default=1.0)

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":

    opt = parse_args_()

    # data param
    past = opt.len_established
    cand = opt.len_future
    near = opt.near

    now = int(round(time.time() * 1000))
    nowname = time.strftime("%Y%m%d_%H_%M_%S", time.localtime(now / 1000))

    test_det_pa = opt.test_path
    model_p = opt.model_ckpt_path
    output_csv_pa = os.path.join(opt.eval_save_path, nowname, "track_result.csv")
    os.makedirs(os.path.join(opt.eval_save_path, nowname))
    save_args_to_file(opt, os.path.join(opt.eval_save_path, nowname, "param.txt"))
    pyfilepath = os.path.abspath(__file__)
    shutil.copy(
        pyfilepath,
        os.path.join(opt.eval_save_path, nowname, os.path.split(pyfilepath)[-1]),
    )
    shutil.copy(
        os.path.join(os.path.split(pyfilepath)[0], "engine/inference.py"),
        os.path.join(opt.eval_save_path, nowname, "inference.py"),
    )
    keep_track = tracking(
        input_detfile=test_det_pa,
        output_trackcsv=output_csv_pa,
        model_path=model_p,
        fract=opt.det_keep_rate,
        Past=past,
        Cand=cand,
        Near=near,
        no_cuda=opt.no_cuda,
        holdnum=opt.holdnum,
        mean_=opt.traindatamean,
        std_=opt.traindatastd,
    )

    xmlfilepath = output_csv_pa.replace(".csv", ".xml")
    resultcsv_2xml(xmlfilepath, output_csv_pa)
