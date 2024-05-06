# vis tracks along frames
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import skimage.io as io

import seaborn as sns
import random
import argparse


__author__ = "Yudong Zhang"


palette = sns.color_palette("hls", 30)


def get_color(seed):
    random.seed(seed)
    # random color
    bbox_color = random.choice(palette)
    bbox_color = [int(255 * c) for c in bbox_color][::-1]
    cl = (
        "#"
        + hex(bbox_color[0])[-2:]
        + hex(bbox_color[1])[-2:]
        + hex(bbox_color[2])[-2:]
    )
    cl = cl.upper()
    return cl


def parse_args_():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--imgfolder",
        type=str,
        default="/data/ldap_shared/synology_shared/zyd/data/20220611_detparticle/challenge/MICROTUBULE snr 7 density low",
    )
    parser.add_argument(
        "--trackcsvpath",
        type=str,
        default="./prediction/20240301_15_25_56/track_result.csv",
    )
    parser.add_argument(
        "--vis_save", type=str, default="./prediction/20240301_15_25_56/track_vis"
    )
    parser.add_argument("--img_fmt", type=str, default="**t{:03d}**.tif")
    parser.add_argument("--vis_dot", default=False, action="store_true")

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":

    opt = parse_args_()

    pos_resize_ratio = 1200 / 132.0
    result_pa = opt.trackcsvpath
    imgfolder = opt.imgfolder
    savefolder = opt.vis_save
    os.makedirs(savefolder, exist_ok=True)

    result = pd.read_csv(result_pa, header=0)
    if result["frame"].values.min() == 1:
        result["frame"] -= 1

    filename = result_pa.split("/")[-1].replace(".csv", "")

    print("[Info] Start")
    for fr in range(0, int(result["frame"].values.max()) + 1):

        if fr % (int(result["frame"].values.max()) + 1 // 5) == 0:
            print(f"[Info] Visualize frame:{fr}")
        # if 'snr' in
        imgpath = glob.glob(os.path.join(imgfolder, opt.img_fmt.format(fr)))
        assert len(imgpath) == 1

        img = io.imread(imgpath[0])

        plt.figure()
        plt.imshow(img, "gray")
        plt.axis("off")
        thisframe_det = result[result["frame"] == fr]
        for nu in range(len(thisframe_det)):
            the_id = thisframe_det.iloc[nu].loc["trackid"]
            #         print(the_id)
            ID_color = get_color(the_id)
            this_iddet = result[result["trackid"] == the_id].sort_values("frame")
            this_iddet_near = this_iddet[
                (this_iddet["frame"] <= fr)
            ]  # &(this_iddet['frame']>fr-10)
            plt.plot(
                this_iddet_near["pos_x"] * pos_resize_ratio,
                this_iddet_near["pos_y"] * pos_resize_ratio,
                linewidth=0.5,
                color=ID_color,
            )
            if opt.vis_dot:
                plt.scatter(
                    [this_iddet_near["pos_x"].values[-1] * pos_resize_ratio],
                    [this_iddet_near["pos_y"].values[-1] * pos_resize_ratio],
                    color=ID_color,
                    marker="o",
                    edgecolors=ID_color,
                    s=1,
                    linewidths=1,
                )

        plt.savefig(
            os.path.join(savefolder, "%03d.jpg" % fr),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.0,
        )
        plt.close()
    #     break
    print("[Info] Success!")
