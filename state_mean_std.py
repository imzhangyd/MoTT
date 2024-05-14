"""
统计一个跟踪结果的所有轨迹的特征的mean std
"""

import numpy as np
import ipdb
import pandas as pd


__author__ = "Yudong Zhang"


track_result_csvpa = "/mnt/data1/ZYDdata/code/MoTT_particle_notsame/MoTT/prediction/20240429_10_39_19/track_result.csv"

track_result = pd.read_csv(track_result_csvpa, header=0, index_col=0)
print(track_result.head())

datapathlist = []

for tkid, line in track_result.groupby("trackid"):
    line = line.sort_values(by="frame")
    passed = [
        [x, y, size, intensity, 0] for tki, x, y, size, intensity, _ in line.values
    ]
    try:

        passed_np = np.array(passed)  # x y size intensity flag(-1 or 0)

        # shift of passed
        passed_shift = (
            passed_np[1:, :-1] - passed_np[:-1, :-1]
        )  # s_x, s_y, s_size, s_inten
        start_ = 0
        if -1 in set(passed_np[:, -1]):
            start_ = np.where(passed_np[:, -1] == -1)[0][-1] + 1
            passed_shift[:start_, :] = 0  # not real shift
        # flag
        flag_list = [0] * (start_) + [1] * (len(passed_shift) - (start_))
        flag_np = np.array(flag_list).reshape(-1, 1)
        # ori pos of passed
        passed_pre = passed_np[1:, :].copy()
        if -1 in set(passed_pre[:, -1]):
            passed_pre[: start_ - 1, :] = 0  # not real pos

        # add abs shift x， abs shift y， abs dist
        abs_shift = np.abs(passed_shift[:, :2])
        abs_dist = np.sqrt(abs_shift[:, 0] ** 2 + abs_shift[:, 1] ** 2).reshape([-1, 1])

        # concate shift and ori pos and abs shift dist and flag
        passed_shift = np.concatenate(
            [passed_shift, passed_pre[:, :-1], abs_shift, abs_dist, flag_np], -1
        )  # s_x, s_y, s_size, s_inten,x, y, size, inten, abs shiftx, abs shift y, abs dist,flag(0 or 1)

        datapathlist.append(passed_shift)
    except TypeError:
        print(line)


reshapelen = len(datapathlist[0][0])
t_passed_ = np.concatenate(datapathlist, 0)
t_shift = t_passed_[t_passed_[:, -1] == 1]

t_mean = t_shift.astype(np.float32).mean(0)
t_std = t_shift.astype(np.float32).std(0)


print(t_mean[:-1])
print(t_std[:-1])
