"""
统计跟踪结果的距离分布
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


thecsv = pd.read_csv(
    "/mnt/data1/ZYDdata/code/MoTT_particle_notsame/MoTT/prediction/20240429_10_39_19/track_result.csv"
)
print(thecsv.head())

alldist = []
longdist = []
longfrm = []
alltkids = set(thecsv["trackid"].values)
for tkid in alltkids:
    onetrack = thecsv[thecsv["trackid"] == tkid].sort_values(by="frame")
    posx = onetrack["pos_x"].values
    posy = onetrack["pos_y"].values
    frms = onetrack["frame"].values

    distnp = (
        np.sqrt((posx[1:] - posx[:-1]) ** 2 + (posy[1:] - posy[:-1]) ** 2)
        * 1200
        / 132.0
    )
    alldist += distnp.tolist()
    longdist += distnp[np.where(distnp > 30)].tolist()
    longfrm += frms[np.where(distnp > 30)].tolist()


print(longdist)
print(longfrm)

print(np.mean(alldist), np.std(alldist))
print(np.max(alldist))

plt.figure()
plt.hist(alldist, bins=100)
plt.savefig(
    "/mnt/data1/ZYDdata/code/MoTT_particle_notsame/MoTT/prediction/20240429_10_39_19/track_result_state_dist.png"
)

# 设定1/20为阈值
