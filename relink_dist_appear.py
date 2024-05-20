"""
Relink the tracklets of MoTT linking results.
We use the dist and appearance for relink here.

"""

import pandas as pd
import numpy as np
import os
import ipdb
import pulp


MAX_GAP = 2
MAX_DIST = 20
MAX_APP_DIST = 20


# pulp solver
def maximization_pulp_solver(costs, trackid_list, detid_list):
    """
    costs: a list (n * 3), looks like [[cost0, trackid_0, detid_0], [cost1, trackid_1, detid_1],...], costs dtype is [[float, int, int]]
    trackid_list: trackid_0,trackid_1,...
    detid_list: detid_0,detid_1,...
    """
    # duplicate removal
    df = pd.DataFrame(np.array(costs))
    df_unique = df.drop_duplicates(subset=[1, 2])
    df_unique[[1, 2]] = df_unique[[1, 2]].astype(int)
    costs = [
        [a, b, c]
        for a, b, c in zip(
            df_unique[0].values.tolist(),
            df_unique[1].values.tolist(),
            df_unique[2].values.tolist(),
        )
    ]

    # set variables
    varis = []
    for i in range(len(costs)):
        if costs[i][2] == -1:
            vari = str(costs[i][1]) + "_" + "x"
        else:
            vari = str(costs[i][1]) + "_" + str(costs[i][2])
        varis.append(vari)
    mottVaris = [
        pulp.LpVariable(f"z_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary)
        for i in varis
    ]

    # set objective
    mottProDp = pulp.LpProblem("mottProDp", sense=pulp.const.LpMaximize)
    ss = []
    for varIndex in range(len(mottVaris)):
        s = mottVaris[varIndex] * costs[varIndex][0]
        ss.append(s)
    mottProDp += pulp.lpSum(ss), "Objective Function"

    # set constraints
    for j in trackid_list:
        flag = False
        exprc = []
        for cc in range(len(costs)):
            if costs[cc][1] == j:
                flag = True
                exprc.append(mottVaris[cc])
        if flag:
            mottProDp += pulp.lpSum(exprc) <= 1

    for j in detid_list:
        exprcj = []
        flag = False
        for cc in range(len(costs)):
            if costs[cc][2] == j:
                flag = True
                exprcj.append(mottVaris[cc])
        if flag:
            mottProDp += pulp.lpSum(exprcj) <= 1

    # set solver and optimize
    pulp_dir = os.path.dirname(pulp.__file__)
    solver = pulp.COIN_CMD(
        path=os.path.join(pulp_dir, "solverdir/cbc/linux/64/cbc"), msg=False
    )
    mottProDp.solve(solver)

    # analyze solution
    solutions = []
    for v in mottProDp.variables():
        if v.varValue > 0.5:
            tempv = str(v)
            sp1 = tempv.split("_")[1]
            sp2 = tempv.split("_")[2]
            if sp2 == "x":
                tempv = "z_" + sp1 + "_-1"
                solutions.append(tempv)
            else:
                solutions.append(str(v))
    return solutions


def find_cand(thetk, candi_firstinfo, MAX_GAP=2, MAX_DIST=512 * 0.05, MAX_APP_DIST=50):

    thetk_endposx = thetk.iloc[-1]["pos_x"]
    thetk_endposy = thetk.iloc[-1]["pos_y"]
    thetk_end_intensity = thetk.iloc[-1]["intensity"]

    candi_firstinfo["dist"] = 0.0
    candi_firstinfo["app_dist"] = 0.0
    # ipdb.set_trace()
    for i in range(len(candi_firstinfo)):
        candi_firstinfo.iloc[i, candi_firstinfo.columns.get_loc("dist")] = np.sqrt(
            (candi_firstinfo.iloc[i]["pos_x"] - thetk_endposx) ** 2
            + (candi_firstinfo.iloc[i]["pos_y"] - thetk_endposy) ** 2
        )

        candi_firstinfo.iloc[i, candi_firstinfo.columns.get_loc("app_dist")] = np.abs(
            candi_firstinfo.iloc[i]["intensity"] - thetk_end_intensity
        )

    res = candi_firstinfo[candi_firstinfo["dist"] < MAX_DIST]
    res = res[(res["app_dist"] < MAX_APP_DIST) & (res["dist"] < MAX_DIST)]
    return res


if __name__ == "__main__":

    result_csvpath = "/mnt/data1/ZYDdata/code/MoTT_particle_notsame/MoTT/prediction/20240514_FN_baseline/20240515_05_23_57/track_result.csv"
    # 0 读取原跟踪结果
    result_csv = pd.read_csv(result_csvpath, index_col=0)

    # 1 统计所有轨迹，各个帧都有哪些轨迹开始，轨迹终止
    start_tkids = {}
    end_tkids = {}

    alltkid = set(result_csv["trackid"].values.tolist())
    # ipdb.set_trace()
    for tkid in alltkid:
        thistrack = result_csv[result_csv["trackid"] == tkid]
        start_fr = thistrack["frame"].values.min()
        end_fr = thistrack["frame"].values.max()

        if start_fr in start_tkids.keys():
            start_tkids[start_fr] += [tkid]
        else:
            start_tkids[start_fr] = [tkid]

        if end_fr in end_tkids.keys():
            end_tkids[end_fr] += [tkid]
        else:
            end_tkids[end_fr] = [tkid]

    # print(end_tkids[3])
    # 2 遍历所有的终止帧处理relink
    allendtkid_frames = list(end_tkids.keys())
    allendtkid_frames.sort()
    # print(allendtkid_frames)
    while len(allendtkid_frames) > 0:
        end_fr = allendtkid_frames[0]
        print(f"[Info] relink end_fr={end_fr}")

        # 终止轨迹ids
        endtracks = end_tkids[end_fr]
        # print(f"end trackis ids:{endtracks}")
        if len(endtracks) == 0:
            allendtkid_frames = [it for it in allendtkid_frames if it > end_fr]
            continue
        # 可以连的轨迹ids
        tolinktracks = []
        for i in range(end_fr + 1, end_fr + MAX_GAP + 1):
            if i in start_tkids.keys():
                tolinktracks += start_tkids[i]

        if len(tolinktracks) == 0:
            allendtkid_frames = [it for it in allendtkid_frames if it > end_fr]
            continue
        # ipdb.set_trace()
        # 根据手工特征过滤和生成cost
        costs = []
        tolinktracks_allspot = result_csv[result_csv["trackid"].isin(tolinktracks)]
        candi_firstinfo = pd.DataFrame(columns=result_csv.columns)
        for tkid, content in tolinktracks_allspot.groupby("trackid"):
            content = content.sort_values("frame")
            candi_firstinfo = candi_firstinfo.append(content.iloc[0])
        # ipdb.set_trace()
        for oneendtrack in endtracks:
            thetk = result_csv[result_csv["trackid"] == oneendtrack].sort_values(
                by="frame"
            )
            if len(thetk) == 0:
                print(f"[ERROE] track id {oneendtrack} is Null")
            thetk_endposx = thetk.iloc[-1]["pos_x"]
            thetk_endposy = thetk.iloc[-1]["pos_y"]
            # 手工特征，规则方法
            canditklist = find_cand(
                thetk, candi_firstinfo, MAX_GAP=2, MAX_DIST=20, MAX_APP_DIST=512 * 0.05
            )
            for i in range(len(canditklist)):
                costs.append(
                    [
                        canditklist.iloc[i]["app_dist"],
                        int(oneendtrack),
                        int(canditklist.iloc[i]["trackid"]),
                    ]
                )

        # 求解
        solution = maximization_pulp_solver(costs, endtracks, tolinktracks)
        # ipdb.set_trace()
        # 关联，更新，无插值
        for so in solution:

            link_track_id = int(so.split("_")[1])
            link_cand_id = int(so.split("_")[2])
            # print(f"[Info] link track {link_track_id} with {link_cand_id}")

            endtk = result_csv[result_csv["trackid"] == link_track_id]
            statk = result_csv[result_csv["trackid"] == link_cand_id]
            statk = statk.sort_values("frame")
            r_fr = statk.iloc[0]["frame"]
            r_end_fr = statk.iloc[-1]["frame"]
            endtk = endtk.sort_values("frame")
            l_fr = endtk.iloc[-1]["frame"]
            l_start_fr = endtk.iloc[0]["frame"]

            # print(f"track{link_track_id}:frame{l_start_fr}---{l_fr}")
            # print(f"track{link_cand_id}:frame{r_fr}---{r_end_fr}")

            if r_fr - l_fr > 1:
                pass

            result_csv.loc[result_csv.trackid == link_cand_id, "trackid"] = (
                link_track_id
            )
            # print(f"end tkid frame{end_fr}:{end_tkids[end_fr]}")
            # print(f"end tkid frame{r_end_fr}:{end_tkids[r_end_fr]}")
            # print(f"start tkid frame{l_fr}:{start_tkids[l_fr]}")
            end_tkids[end_fr].remove(link_track_id)
            end_tkids[r_end_fr] += [link_track_id]

            end_tkids[r_end_fr].remove(link_cand_id)
            start_tkids[r_fr].remove(link_cand_id)
        # ipdb.set_trace()
        allendtkid_frames = list(end_tkids.keys())
        allendtkid_frames.sort()
        allendtkid_frames = [it for it in allendtkid_frames if it > end_fr]

result_csv.to_csv(os.path.join(os.path.split(result_csvpath)[0], "relink_result.csv"))
print("[Info] Success!")
