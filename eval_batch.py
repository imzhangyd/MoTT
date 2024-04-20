import glob
import os
import subprocess


track_respath = "/mnt/data1/ZYDdata/code/MoTT/prediction/20240419_cascade_v32_hold0"
alltrackrepathlist = glob.glob(os.path.join(track_respath, "**"))

for repa in alltrackrepathlist:
    parampa = os.path.join(repa, "param.txt")
    with open(parampa, "r") as File:
        alllines = File.readlines()

    ref_line = alllines[6]
    if ref_line[:9] != "test_path":
        for line in alllines:
            if line[:9] == "test_path":
                ref_line = line
                break

    assert ref_line[:9] == "test_path"

    ref_path = ref_line.rstrip().rstrip("\n").replace("test_path: ", "")
    ref_name = os.path.split(ref_path)[-1].replace(".csv", "")

    ref = f"/mnt/data1/ZYDdata/code/MoTT_particle_notsame/MoTT/dataset/tracks10/GTxml/testdata/{ref_name}.xml"
    can = os.path.join(repa, "track_result.xml")
    out = can.replace(".xml", ".txt")
    subprocess.call(
        [
            "java",
            "-jar",
            "trackingPerformanceEvaluation.jar",
            "-r",
            ref,
            "-c",
            can,
            "-o",
            out,
        ]
    )

    print("[Info] Finish evaluating")
    print(f"Save file:{out}")
