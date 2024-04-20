import glob
import os
import pandas as pd


metric_root = "/mnt/data1/ZYDdata/code/MoTT/prediction"

allmetric_filepathlist = glob.glob(os.path.join(metric_root, "**cascade**"))

columnsname = [
    "method",
    "test_2024_04_08__14_44_54_alpha",
    "test_2024_04_08__14_44_54_beta",
    "test_2024_04_08__14_44_54_jsct",
    "test_2024_04_08__14_44_54_jsc",
    "test_2024_04_08__14_45_08_alpha",
    "test_2024_04_08__14_45_08_beta",
    "test_2024_04_08__14_45_08_jsct",
    "test_2024_04_08__14_45_08_jsc",
    "test_2024_04_08__14_45_29_alpha",
    "test_2024_04_08__14_45_29_beta",
    "test_2024_04_08__14_45_29_jsct",
    "test_2024_04_08__14_45_29_jsc",
    "test_2024_04_08__14_45_58_alpha",
    "test_2024_04_08__14_45_58_beta",
    "test_2024_04_08__14_45_58_jsct",
    "test_2024_04_08__14_45_58_jsc",
    "test_2024_04_08__14_46_12_alpha",
    "test_2024_04_08__14_46_12_beta",
    "test_2024_04_08__14_46_12_jsct",
    "test_2024_04_08__14_46_12_jsc",
]


pres_df = pd.DataFrame(columns=columnsname)
for tkrepa in allmetric_filepathlist:
    methed_name = os.path.split(tkrepa)[1]
    adict = {"method": [methed_name]}
    allthismethod_testmetrics = glob.glob(os.path.join(tkrepa, "**"))
    for onemetricfile in allthismethod_testmetrics:
        parampa = os.path.join(onemetricfile, "param.txt")
        with open(parampa, "r") as File:
            alllines = File.readlines()
        File.close()
        ref_line = alllines[6]
        if ref_line[:9] != "test_path":
            for line in alllines:
                if line[:9] == "test_path":
                    ref_line = line
                    break

        assert ref_line[:9] == "test_path"

        ref_path = ref_line.rstrip().rstrip("\n").replace("test_path: ", "")
        ref_name = os.path.split(ref_path)[-1].replace(".csv", "")

        testname = ref_name
        with open(os.path.join(onemetricfile, "track_result.txt"), "r") as Fi:
            allline = Fi.readlines()
            alpha = eval(allline[1].split(":")[0].strip())
            beta = eval(allline[2].split(":")[0].strip())
            jsct = eval(allline[5].split(":")[0].strip())
            jsc = eval(allline[11].split(":")[0].strip())
        Fi.close()
        adict[f"{testname}_alpha"] = [alpha * 100]
        adict[f"{testname}_beta"] = [beta * 100]
        adict[f"{testname}_jsct"] = [jsct * 100]
        adict[f"{testname}_jsc"] = [jsc * 100]
    pres_df = pres_df.append(pd.DataFrame(adict))

pres_df = pres_df.sort_values(by="method")
pres_df.to_csv("./merge_cascade_metrics.csv")
