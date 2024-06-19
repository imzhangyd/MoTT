import pandas as pd
import glob
import os

__author__ = "Yudong Zhang"


def sythetic_2xml(xmlfilepath, output_csv_pa, testfilename=None):

    result_csv = pd.read_csv(output_csv_pa)
    if testfilename:
        snr = testfilename.split(" ")[2]
        dens = testfilename.split(" ")[-1]
        scenario = testfilename.split(" ")[0]
    else:
        snr = 0
        dens = "none"
        scenario = "none"
    method = "_MoTT"
    thrs = 0

    t_trackid = list(set(result_csv["ID"]))
    # csv to xml
    with open(xmlfilepath, "w+") as output:
        output.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
        output.write("<root>\n")
        output.write(
            '<TrackContestISBI2012 SNR="'
            + str(snr)
            + '" density="'
            + dens
            + '" scenario="'
            + scenario
            + '" '
            + method
            + '="'
            + str(thrs)
            + '">\n'
        )

        for trackid in t_trackid:
            thistrack = result_csv[result_csv["ID"] == trackid]
            if len(thistrack) > 1:
                thistrack.sort_values("frame", inplace=True)
                thistrack_np = thistrack[["w_position", "h_position", "frame"]].values

                output.write("<particle>\n")
                for pos in thistrack_np:
                    output.write(
                        '<detection t="'
                        + str(int(pos[-1]))
                        + '" x="'
                        + str(pos[0])
                        + '" y="'
                        + str(pos[1])
                        + '" z="0"/>\n'
                    )
                output.write("</particle>\n")
        output.write("</TrackContestISBI2012>\n")
        output.write("</root>\n")
        output.close()


if __name__ == "__main__":

    src_folder = glob.glob("../data/tracks10_itp1/**.csv")
    for src_csv in src_folder:
        GT_xmlpa = os.path.join(
            os.path.split(src_csv)[0],
            "GTxml",
            os.path.split(src_csv)[1].replace(".csv", ".xml"),
        )
        print(GT_xmlpa)
        os.makedirs(os.path.split(GT_xmlpa)[0], exist_ok=True)
        sythetic_2xml(GT_xmlpa, src_csv, None)
