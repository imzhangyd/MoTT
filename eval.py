import subprocess
import argparse


__author__ = "Yudong Zhang"


def parse_args_():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--GTxmlpath",
        type=str,
        default="dataset/ground_truth/MICROTUBULE snr 7 density mid.xml",
    )
    parser.add_argument(
        "--pred_xmlpath",
        type=str,
        default="dataset/ground_truth/MICROTUBULE snr 7 density mid.xml",
    )

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":

    opt = parse_args_()

    # prepare gt path, result path, and output path
    ref = opt.GTxmlpath
    can = opt.pred_xmlpath
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
