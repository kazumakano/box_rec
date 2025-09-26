import csv
import os.path as path
import pickle
from datetime import datetime
from glob import glob
import cv2
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch import jit
from torchvision.transforms import functional as TF
from tqdm import tqdm
import denoise.noise_filter_lib as denoise
import script.utility as util
from script.data import Usage

BEGIN = "00:00:00"
END = "23:59:59"
REC_PERIOD_IN_SEC = 1

@torch.no_grad
def infer(box_info_file: str, gpu_id: int, layout_file: str, model_file: str, pj_file: str, result_file: str, ts_cache_file, vid_dir: str) -> None:
    # load constants for box recognition
    box_info = pd.read_csv(box_info_file, usecols=("no", "x", "y"))
    device = torch.device("cuda", gpu_id)
    layout = util.load_param(layout_file)
    model = jit.load(model_file, map_location=device)
    pjs = {n: np.array(p["projective_matrix"], dtype=np.float32) for n, p in util.load_param(pj_file).items()}

    # load constants for time synchronization
    begin_in_sec = int((datetime.strptime(BEGIN, "%H:%M:%S") - datetime(1900, 1, 1)).total_seconds())
    end_in_sec = int((datetime.strptime(END, "%H:%M:%S") - datetime(1900, 1, 1)).total_seconds())
    with open(ts_cache_file, mode="rb") as f:
        ts_cache: tuple[dict[str, list[tuple[int, int]]], dict[str, int]] = pickle.load(f)

    # prepare constants
    cam_names = {l["camera"] for l in layout}
    denoise_filter = denoise.create_noise_filter()
    denoise_filter.initialize(initial_data={l["box"]: {"usage": Usage.FLOOR.name.lower()} for l in layout}, init_timestamp=begin_in_sec)    # initialize as floor
    pjs, frm_size, offset = util.crop(pjs)
    box_info.loc[:, "x"] -= offset[0]
    box_info.loc[:, "y"] -= offset[1]
    status: dict[str, dict[str, int | np.ndarray | cv2.VideoCapture]] = {}

    # infer
    with open(result_file, mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(("ts", "no", "cls", "conf"))

        for cur_in_sec in tqdm(np.arange(begin_in_sec, end_in_sec, step=REC_PERIOD_IN_SEC), desc="inferring"):
            # read and warp frames
            warped_frms = {}
            for n in cam_names:
                vid_idx, frm_idx = ts_cache[0][n][round(5 * (cur_in_sec - ts_cache[1][n]))]

                if n in status.keys() and status[n]["vid_idx"] != vid_idx:
                    status[n]["cap"].release()
                if n not in status.keys() or status[n]["vid_idx"] != vid_idx:
                    status[n] = {"cap": cv2.VideoCapture(filename=glob(path.join(vid_dir, f"camera{n}/video_??-??-??_{vid_idx:02d}.mp4"))[0]), "vid_idx": vid_idx}

                while status[n]["cap"].get(cv2.CAP_PROP_POS_FRAMES) <= frm_idx:
                    status[n]["frm"] = status[n]["cap"].read()[1]

                warped_frms[n] = cv2.warpPerspective(status[n]["frm"], pjs[n], frm_size)

            # preprocess images
            input = torch.empty((len(layout), 3, 64, 64), dtype=torch.float32)
            for i, l in enumerate(layout):
                input[i] = TF.to_tensor(util.extract_box_v2(box_info.loc[box_info.loc[:, "no"] == l["box"]].reset_index(), 110, warped_frms[l["camera"]]).squeeze(axis=0))

            # predict
            output: torch.Tensor = model(input.to(device=device))

            results = []
            p: int
            for i, p in enumerate(softmax(output.cpu().numpy(), axis=1).argmax(axis=1)):
                results.append({"entity_id": layout[i]["box"], "usage": tuple(Usage)[p].name.lower()})

            # postprocess results
            denoise.process_detection_data(denoise_filter, results, cur_in_sec)

            # save results
            for i, s in denoise_filter.get_state().items():
                match s:
                    case "exist":
                        conf = 1
                        usage = Usage.ITEM.name.lower()
                    case "exist_probably":
                        conf = 0
                        usage = Usage.ITEM.name.lower()
                    case "not_exist":
                        conf = 1
                        usage = Usage.FLOOR.name.lower()
                    case "not_exist_probably":
                        conf = 0
                        usage = Usage.FLOOR.name.lower()
                    case "error":
                        raise RuntimeError("error occured at denoising")

                writer.writerow((util.sec2str(cur_in_sec), i, usage, conf))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--box_info_file", required=True, help="specify box info file", metavar="PATH_TO_BOX_INFO_FILE")
    parser.add_argument("-l", "--layout_file", required=True, help="specify layout file", metavar="PATH_TO_LAYOUT_FILE")
    parser.add_argument("-m", "--model_file", required=True, help="specify compiled model file", metavar="PATH_TO_MODEL_FILE")
    parser.add_argument("-p", "--pj_file", required=True, help="specify projection matrix file", metavar="PATH_TO_PJ_FILE")
    parser.add_argument("-r", "--result_file", required=True, help="specify result file", metavar="PATH_TO_RESULT_FILE")
    parser.add_argument("-t", "--ts_cache_file", required=True, help="specify timestamp cache file", metavar="PATH_TO_TS_CACHE_FILE")
    parser.add_argument("-v", "--vid_dir", required=True, help="specify undistorted video directory", metavar="PATH_TO_VID_DIR")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="specify GPU device ID", metavar="GPU_ID")
    args = parser.parse_args()

    infer(args.box_info_file, args.gpu_id, args.layout_file, args.model_file, args.pj_file, args.result_file, args.ts_cache_file, args.vid_dir)
