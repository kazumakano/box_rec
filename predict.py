import csv
import cv2
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch import jit
from torchvision.transforms import functional as TF
from tqdm import tqdm
import script.data as D
import script.utility as util


def _compute_offset(pjs: dict[str, np.ndarray]) -> tuple[float, float]:
    offset = [np.inf, np.inf]
    for p in pjs.values():
        tf_corners = cv2.perspectiveTransform(np.array(((0, 0), (1920, 0), (0, 1080)), dtype=np.float32)[:, np.newaxis], p).squeeze(axis=1)
        offset[0] = min(offset[0], tf_corners[0, 0], tf_corners[2, 0])
        offset[1] = min(offset[1], tf_corners[0, 1], tf_corners[1, 1])

    return tuple(offset)

def predict(box_info_file: str, gpu_id: int, model_file: str, pj_file: str, result_file: str, vid_file: str) -> None:
    offset = _compute_offset({cn: np.array(p["projective_matrix"], dtype=np.float32) for cn, p in util.load_param(pj_file).items()})
    box_info = pd.read_csv(box_info_file, usecols=("no", "x", "y"))
    box_info.loc[:, "x"] -= offset[0]
    box_info.loc[:, "y"] -= offset[1]
    cap = cv2.VideoCapture(filename=vid_file)
    device = torch.device("cuda", gpu_id)
    model = jit.load(model_file, map_location=device)

    with open(result_file, mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(("frm_idx", "no", "cls", "conf"))

        with torch.no_grad():
            bar = tqdm(desc="predicting", total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            while True:
                ret, frm = cap.read()
                if not ret:
                    break

                input = torch.empty((len(box_info), 3, 64, 64), dtype=torch.float32)
                for i, img in enumerate(util.extract_box_v2(box_info, 100, frm)):
                    input[i] = TF.to_tensor(img)

                output: torch.Tensor = model(input.to(device=device))

                p: np.ndarray
                for i, p in enumerate(softmax(output.cpu().numpy(), axis=1)):
                    writer.writerow((int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1, int(box_info.iloc[i]["no"]), util.look_up_key_from_val(D.USAGE, p.argmax()), format(p.max(), ".2f")))

                bar.update()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--box_info_file", required=True, help="specify box info file", metavar="PATH_TO_BOX_INFO_FILE")
    parser.add_argument("-m", "--model_file", required=True, help="specify compiled model file", metavar="PATH_TO_MODEL_FILE")
    parser.add_argument("-p", "--pj_file", required=True, help="specify projection matrix file", metavar="PATH_TO_PJ_FILE")
    parser.add_argument("-r", "--result_file", required=True, help="specify result file", metavar="PATH_TO_RESULT_FILE")
    parser.add_argument("-v", "--vid_file", required=True, help="specify stitched video file", metavar="PATH_TO_VID_FILE")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="specify GPU device ID", metavar="GPU_ID")
    args = parser.parse_args()

    predict(args.box_info_file, args.gpu_id, args.model_file, args.pj_file, args.result_file, args.vid_file)
