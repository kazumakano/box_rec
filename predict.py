import csv
import cv2
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch import jit
from torchvision.transforms import functional as TF
from tqdm import tqdm
import script.utility as util
from script.data import Usage


@torch.no_grad
def predict(box_info_file: str, gpu_id: int, model_file: str, pj_file: str, result_file: str, vid_file: str) -> None:
    offset = util.crop({cn: np.array(p["projective_matrix"], dtype=np.float32) for cn, p in util.load_param(pj_file).items()})[2]
    box_info = pd.read_csv(box_info_file, usecols=("no", "x", "y"))
    box_info.loc[:, "x"] -= offset[0]
    box_info.loc[:, "y"] -= offset[1]
    cap = cv2.VideoCapture(filename=vid_file)
    device = torch.device("cuda", gpu_id)
    model = jit.load(model_file, map_location=device)

    with open(result_file, mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(("frm_idx", "no", "cls", "conf"))

        bar = tqdm(desc="predicting", total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        while True:
            ret, frm = cap.read()
            if not ret:
                break

            input = torch.empty((len(box_info), 3, 64, 64), dtype=torch.float32)
            for i, img in enumerate(util.extract_box_v2(box_info, 110, frm)):
                input[i] = TF.to_tensor(img)

            output: torch.Tensor = model(input.to(device=device))

            p: np.ndarray
            for i, p in enumerate(softmax(output.cpu().numpy(), axis=1)):
                writer.writerow((int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1, int(box_info.iloc[i]["no"]), tuple(Usage)[p.argmax()].name.lower(), format(p.max(), ".2f")))

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
