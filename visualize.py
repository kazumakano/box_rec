import cv2
import pandas as pd
import torch
from scipy.special import softmax
from torch import jit
from torchvision.transforms import functional as TF
import script.utility as util
from script.data import Usage

COLORS = {
    "floor": (255, 255, 255),
    "item": (0, 255, 0),
    "empty_pallet": (0, 0, 0),
    "hand_pallet": (0, 0, 255),
    "other": (255, 0, 0)
}

def vis(box_info_file: str, gpu_id: int, model_file: str, scale: float, vid_file: str) -> None:
    box_info = pd.read_csv(box_info_file)
    cap = cv2.VideoCapture(filename=vid_file)
    device = torch.device("cuda", gpu_id)
    model = jit.load(model_file, map_location=device)

    with torch.no_grad():
        while True:
            ret, frm = cap.read()
            if not ret:
                break

            input = torch.empty((len(box_info), 3, 64, 64), dtype=torch.float32)
            for i, img in enumerate(util.extract_box(box_info, frm)):
                input[i] = TF.to_tensor(img)

            p: int
            for i, p in enumerate(softmax(model(input.to(device=device)).cpu().numpy(), axis=1).argmax(axis=1)):
                frm = cv2.rectangle(
                    frm,
                    (box_info.loc[i, "l"], box_info.loc[i, "t"]),
                    (box_info.loc[i, "r"], box_info.loc[i, "b"]),
                    COLORS[tuple(Usage)[p].name.lower()],
                    thickness=10
                )

            cv2.imshow("prediction", cv2.resize(frm, None, fx=scale, fy=scale))
            if cv2.waitKey(delay=1) != -1:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--box_info_file", required=True, help="specify box info file", metavar="PATH_TO_BOX_INFO_FILE")
    parser.add_argument("-m", "--model_file", required=True, help="specify compiled model file", metavar="PATH_TO_MODEL_FILE")
    parser.add_argument("-v", "--vid_file", required=True, help="specify video file", metavar="PATH_TO_VID_FILE")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="specify GPU device ID", metavar="GPU_ID")
    parser.add_argument("-s", "--scale", default=0.5, type=float, help="specify scale to show", metavar="SCALE")
    args = parser.parse_args()

    vis(args.box_info_file, args.gpu_id, args.model_file, args.scale, args.vid_file)
