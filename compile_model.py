import os.path as path
from glob import glob
import torch
import script.model as M
import script.utility as util
from script.data import Usage


def compile_model(batch: bool, result_dir: str, ver: int = 0) -> None:
    param = util.load_param(path.join(result_dir, f"version_{ver}/hparams.yaml"))
    M.get_model_cls(param["arch"]).load_from_checkpoint(
        glob(path.join(result_dir, f"version_{ver}/", "checkpoints/", "epoch=*-step=*.ckpt"))[0],
        loss_weight=torch.empty(len(Usage), dtype=torch.float32),
        map_location=torch.device("cuda", 0)
    ).to_torchscript(
        file_path=path.join(result_dir, f"version_{ver}/", "model.pt"),
        method="trace",
        example_inputs=torch.empty(1, 3, param["img_size"], param["img_size"]) if batch else torch.empty(3, param["img_size"], param["img_size"])
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_dir", required=True, help="specify train result directory", metavar="PATH_TO_RESULT_DIR")
    parser.add_argument("-v", "--ver", default=0, type=int, help="specify version", metavar="VER")
    parser.add_argument("-b", "--batch", action="store_true", help="enable batching")
    args = parser.parse_args()

    compile_model(args.batch, args.result_dir, args.ver)
