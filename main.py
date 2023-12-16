import os.path as path
from glob import glob
from typing import Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import script.data as data
import script.utility as util
from script.model import CNN3


def train(box_dir: str, gpu_id: int, param: dict[str, util.Param] | str, ckpt_file: Optional[str] = None, result_dir_name: Optional[str] = None) -> None:
    torch.set_float32_matmul_precision("high")

    if isinstance(param, str):
        param = util.load_param(param)

    datamodule = data.DataModule(param, box_dir)
    trainer = pl.Trainer(
        logger=TensorBoardLogger(util.get_result_dir(result_dir_name), name=None, default_hp_metric=False),
        callbacks=ModelCheckpoint(monitor="validation_loss", save_last=True),
        devices=[gpu_id],
        max_epochs=param["max_epoch"],
        accelerator="gpu"
    )

    if ckpt_file is None:
        datamodule.setup("fit")
        model = CNN3(param, torch.from_numpy(len(datamodule.dataset["train"].label) / datamodule.dataset["train"].breakdown).to(dtype=torch.float32))
        trainer.fit(model, datamodule=datamodule)
        CNN3.load_from_checkpoint(glob(path.join(trainer.log_dir, "checkpoints/", "epoch=*-step=*.ckpt"))[0], loss_weight=torch.empty(len(data.USAGE), dtype=torch.float32))
    else:
        model = CNN3.load_from_checkpoint(ckpt_file, param=param, loss_weight=torch.empty(len(data.USAGE), dtype=torch.float32))

    trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--box_dir", nargs="+", help="specify box dataset directory", metavar="PATH_TO_BOX_DIR")
    parser.add_argument("-g", "--gpu_id", default=0, type=int, help="specify GPU device ID", metavar="GPU_ID")
    parser.add_argument("-r", "--result_dir_name", help="specify result directory name", metavar="RESULT_DIR_NAME")

    if sys.stdin.isatty():
        parser.add_argument("-p", "--param_file", required=True, help="specify parameter file", metavar="PATH_TO_PARAM_FILE")
        parser.add_argument("-c", "--ckpt_file", help="specify checkpoint file", metavar="PATH_TO_CKPT_FILE")
        args = parser.parse_args()

        train(args.ts_fig_dir, args.gpu_id, args.param_file, args.ckpt_file, args.result_dir_name)

    else:
        args = parser.parse_args()
        lines = sys.stdin.readlines()

        train(args.ts_fig_dir, args.gpu_id, json.loads(lines[1]), lines[3].rstrip(), args.result_dir_name)
