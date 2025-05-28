import os.path as path
import random
from enum import IntEnum
from glob import glob
from typing import Optional, Self
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils import data
from torchvision.transforms import functional as TF
from tqdm import tqdm
from . import utility as util


class Usage(IntEnum):
    FLOOR        = 0
    ITEM         = 1
    EMPTY_PALLET = 2
    HAND_PALLET  = 3
    OTHER        = 4

class BoxDataset(data.Dataset):
    def __init__(self, files: list[str], rotate: bool = True) -> None:
        self.aug_num = 4 if rotate else 1

        self.img = torch.empty((self.aug_num * len(files), 3, 64, 64), dtype=torch.float32)
        self.label = torch.empty(len(files), dtype=torch.int64)
        for i, f in enumerate(tqdm(files, desc="loading box images")):
            self.img[self.aug_num * i:self.aug_num * i + self.aug_num] = util.aug_img(TF.to_tensor(cv2.imread(f)), self.aug_num)
            self.label[i] = Usage[path.splitext(path.basename(f))[0].split("_", 3)[3].upper()]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.img[idx], self.label[idx // self.aug_num]

    def __len__(self) -> int:
        return len(self.img)

    @property
    def breakdown(self) -> np.ndarray:
        breakdown = np.empty(len(Usage), dtype=np.int32)
        for i, u in enumerate(Usage):
            breakdown[i] = torch.count_nonzero(self.label == u)

        return breakdown

    def calc_loss_weight(self) -> torch.Tensor:
        return torch.from_numpy(1 / (1 / self.breakdown).sum() / self.breakdown).to(dtype=torch.float32)

class DataModule(pl.LightningDataModule):
    def __init__(self, param: dict[str | util.Param], box_dirs: Optional[list[str]] = None, max_num_per_usage: Optional[int] = None, prop: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0) -> None:
        random.seed(a=seed)
        super().__init__()

        self.dataset: dict[str, BoxDataset] = {}
        self.save_hyperparameters(param)

        if box_dirs is not None:
            files: dict[Usage, list[str]] = {}
            for u in Usage:
                files[u] = []
            if max_num_per_usage is not None:
                cnt = {}
                for u in Usage:
                    cnt[u] = 0
            for d in random.sample(box_dirs, len(box_dirs)):
                box_img_files = glob(path.join(d, "*_*_*_*.jpg"))
                for f in random.sample(box_img_files, len(box_img_files)):
                    label = Usage[path.splitext(path.basename(f))[0].split("_", 3)[3].upper()]

                    if max_num_per_usage is not None:
                        cnt[label] += 1
                        if cnt[label] > max_num_per_usage:
                            continue

                    files[label].append(f)

            self.train_files, self.val_files, self.test_files = [], [], []
            for l in files.values():
                tmp = util.random_split(l, prop, seed)
                self.train_files += tmp[0]
                self.val_files += tmp[1]
                self.test_files += tmp[2]

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                if "train" not in self.dataset.keys():
                    self.dataset["train"] = BoxDataset(self.train_files)
                    self.dataset["validate"] = BoxDataset(self.val_files, False)
            case "test":
                self.dataset["test"] = BoxDataset(self.test_files, False)

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset["train"], batch_size=self.hparams["batch_size"], shuffle=self.hparams["shuffle"], num_workers=self.hparams["num_workers"])

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset["validate"], batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"])

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"])

    @classmethod
    def load(cls, dir: str) -> Self:
        dataset, param = torch.load(path.join(dir, "data.pt"))
        self = cls(param)
        self.dataset = dataset

        return self

    def save(self, dir: str) -> None:
        torch.save((self.dataset, self.hparams), path.join(dir, "data.pt"))

    @staticmethod
    def unpack_param_list(param_list: dict[str, list[util.Param]]) -> dict[str, util.Param]:
        return {
            "batch_size": param_list["batch_size"][0],
            "num_workers": param_list["num_workers"][0],
            "shuffle": param_list["shuffle"][0]
        }
