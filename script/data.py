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

class UsageV2(IntEnum):
    FLOOR        = 0
    GOODS        = 1
    PALLET_EMPTY = 2
    HAND         = 3
    OTHER        = 4

class BoxImgDataset(data.Dataset):
    def __init__(self, files: list[str], aug_num: int = 16, brightness: float = 0.1, contrast: float = 0.1, hue: float = 0.1, saturation: float = 0.1) -> None:
        self.aug_num = aug_num

        jitter_color = util.use_color_jitter(brightness, contrast, hue, saturation)
        flip_and_rot = util.use_flip_and_rot()

        self.img = torch.empty((self.aug_num * len(files), 3, 64, 64), dtype=torch.float32)
        self.label = torch.empty(len(files), dtype=torch.int64)
        for i, f in enumerate(tqdm(files, desc="loading box images")):
            self.img[self.aug_num * i:self.aug_num * i + self.aug_num] = util.aug_img(TF.to_tensor(cv2.imread(f)), self.aug_num, jitter_color, flip_and_rot)
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

class BoxFrmDataset(BoxImgDataset):
    def __init__(self, annots: dict[str, list[dict[str, int | str] | dict[str, float | int | list[float]]]], data_dir: str, img_size: int, pjs: dict[str, np.ndarray], stitched_frm_size: tuple[int, int], aug_num = 16, brightness = 0.1, contrast = 0.1, hue = 0.1, saturation = 0.1):
        self.aug_num = aug_num

        jitter_color = util.use_color_jitter(brightness, contrast, hue, saturation)
        flip_and_rot = util.use_flip_and_rot()

        self.img = torch.empty((self.aug_num * len(annots["annotations"]), 3, img_size, img_size))
        self.label = torch.empty(len(annots["annotations"]), dtype=torch.int64)
        for j, a in enumerate(annots["annotations"]):
            for i in annots["images"]:
                if i["id"] == a["image_id"]:
                    pj = pjs[i["file_name"].split("_")[0]]
                    tf_center = cv2.perspectiveTransform(np.array((a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2), dtype=np.float32)[np.newaxis, np.newaxis], pj).squeeze(axis=(0, 1))
                    warped_frm = cv2.warpPerspective(cv2.imread(path.join(data_dir, "original/", i["file_name"])), pj, stitched_frm_size)
                    ori_img = TF.resized_crop(TF.to_tensor(warped_frm), round(tf_center[0] - 0.859375 * img_size), round(tf_center[1] - 0.859375 * img_size), round(1.71875 * img_size), round(1.71875 * img_size), (img_size, img_size))    # 110 / 64 = 1.71875
                    self.img[self.aug_num * j:self.aug_num * j + self.aug_num] = util.aug_img(ori_img, self.aug_num, jitter_color, flip_and_rot)
                    break
            for c in annots["categories"]:
                if c["id"] == a["category_id"]:
                    self.label[j] = UsageV2[c["name"].upper()]
                    break

    @property
    def breakdown(self) -> np.ndarray:
        breakdown = np.empty(len(UsageV2), dtype=np.int32)
        for i, u in enumerate(UsageV2):
            breakdown[i] = torch.count_nonzero(self.label == u)

        return breakdown

class ImgDataModule(pl.LightningDataModule):
    def __init__(self, param: dict[str | util.Param], box_dirs: Optional[list[str]] = None, prop: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0) -> None:
        random.seed(a=seed)
        super().__init__()

        self.dataset: dict[str, BoxImgDataset] = {}
        self.save_hyperparameters(param)

        if box_dirs is not None:
            files: dict[Usage, list[str]] = {}
            for u in Usage:
                files[u] = []
            if self.hparams["max_num_per_usage"] is not None:
                cnt = {}
                for u in Usage:
                    cnt[u] = 0
            for d in random.sample(box_dirs, len(box_dirs)):
                box_img_files = glob(path.join(d, "*_*_*_*.jpg"))
                for f in random.sample(box_img_files, len(box_img_files)):
                    label = Usage[path.splitext(path.basename(f))[0].split("_", 3)[3].upper()]

                    if self.hparams["max_num_per_usage"] is not None:
                        cnt[label] += 1
                        if cnt[label] > self.hparams["max_num_per_usage"]:
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
                    self.dataset["train"] = BoxImgDataset(self.train_files)
                    self.dataset["validate"] = BoxImgDataset(self.val_files, 1)
            case "test":
                self.dataset["test"] = BoxImgDataset(self.test_files, 1)

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

class FrmDataModule(ImgDataModule):
    def __init__(self, param: dict[str | util.Param], data_dir: Optional[str] = None, pj_file: Optional[str] = None, prop: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0) -> None:
        random.seed(a=seed)
        super().__init__()

        self.dataset: dict[str, BoxImgDataset] = {}
        self.save_hyperparameters(param)

        if data_dir is not None and pj_file is not None:
            self.data_dir = data_dir
            self.pjs, self.stitched_frm_size = util.crop({n: np.array(p["projective_matrix"], dtype=np.float64) for n, p in util.load_param(pj_file).items()})

            annots: dict[str, list[dict[str, int | str] | dict[str, float | int | list[float]]]] = util.load_param(path.join(self.data_dir, "coco/annotations.json"))

            pruned_annots: dict[Usage, list[str]] = {}
            for u in Usage:
                pruned_annots[u] = []
            if self.hparams["max_num_per_usage"] is not None:
                cnt = {}
                for u in Usage:
                    cnt[u] = 0
            for a in random.sample(annots["annotations"], len(annots["annotations"])):
                for c in annots["categories"]:
                    if c["id"] == a["category_id"]:
                        label = UsageV2[c["name"].upper()]
                        break

                if self.hparams["max_num_per_usage"] is not None:
                    cnt[label] += 1
                    if cnt[label] > self.hparams["max_num_per_usage"]:
                        continue

                pruned_annots[label].append(a)

            self.train_annots, self.val_annots, self.test_annots = [], [], []
            for l in pruned_annots.values():
                tmp = util.random_split(l, prop, seed)
                self.train_annots += tmp[0]
                self.val_annots += tmp[1]
                self.test_annots += tmp[2]

    def setup(self, stage: str) -> None:
        annots = util.load_param(path.join(self.data_dir, "coco/annotations.json"))
        match stage:
            case "fit":
                if "train" not in self.dataset.keys():
                    self.dataset["train"] = BoxFrmDataset({**annots, "annotations": self.train_annots}, self.data_dir, self.hparams["img_size"], self.pjs, self.stitched_frm_size)
                    self.dataset["validate"] = BoxFrmDataset({**annots, "annotations": self.val_annots}, self.data_dir, self.hparams["img_size"], self.pjs, self.stitched_frm_size, 1)
            case "test":
                self.dataset["test"] = BoxFrmDataset({**annots, "annotations": self.test_annots}, self.data_dir, self.hparams["img_size"], self.pjs, self.stitched_frm_size, 1)
