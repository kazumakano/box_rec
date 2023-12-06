import os.path as path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import yaml
from torchvision.transforms import functional as TF


Param = bool | float | int

def aug_img(img: torch.Tensor, aug_num: int) -> torch.Tensor:
    """
    Augment image by rotation.

    Parameters
    ----------
    img : Tensor[float32]
        Original image.
        Shape is (channel, height, width).
    aug_num : int
        The number of images to augment.

    Returns
    -------
    imgs : Tensor[float32]
        Augmented images.
        Shape is (aug_num, channel, height, width).
    """

    auged_imgs = torch.empty((aug_num, *img.shape), dtype=torch.float32)
    for i in range(aug_num):
        auged_imgs[i] = TF.rotate(img, 90 * i)

    return auged_imgs

def extract_box(box_info: pd.DataFrame, frm: np.ndarray) -> np.ndarray:
    """
    Extract box images from video frame.

    Parameters
    ----------
    box_info : DataFrame
        Location information of every box.
    frm : ndarray[uint8]
        Frame image.
        Shape is (height, width, channel).

    Returns
    -------
    imgs : ndarray[uint8]
        Images of every box.
        Shape is (box_num, height, width, channel).
    """

    box_imgs = np.empty((len(box_info), 64, 64, 3), dtype=np.uint8)
    for i, b in box_info.iterrows():
        box_imgs[i] = frm[b["y0"]:b["y1"] + 1, b["x0"]:b["x1"] + 1]

    return box_imgs

def get_result_dir(dir_name: str | None) -> str:
    if dir_name is None:
        dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return path.join(path.dirname(__file__), "../result/", dir_name)

def load_param(file: str) -> dict[str, Param | list[Param] | list[str]]:
    with open(file) as f:
        return yaml.safe_load(f)

def random_split(files: list[str], prop: tuple[float, float, float], seed: int = 0) -> tuple[list[str], list[str], list[str]]:
    mixed_idxes = torch.randperm(len(files), generator=torch.Generator().manual_seed(seed), dtype=torch.int32).numpy()

    train_num = round(prop[0] * len(mixed_idxes) / sum(prop))
    train_files = []
    for i in mixed_idxes[:train_num]:
        train_files.append(files[i])

    val_num = round(prop[1] * len(mixed_idxes) / sum(prop))
    val_files = []
    for i in mixed_idxes[train_num:train_num + val_num]:
        val_files.append(files[i])

    test_files = []
    for i in mixed_idxes[train_num + val_num:]:
        test_files.append(files[i])

    return train_files, val_files, test_files
