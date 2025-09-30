import json
import os.path as path
from datetime import datetime
from typing import Any
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torchvision import transforms as T
from torchvision.transforms import functional as TF


Param = bool | float | int | str | None

def aug_img(img: torch.Tensor, aug_num: int, jitter_color: T.ColorJitter, tf_shape: T.Compose) -> torch.Tensor:
    """
    Augment image by color jittering and shape transformation.

    Parameters
    ----------
    img : Tensor[float32]
        Original image.
        Shape is (channel, height, width).
    aug_num : int
        The number of images to augment.
    jitter_color : ColorJitter
        Function to randomly jitter color.
    tf_shape : Compose
        Function to randomly transform shape.

    Returns
    -------
    imgs : Tensor[float32]
        Augmented images.
        Shape is (aug_num, channel, height, width).
    """

    auged_imgs = torch.empty((aug_num, *img.shape), dtype=torch.float32)
    auged_imgs[0] = img
    for i in range(1, aug_num):
        auged_imgs[i] = tf_shape(jitter_color(img))

    return auged_imgs

def crop(pjs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], tuple[int, int], tuple[float, float]]:
    """
    Add margins or remove paddings to fit stitched images.

    Parameters
    ----------
    pjs : dict[str, ndarray[float64]]
        Dictionary of camera names and original projection matrices.

    Returns
    -------
    pjs : dict[str, ndarray[float64]]
        Dictionary of camera names and cropped projection matrices.
    img_size : tuple[int, int]
        Stitched image size.
    offset : tuple[float, float]
        Coordinate offset.
    """

    stitched_ltrb = [np.inf, np.inf, -np.inf, -np.inf]
    for p in pjs.values():
        tf_corners = cv2.perspectiveTransform(np.array(((0, 0), (1920, 0), (0, 1080), (1920, 1080)), dtype=np.float32)[np.newaxis], p).squeeze(axis=0)
        stitched_ltrb[0] = min(stitched_ltrb[0], tf_corners[0, 0], tf_corners[2, 0])
        stitched_ltrb[1] = min(stitched_ltrb[1], tf_corners[0, 1], tf_corners[1, 1])
        stitched_ltrb[2] = max(stitched_ltrb[2], tf_corners[1, 0], tf_corners[3, 0])
        stitched_ltrb[3] = max(stitched_ltrb[3], tf_corners[2, 1], tf_corners[3, 1])
    pjs = pjs.copy()
    for n, p in pjs.items():
        pjs[n] = np.dot(np.array((
            (1, 0, -stitched_ltrb[0]),
            (0, 1, -stitched_ltrb[1]),
            (0, 0, 1)
        ), dtype=np.float64), p)

    return pjs, (int(stitched_ltrb[2] - stitched_ltrb[0]), int(stitched_ltrb[3] - stitched_ltrb[1])), (stitched_ltrb[0], stitched_ltrb[1])

def extract_box(box_info: pd.DataFrame, frm: np.ndarray) -> np.ndarray:
    """
    Extract box images from video frame.

    Parameters
    ----------
    box_info : DataFrame
        Box location information.
    frm : ndarray[uint8]
        Frame image.
        Shape is (height, width, channel).

    Returns
    -------
    imgs : ndarray[uint8]
        Box images.
        Shape is (box_num, height, width, channel).
    """

    box_imgs = np.empty((len(box_info), 64, 64, 3), dtype=np.uint8)
    for i, b in box_info.iterrows():
        box_imgs[i] = frm[b["t"]:b["b"] + 1, b["l"]:b["r"] + 1]

    return box_imgs

def extract_box_v2(box_info: pd.DataFrame, size: int, frm: np.ndarray) -> np.ndarray:
    """
    Extract box images from video frame and resize them to 64 x 64 [px].

    Parameters
    ----------
    box_info : DataFrame
        Box location information.
    size : int
        Box size [px] on frame image.
    frm : ndarray[uint8]
        Frame image.
        Shape is (height, width, channel).

    Returns
    -------
    imgs : ndarray[uint8]
        Box images.
        Shape is (box_num, height, width, channel).
    """

    box_imgs = np.empty((len(box_info), 64, 64, 3), dtype=np.uint8)
    for i, b in box_info.iterrows():
        box_imgs[i] = cv2.resize(frm[round(b["y"] - size / 2):round(b["y"] + size / 2), round(b["x"] - size / 2):round(b["x"] + size / 2)], (64, 64))

    return box_imgs

def get_result_dir(dir_name: str | None) -> str:
    if dir_name is None:
        dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return path.join(path.dirname(__file__), "../result/", dir_name)

def load_param(file: str) -> dict[str, Any] | list[Any]:
    with open(file) as f:
        match path.splitext(file)[1]:
            case ".json":
                return json.load(f)
            case ".yaml":
                return yaml.safe_load(f)
            case _:
                raise Exception("only json and yaml are supported")

def load_test_result(result_dir: str, ver: int = 0) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Param]]:
    return tuple(np.load(path.join(result_dir, f"version_{ver}/", "test_outputs.npz")).values()), load_param(path.join(result_dir, f"version_{ver}/", "hparams.yaml"))

def look_up_key_from_val(src: dict, val: Any) -> Any:
    for k, v in src.items():
        if v == val:
            return k

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

def sec2str(ts: float) -> str:
    return f"{int(ts // 3600):02d}:{int(ts % 3600 // 60):02d}:{ts % 60:05.2f}"

def use_color_jitter(brightness: float, contrast: float, hue: float, saturation: float) -> T.ColorJitter:
    return T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

def use_flip_and_rot() -> T.Compose:
    return T.Compose((
        T.RandomApply((T.Lambda(lambda img: TF.rotate(img, 90)), )),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip()
    ))
