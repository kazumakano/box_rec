import os
import os.path as path
from glob import glob
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import script.data as D
import script.utility as util

BOX_INFO_FILE = "/mnt/bigdata/00_students/kazuma_nis/box_rec/config/box.csv"
SRC_DIR = "/mnt/nict-dbp-2/20231123work/scripts/imgs/"
TGT_DIR = "/mnt/bigdata/01_projects/box_dataset/"

def _back_states(box_num: int) -> None:
    st.session_state["box_idx"] -= 1
    if st.session_state["box_idx"] == -1:
        st.session_state["img_idx"] -= 1
        st.session_state["box_idx"] = box_num - 1

def _next_states(box_num: int) -> None:
    st.session_state["box_idx"] += 1
    if st.session_state["box_idx"] == box_num:
        st.session_state["img_idx"] += 1
        st.session_state["box_idx"] = 0

def _reset_states() -> None:
    st.session_state["img_idx"] = 0
    st.session_state["box_idx"] = 0

def _check_exist(box_num: int) -> None:
    while True:
        if len(glob(path.join(TGT_DIR, f"*_{st.session_state['img_idx']}_{st.session_state['box_idx']}_*.jpg"))) > 0:
            _next_states(box_num)
        else:
            break

def _save_box_img(box_num: int, img: np.ndarray, label: str, usr_name: str):
    file_name = f"{usr_name}_{st.session_state['img_idx']}_{st.session_state['box_idx']}_{label}.jpg"
    if cv2.imwrite(path.join(TGT_DIR, file_name), img):
        st.success(f"saved to {file_name}")
    else:
        st.error("failed to save image")

    _next_states(box_num)

def _label_btn(box_num: int, img: np.ndarray, label: str, usr_name: str) -> None:
    st.button(label, on_click=lambda: _save_box_img(box_num, img, label, usr_name))

def _undo(box_num: int, usr_name: str) -> None:
    _back_states(box_num)

    files = glob(path.join(TGT_DIR, f"{usr_name}_{st.session_state['img_idx']}_{st.session_state['box_idx']}_*.jpg"))
    if len(files) > 0:
        os.remove(files[0])
        st.info(f"deleted {path.basename(files[0])}")

def render() -> None:
    if len(st.session_state) == 0:
        _reset_states()

    st.title("Dataset Creator")

    box_info = pd.read_csv(BOX_INFO_FILE)
    usr_name = st.text_input("input your name")

    if usr_name != "":
        files = glob(path.join(SRC_DIR, f"????-??-??U-?????-{st.session_state['img_idx']:04d}.jpg"))
        if len(files) == 0:
            st.write(f"image {st.session_state['img_idx']} was not found")
        else:
            for f in files:
                _check_exist(len(box_info))

                frm = cv2.imread(f)
                box_img = util.extract_box(box_info, frm)[st.session_state["box_idx"]]

                st.image(cv2.rectangle(frm, (box_info.loc[st.session_state["box_idx"], "l"], box_info.loc[st.session_state["box_idx"], "t"]), (box_info.loc[st.session_state["box_idx"], "r"], box_info.loc[st.session_state["box_idx"], "b"]), (0, 255, 0), thickness=4), channels="BGR")
                st.write(f"showing box {st.session_state['box_idx']} on image {st.session_state['img_idx']}")
                st.image(box_img, width=256, channels="BGR")

                for i, c in enumerate(st.columns(len(D.USAGE))):
                    with c:
                        _label_btn(len(box_info), box_img, tuple(D.USAGE.keys())[i], usr_name)

                cols = st.columns(8)
                with cols[0]:
                    st.button("undo", on_click=lambda: _undo(len(box_info), usr_name))
                with cols[1]:
                    st.button("skip", on_click=lambda: _next_states(len(box_info)))

        new_img_idx = st.text_input("img_idx")
        if new_img_idx != "":
            st.session_state["img_idx"] = int(new_img_idx)
            st.session_state["box_idx"] = 0

if __name__ == "__main__":
    render()
