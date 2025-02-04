import pandas as pd

USAGE_ALIAS = {
    "floor": "FLOOR",
    "item": "GOODS",
    "empty_pallet": "EMPTY",
    "hand_pallet": "HAND",
    "other": "OTHER"
}

def export(src_file: str, tgt_file: str) -> None:
    result = pd.read_csv(src_file, header=0, names=("frame_id", "box_no", "pred_name", "pred_prob"))
    result.loc[:, "pred_name"] = result.loc[:, "pred_name"].map(USAGE_ALIAS)
    result.loc[:, "pred_prob"] *= 100
    result.to_csv(tgt_file, index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_file", required=True, help="specify source file", metavar="PATH_TO_SRC_FILE")    # csv format for box_rec
    parser.add_argument("-t", "--tgt_file", required=True, help="specify target file", metavar="PATH_TO_TGT_FILE")    # csv format for asayu
    args = parser.parse_args()

    export(args.src_file, args.tgt_file)
