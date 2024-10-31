import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pd.options.mode.chained_assignment = None


def eval_dcr(syn_path, real_path, test_path, info_path):
    with open(info_path, "r") as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)
    test_data = pd.read_csv(test_path)

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    task_type = info["task_type"]
    if task_type == "regression":
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    num_ranges = []

    real_data.columns = list(np.arange(len(real_data.columns)))
    syn_data.columns = list(np.arange(len(real_data.columns)))
    test_data.columns = list(np.arange(len(real_data.columns)))
    for i in num_col_idx:
        num_ranges.append(real_data[i].max() - real_data[i].min())

    num_ranges = np.array(num_ranges)

    num_real_data = real_data[num_col_idx]
    cat_real_data = real_data[cat_col_idx]
    num_syn_data = syn_data[num_col_idx]
    cat_syn_data = syn_data[cat_col_idx]
    num_test_data = test_data[num_col_idx]
    cat_test_data = test_data[cat_col_idx]

    num_real_data_np = num_real_data.to_numpy()
    cat_real_data_np = cat_real_data.to_numpy().astype("str")
    num_syn_data_np = num_syn_data.to_numpy()
    cat_syn_data_np = cat_syn_data.to_numpy().astype("str")
    num_test_data_np = num_test_data.to_numpy()
    cat_test_data_np = cat_test_data.to_numpy().astype("str")

    if len(cat_col_idx) != 0:
        encoder = OneHotEncoder()
        encoder.fit(cat_real_data_np)

        cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
        cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()
        cat_test_data_oh = encoder.transform(cat_test_data_np).toarray()

        num_real_data_np = num_real_data_np / num_ranges
        num_syn_data_np = num_syn_data_np / num_ranges
        num_test_data_np = num_test_data_np / num_ranges

        real_data_np = np.concatenate([num_real_data_np, cat_real_data_oh], axis=1)
        syn_data_np = np.concatenate([num_syn_data_np, cat_syn_data_oh], axis=1)
        test_data_np = np.concatenate([num_test_data_np, cat_test_data_oh], axis=1)
    else:
        real_data_np = num_real_data_np
        syn_data_np = num_syn_data_np
        test_data_np = num_test_data_np

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    real_data_th = torch.tensor(real_data_np).to(device)
    syn_data_th = torch.tensor(syn_data_np).to(device)
    test_data_th = torch.tensor(test_data_np).to(device)

    dcrs_real = []
    dcrs_test = []
    batch_size = 100

    for i in range((syn_data_th.shape[0] // batch_size) + 1):
        if i != (syn_data_th.shape[0] // batch_size):
            batch_syn_data_th = syn_data_th[i * batch_size : (i + 1) * batch_size]
        else:
            batch_syn_data_th = syn_data_th[i * batch_size :]

        dcr_real = (
            (batch_syn_data_th[:, None] - real_data_th)
            .abs()
            .sum(dim=2)
            .min(dim=1)
            .values
        )
        dcr_test = (
            (batch_syn_data_th[:, None] - test_data_th)
            .abs()
            .sum(dim=2)
            .min(dim=1)
            .values
        )
        dcrs_real.append(dcr_real)
        dcrs_test.append(dcr_test)

    dcrs_real = torch.cat(dcrs_real)
    dcrs_test = torch.cat(dcrs_test)

    score = (dcrs_real < dcrs_test).nonzero().shape[0] / dcrs_real.shape[0]
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="adult")
    parser.add_argument("--model", type=str, default="model")
    parser.add_argument(
        "--path", type=str, default=None, help="The file path of the synthetic data"
    )

    args = parser.parse_args()

    dataname = args.dataname
    model = args.model

    if not args.path:
        syn_path = f"/projects/aieng/diffusion_bootcamp/data/tabular/synthetic_data/{dataname}/{model}.csv"
    else:
        syn_path = args.path
    real_path = f"/projects/aieng/diffusion_bootcamp/data/tabular/processed_data/{dataname}/train.csv"
    test_path = f"/projects/aieng/diffusion_bootcamp/data/tabular/processed_data/{dataname}/test.csv"

    info_path = f"/projects/aieng/diffusion_bootcamp/data/tabular/processed_data/{dataname}/info.json"

    save_path = f"/projects/aieng/diffusion_bootcamp/data/tabular/synthetic_data/{dataname}/{model}_dcr.txt"
    dcr_score = eval_dcr(syn_path, real_path, test_path, info_path)

    print("DCR Score, a value closer to 0.5 is better")
    print(f"{dataname}-{model}, DCR Score = {dcr_score}")

    with open(save_path, "w") as f:
        f.write("DCR Score, a value closer to 0.5 is better\n")
        f.write(f"{dataname}-{model}, DCR Score = {dcr_score}")
