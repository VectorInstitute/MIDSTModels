import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from synthcity.metrics import eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pd.options.mode.chained_assignment = None


def eval_quality(syn_path, real_path, info_path):
    with open(info_path, "r") as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)

    """ Special treatment for default dataset"""

    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]
    if info["task_type"] == "regression":
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    num_real_data = real_data[num_col_idx]
    cat_real_data = real_data[cat_col_idx]

    num_real_data_np = num_real_data.to_numpy()
    cat_real_data_np = cat_real_data.to_numpy().astype("str")

    num_syn_data = syn_data[num_col_idx]
    cat_syn_data = syn_data[cat_col_idx]

    num_syn_data_np = num_syn_data.to_numpy()
    cat_syn_data_np = cat_syn_data.to_numpy().astype("str")
    if len(cat_col_idx) != 0:
        encoder = OneHotEncoder()
        encoder.fit(cat_real_data_np)

        cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
        cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()

        le_real_data = pd.DataFrame(
            np.concatenate((num_real_data_np, cat_real_data_oh), axis=1)
        ).astype(float)

        le_syn_data = pd.DataFrame(
            np.concatenate((num_syn_data_np, cat_syn_data_oh), axis=1)
        ).astype(float)
    else:
        le_real_data = pd.DataFrame(num_real_data_np).astype(float)
        le_syn_data = pd.DataFrame(num_syn_data_np).astype(float)

    np.set_printoptions(precision=4)

    print("=========== All Features ===========")
    print("Data shape: ", le_syn_data.shape)

    X_syn_loader = GenericDataLoader(le_syn_data)
    X_real_loader = GenericDataLoader(le_real_data)

    quality_evaluator = eval_statistical.AlphaPrecision()
    qual_res = quality_evaluator.evaluate(X_real_loader, X_syn_loader)
    qual_res = {
        k: v for (k, v) in qual_res.items() if "naive" in k
    }  # use the naive implementation of AlphaPrecision

    alpha_precision_all = qual_res["delta_precision_alpha_naive"]
    beta_recall_all = qual_res["delta_coverage_beta_naive"]

    return alpha_precision_all, beta_recall_all


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

    info_path = f"/projects/aieng/diffusion_bootcamp/data/tabular/processed_data/{dataname}/info.json"

    save_dir = f"/projects/aieng/diffusion_bootcamp/data/tabular/synthetic_data/{dataname}/{model}_quality.txt"

    alpha_precision_all, beta_recall_all = eval_quality(syn_path, real_path, info_path)

    print("Alpha precision:", alpha_precision_all)
    print("Beta recall:", beta_recall_all)

    with open(save_dir, "w") as f:
        f.write(f"Alpha precision: {alpha_precision_all}\n")
        f.write(f"Beta recall: {beta_recall_all}\n")
