import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from src import load_config


def eval_impute(dataname, processed_data_dir, impute_path, col=0):
    # set paths
    data_dir = os.path.join(processed_data_dir, dataname)
    real_path = f"{data_dir}/test.csv"

    # get model config
    config_path = os.path.join("src/baselines/tabsyn/configs", f"{dataname}.toml")
    raw_config = load_config(config_path)
    # number of resampling trials in imputation
    num_trials = raw_config["impute"]["num_trials"]

    encoder = OneHotEncoder()

    real_data = pd.read_csv(real_path)
    target_col = real_data.columns[-1]
    real_target = real_data[target_col].to_numpy().reshape(-1, 1)
    real_y = encoder.fit_transform(real_target).toarray()

    syn_y = []
    for i in range(num_trials):
        syn_path = os.path.join(impute_path, dataname, f"{i}.csv")
        syn_data = pd.read_csv(syn_path)
        target = syn_data[target_col].to_numpy().reshape(-1, 1)
        syn_y.append(encoder.transform(target).toarray())

    syn_y = np.stack(syn_y).mean(0)

    micro_f1 = f1_score(real_y.argmax(axis=1), syn_y.argmax(axis=1), average="micro")
    auc = roc_auc_score(real_y, syn_y, average="micro")

    print("Micro-F1:", micro_f1)
    print("AUC:", auc)
