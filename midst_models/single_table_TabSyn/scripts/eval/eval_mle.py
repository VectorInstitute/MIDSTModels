import argparse
import json
import os
import sys
import warnings

import pandas as pd

from midst_models.single_table_TabSyn.scripts.eval.mle.mle import get_evaluator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")


def eval_mle(train_path, test_path, info_path):
    train = pd.read_csv(train_path).to_numpy()
    test = pd.read_csv(test_path).to_numpy()

    with open(info_path, "r") as f:
        info = json.load(f)

    task_type = info["task_type"]

    evaluator = get_evaluator(task_type)

    if task_type == "regression":
        best_r2_scores, best_rmse_scores = evaluator(train, test, info)

        overall_scores = {}
        for score_name in ["best_r2_scores", "best_rmse_scores"]:
            overall_scores[score_name] = {}

            scores = eval(score_name)
            for method in scores:
                name = method["name"]
                method.pop("name")
                overall_scores[score_name][name] = method

    else:
        (
            best_f1_scores,
            best_weighted_scores,
            best_auroc_scores,
            best_acc_scores,
            best_avg_scores,
        ) = evaluator(train, test, info)

        overall_scores = {}
        for score_name in [
            "best_f1_scores",
            "best_weighted_scores",
            "best_auroc_scores",
            "best_acc_scores",
            "best_avg_scores",
        ]:
            overall_scores[score_name] = {}

            scores = eval(score_name)
            for method in scores:
                name = method["name"]
                method.pop("name")
                overall_scores[score_name][name] = method
    return overall_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="adult")
    parser.add_argument("--model", type=str, default="real")
    parser.add_argument(
        "--path", type=str, default=None, help="The file path of the synthetic data"
    )

    args = parser.parse_args()

    dataname = args.dataname
    model = args.model

    if not args.path:
        syn_path = f"/projects/aieng/diffusion_bootcamp/data/tabular/synthetic_data/{dataname}/{model}.csv"
    else:
        train_path = args.path
    real_path = f"/projects/aieng/diffusion_bootcamp/data/tabular/processed_data/{dataname}/test.csv"

    info_path = f"/projects/aieng/diffusion_bootcamp/data/tabular/processed_data/{dataname}/info.json"

    save_path = f"/projects/aieng/diffusion_bootcamp/data/tabular/synthetic_data/{dataname}/{model}_mle.json"

    overall_scores = eval_mle(syn_path, real_path, info_path)

    print("Saving scores to ", save_path)
    with open(save_path, "w") as json_file:
        json.dump(overall_scores, json_file, indent=4, separators=(", ", ": "))
