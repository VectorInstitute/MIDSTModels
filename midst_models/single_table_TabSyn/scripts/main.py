import argparse

import torch
from src.tabddpm.main_sample import main as sample_tabddpm  # noqa: F401
from src.tabddpm.main_train import main as train_tabddpm  # noqa: F401
from src.tabsyn.main_sample import main as sample_tabsyn  # noqa: F401
from src.tabsyn.main_train import main as train_tabsyn  # noqa: F401
from src.tabsyn.main_vae import main as train_vae  # noqa: F401


def execute_function(method, mode):
    if method == "vae":
        mode = "train"

    main_fn = eval(f"{mode}_{method}")

    return main_fn


def get_args():
    parser = argparse.ArgumentParser(description="Pipeline")

    # General configs
    parser.add_argument(
        "--dataname", type=str, default="adult", help="Name of dataset."
    )
    parser.add_argument(
        "--mode", type=str, default="train", help="Mode: train or sample."
    )
    parser.add_argument(
        "--method", type=str, default="tabsyn", help="Method: tabsyn or baseline."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")

    # configs for TabDDPM
    parser.add_argument(
        "--ddim", action="store_true", default=False, help="Whether use DDIM sampler"
    )

    # configs for traing TabSyn's VAE
    parser.add_argument("--max_beta", type=float, default=1e-2, help="Maximum beta")
    parser.add_argument("--min_beta", type=float, default=1e-5, help="Minimum beta.")
    parser.add_argument("--lambd", type=float, default=0.7, help="Batch size.")

    # configs for sampling
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save synthetic data."
    )
    parser.add_argument("--steps", type=int, default=50, help="NFEs.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    if torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"

    if not args.save_path:
        args.save_path = f"/projects/aieng/diffussion_bootcamp/data/synthetic_tabular/{args.dataname}/{args.method}.csv"
    main_fn = execute_function(args.method, args.mode)

    main_fn(args)
