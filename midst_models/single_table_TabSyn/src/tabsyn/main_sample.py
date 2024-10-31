import argparse
import time
import warnings

import torch

from midst_models.single_table_TabSyn.src.tabsyn.model.gaussian_diffusion import Model
from midst_models.single_table_TabSyn.src.tabsyn.model.modules import MLPDiffusion
from midst_models.single_table_TabSyn.src.tabsyn.utils import (
    get_input_generate,
    recover_data,
    split_num_cat_target,
)

warnings.filterwarnings("ignore")


def main(args):
    # dataname = args.dataname
    device = args.device
    # steps = args.steps
    save_path = args.save_path

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1]

    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)

    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)

    model.load_state_dict(torch.load(f"{ckpt_path}/model.pt"))

    """
        Generating samples
    """
    start_time = time.time()

    num_samples = train_z.shape[0]
    sample_dim = in_dim

    x_next = model.sample(num_samples, sample_dim)
    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(
        syn_data, info, num_inverse, cat_inverse, args.device
    )

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info["idx_name_mapping"]
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns=idx_name_mapping, inplace=True)
    syn_df.to_csv(save_path, index=False)

    end_time = time.time()
    print("Time:", end_time - start_time)

    print("Saving sampled data to {}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation")

    parser.add_argument(
        "--dataname", type=str, default="adult", help="Name of dataset."
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch.")
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of function evaluations."
    )

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"
