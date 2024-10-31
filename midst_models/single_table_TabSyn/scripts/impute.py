import json
import os
import warnings

import numpy as np
import torch

from midst_models.single_table_TabSyn.src import load_config
from midst_models.single_table_TabSyn.src.data import preprocess
from midst_models.single_table_TabSyn.src.tabsyn.model.modules import (
    MLPDiffusion,
    Model,
)
from midst_models.single_table_TabSyn.src.tabsyn.tabsyn.model.vae import (
    Decoder_model,
    Encoder_model,
    Model_VAE,
)
from midst_models.single_table_TabSyn.src.tabsyn.utils import (
    recover_data,
    split_num_cat_target,
)

warnings.filterwarnings("ignore")

# class_labels = None


## One denoising step from t to t-1
def step(net, num_steps, i, t_cur, t_next, x_next, S_churn, S_min, S_max, S_noise):
    x_cur = x_next
    # Increase noise temporarily.
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(x_cur)
    # Euler step.

    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def impute(dataname, processed_data_dir, info_path, model_path, impute_path, device):
    # dataname = args.dataname
    # device = args.device
    # epoch = args.epoch
    # mask_cols = [0]

    data_dir = os.path.join(processed_data_dir, dataname)

    # get model config
    config_path = os.path.join("src/baselines/tabsyn/configs", f"{dataname}.toml")
    raw_config = load_config(config_path)

    # number of resampling trials in imputation
    num_trials = raw_config["impute"]["num_trials"]

    # determine imputation parameters
    SIGMA_MIN = raw_config["impute"]["SIGMA_MIN"]
    SIGMA_MAX = raw_config["impute"]["SIGMA_MAX"]
    rho = raw_config["impute"]["rho"]
    S_churn = raw_config["impute"]["S_churn"]
    S_min = raw_config["impute"]["S_min"]
    S_max = raw_config["impute"]["S_max"]
    S_noise = raw_config["impute"]["S_noise"]

    # get model params
    d_token = raw_config["model_params"]["d_token"]
    # token_bias = True
    n_head = raw_config["model_params"]["n_head"]
    factor = raw_config["model_params"]["factor"]
    num_layers = raw_config["model_params"]["num_layers"]

    # get info about dataset columns
    with open(info_path, "r") as f:
        info = json.load(f)

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    task_type = info["task_type"]

    # get trained VAE checkpoint
    ckpt_dir = os.path.join(model_path, dataname, "vae")
    model_save_path = os.path.join(ckpt_dir, "model.pt")
    encoder_save_path = os.path.join(ckpt_dir, "encoder.pt")
    decoder_save_path = os.path.join(ckpt_dir, "decoder.pt")

    for trial in range(num_trials):
        print(f"Trial {trial} started!")
        # prepare data
        X_num, X_cat, categories, d_numerical = preprocess(
            data_dir,
            task_type=info["task_type"],
            transforms=raw_config["transforms"],
        )

        X_train_num, X_test_num = X_num
        X_train_cat, X_test_cat = X_cat

        X_train_num, X_test_num = (
            torch.tensor(X_train_num).float(),
            torch.tensor(X_test_num).float(),
        )
        X_train_cat, X_test_cat = torch.tensor(X_train_cat), torch.tensor(X_test_cat)

        # mask target column
        mask_idx = 0
        if task_type == "binclass":
            unique_values, counts = np.unique(
                X_train_cat[:, mask_idx], return_counts=True
            )
            sampled_cat = np.random.choice(
                unique_values, size=1, p=counts / counts.sum()
            )

            # Replacing the target column with the sampled class
            X_train_cat[:, mask_idx] = torch.tensor(
                unique_values[sampled_cat[0]]
            ).long()
            X_test_cat[:, mask_idx] = torch.tensor(unique_values[sampled_cat[0]]).long()

        else:
            avg = X_train_num[:, mask_idx].mean(0)

            X_train_num[:, mask_idx] = avg
            X_test_num[:, mask_idx] = avg

        # load trained VAE from checkpoint
        model = Model_VAE(
            num_layers,
            d_numerical,
            categories,
            d_token,
            n_head=n_head,
            factor=factor,
            bias=True,
        )
        model = model.to(device)

        model.load_state_dict(torch.load(f"{ckpt_dir}/model.pt"))

        pre_encoder = Encoder_model(
            num_layers, d_numerical, categories, d_token, n_head=n_head, factor=factor
        ).to(device)
        pre_decoder = Decoder_model(
            num_layers, d_numerical, categories, d_token, n_head=n_head, factor=factor
        )

        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        X_train_num = X_train_num.to(device)
        X_train_cat = X_train_cat.to(device)

        X_test_num = X_test_num.to(device)
        X_test_cat = X_test_cat.to(device)

        # embed masked and unmasked data together
        x = pre_encoder(X_test_num, X_test_cat).detach().cpu().numpy()
        embedding_save_path = os.path.join(model_path, dataname, "vae", "train_z.npy")

        # load and normalized embedded data
        train_z = torch.tensor(np.load(embedding_save_path)).float()
        train_z = train_z[:, 1:, :]

        B, num_tokens, token_dim = train_z.size()
        in_dim = num_tokens * token_dim

        train_z = train_z.view(B, in_dim)
        mean, std = train_z.mean(0), train_z.std(0)

        x = torch.tensor(x[:, 1:, :]).view(-1, in_dim)

        x = ((x - mean) / 2).to(device)

        # load trained diffusion model from checkpoint
        denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
        model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)
        model.load_state_dict(
            torch.load(os.path.join(model_path, dataname, "model.pt"))
        )

        # Define the masking area
        mask_idx = np.array([0])
        if task_type == "binclass":
            mask_idx += d_numerical

        mask_list = [list(range(i * token_dim, (i + 1) * token_dim)) for i in mask_idx]
        mask = torch.zeros(num_tokens * token_dim, dtype=torch.bool)
        mask[mask_list] = True

        ###########################
        # Diffusion Denoising
        ###########################

        # configs setup

        num_steps = raw_config["impute"]["num_steps"]
        N = raw_config["impute"]["N"]
        net = model.denoise_fn_D

        num_samples, dim = x.shape[0], x.shape[1]
        x_t = torch.randn([num_samples, dim], device="cuda")

        step_indices = torch.arange(num_steps, dtype=torch.float32, device=x_t.device)

        sigma_min = max(SIGMA_MIN, net.sigma_min)
        sigma_max = min(SIGMA_MAX, net.sigma_max)

        # setup diffusion steps
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        mask = mask.to(torch.int).to(device)
        x_t = x_t.to(torch.float32) * t_steps[0]

        # reverse diffusion for imputation
        with torch.no_grad():
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                if i < num_steps - 1:
                    for j in range(N):
                        n_curr = torch.randn_like(x).to(device) * t_cur
                        n_prev = torch.randn_like(x).to(device) * t_next

                        x_known_t_prev = x + n_prev
                        x_unknown_t_prev = step(
                            net,
                            num_steps,
                            i,
                            t_cur,
                            t_next,
                            x_t,
                            S_churn,
                            S_min,
                            S_max,
                            S_noise,
                        )

                        x_t_prev = x_known_t_prev * (1 - mask) + x_unknown_t_prev * mask

                        n = torch.randn_like(x) * (t_cur.pow(2) - t_next.pow(2)).sqrt()

                        if j == N - 1:
                            x_t = x_t_prev  # turn to x_{t-1}
                        else:
                            x_t = x_t_prev + n  # new x_t

        # get detokenizer
        _, _, _, _, num_inverse, cat_inverse = preprocess(
            data_dir,
            task_type=info["task_type"],
            transforms=raw_config["transforms"],
            inverse=True,
        )
        x_t = x_t * 2 + mean.to(device)

        info["pre_decoder"] = pre_decoder
        info["token_dim"] = token_dim

        syn_data = x_t.float().cpu().numpy()
        syn_num, syn_cat, syn_target = split_num_cat_target(
            syn_data, info, num_inverse, cat_inverse, device
        )

        # impute data
        syn_df = recover_data(syn_num, syn_cat, syn_target, info)

        idx_name_mapping = info["idx_name_mapping"]
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        syn_df.rename(columns=idx_name_mapping, inplace=True)

        save_dir = os.path.join(impute_path, dataname)
        os.makedirs(save_dir) if not os.path.exists(save_dir) else None

        syn_df.to_csv(os.path.join(save_dir, f"{trial}.csv"), index=False)
