import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from midst_models.single_table_TabSyn.src.tabsyn.model.modules import (
    MLPDiffusion,
    Model,
)
from midst_models.single_table_TabSyn.src.tabsyn.model.utils import sample
from midst_models.single_table_TabSyn.src.tabsyn.model.vae import (
    Decoder_model,
    Encoder_model,
    Model_VAE,
)
from midst_models.single_table_TabSyn.src.tabsyn.utils import (
    recover_data,
    split_num_cat_target,
)


class TabSyn:
    def __init__(
        self,
        train_loader,
        X_test_num,
        X_test_cat,
        num_numerical_features,
        num_classes,
        device=None,
    ):
        """Train, sample, load and save TabSyn model."""
        self.train_loader = train_loader
        self.X_test_num = X_test_num
        self.X_test_cat = X_test_cat
        self.d_numerical = num_numerical_features
        self.categories = num_classes
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def instantiate_vae(self, n_head, factor, num_layers, d_token, optim_params):
        """Construct VAE model and its optimizer and lr scheduler."""
        # construct vae model
        self.vae_model, self.pre_encoder, self.pre_decoder = self.__get_vae_model(
            n_head, factor, num_layers, d_token
        )
        # construct vae optimizer and scheduler
        if optim_params is not None:
            self.vae_optimizer, self.vae_scheduler = self.__load_optim(
                self.vae_model, **optim_params
            )
        print("Successfully instantiated VAE model.")

    def instantiate_diffusion(self, in_dim, hid_dim, optim_params):
        """Construct Diffusion model and its optimizer and lr scheduler."""
        # load diffusion model
        self.dif_model = self.__get_diffusion_model(in_dim=in_dim, hid_dim=hid_dim)
        # load optimizer and scheduler
        if optim_params is not None:
            self.dif_optimizer, self.dif_scheduler = self.__load_optim(
                self.dif_model, **optim_params
            )
        print("Successfully instantiated diffusion model.")

    def __get_vae_model(self, n_head, factor, num_layers, d_token):
        model = Model_VAE(
            num_layers,
            self.d_numerical,
            self.categories,
            d_token,
            n_head=n_head,
            factor=factor,
            bias=True,
        )
        model = model.to(self.device)

        pre_encoder = Encoder_model(
            num_layers,
            self.d_numerical,
            self.categories,
            d_token,
            n_head=n_head,
            factor=factor,
        ).to(self.device)
        pre_decoder = Decoder_model(
            num_layers,
            self.d_numerical,
            self.categories,
            d_token,
            n_head=n_head,
            factor=factor,
        ).to(self.device)

        pre_encoder.eval()
        pre_decoder.eval()

        return model, pre_encoder, pre_decoder

    def __get_diffusion_model(self, in_dim, hid_dim):
        denoise_fn = MLPDiffusion(in_dim, 1024).to(self.device)
        print(denoise_fn)

        num_params = sum(p.numel() for p in denoise_fn.parameters())
        print("The number of parameters:", num_params)

        model = Model(denoise_fn=denoise_fn, hid_dim=hid_dim).to(self.device)
        return model

    def __load_optim(self, model, lr, weight_decay, factor, patience):
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, verbose=True
        )
        return optimizer, scheduler

    def train_vae(self, max_beta, min_beta, lambd, num_epochs, save_path):
        # determine model save paths
        model_save_path = os.path.join(save_path, "model.pt")
        encoder_save_path = os.path.join(save_path, "encoder.pt")
        decoder_save_path = os.path.join(save_path, "decoder.pt")

        # set initial state
        current_lr = self.vae_optimizer.param_groups[0]["lr"]
        patience = 0
        best_train_loss = float("inf")
        beta = max_beta

        # training loop
        start_time = time.time()
        for epoch in range(num_epochs):
            pbar = tqdm(self.train_loader, total=len(self.train_loader))
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

            curr_loss_multi = 0.0
            curr_loss_gauss = 0.0
            curr_loss_kl = 0.0

            curr_count = 0

            for batch_num, batch_cat in pbar:
                self.vae_model.train()
                self.vae_optimizer.zero_grad()

                batch_num = batch_num.to(self.device)
                batch_cat = batch_cat.to(self.device)

                Recon_X_num, Recon_X_cat, mu_z, std_z = self.vae_model(
                    batch_num, batch_cat
                )

                loss_mse, loss_ce, loss_kld, train_acc = self.compute_loss(
                    batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z
                )

                loss = loss_mse + loss_ce + beta * loss_kld
                loss.backward()
                self.vae_optimizer.step()

                batch_length = batch_num.shape[0]
                curr_count += batch_length
                curr_loss_multi += loss_ce.item() * batch_length
                curr_loss_gauss += loss_mse.item() * batch_length
                curr_loss_kl += loss_kld.item() * batch_length

            num_loss = curr_loss_gauss / curr_count
            cat_loss = curr_loss_multi / curr_count
            kl_loss = curr_loss_kl / curr_count

            """
                Evaluation
            """
            self.vae_model.eval()
            with torch.no_grad():
                Recon_X_num, Recon_X_cat, mu_z, std_z = self.vae_model(
                    self.X_test_num, self.X_test_cat
                )

                val_mse_loss, val_ce_loss, val_kl_loss, val_acc = self.compute_loss(
                    self.X_test_num,
                    self.X_test_cat,
                    Recon_X_num,
                    Recon_X_cat,
                    mu_z,
                    std_z,
                )
                val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()

                self.vae_scheduler.step(val_loss)
                new_lr = self.vae_optimizer.param_groups[0]["lr"]

                if new_lr != current_lr:
                    current_lr = new_lr
                    print(f"Learning rate updated: {current_lr}")

                train_loss = val_loss
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    patience = 0
                    torch.save(self.vae_model.state_dict(), model_save_path)
                else:
                    patience += 1
                    if patience == 10:
                        if beta > min_beta:
                            beta = beta * lambd

            # print("epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Train ACC:{:6f}".format(epoch, beta, num_loss, cat_loss, kl_loss, train_acc.item()))
            print(
                "epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}".format(
                    epoch,
                    beta,
                    num_loss,
                    cat_loss,
                    kl_loss,
                    val_mse_loss.item(),
                    val_ce_loss.item(),
                    train_acc.item(),
                    val_acc.item(),
                )
            )

        end_time = time.time()
        print("Training time: {:.4f} mins".format((end_time - start_time) / 60))

        # load and save encoder and decoder states
        self.pre_encoder.load_weights(self.vae_model)
        self.pre_decoder.load_weights(self.vae_model)

        torch.save(self.pre_encoder.state_dict(), encoder_save_path)
        torch.save(self.pre_decoder.state_dict(), decoder_save_path)

        print("Successfully trained and saved the VAE model!")

    def compute_loss(self, X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
        ce_loss_fn = nn.CrossEntropyLoss()
        mse_loss = (X_num - Recon_X_num).pow(2).mean()
        ce_loss = 0
        acc = 0
        total_num = 0

        for idx, x_cat in enumerate(Recon_X_cat):
            if x_cat is not None:
                ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
                x_hat = x_cat.argmax(dim=-1)
            acc += (x_hat == X_cat[:, idx]).float().sum()
            total_num += x_hat.shape[0]

        ce_loss /= idx + 1
        acc /= total_num
        # loss = mse_loss + ce_loss

        temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

        loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
        return mse_loss, ce_loss, loss_kld, acc

    def save_vae_embeddings(self, X_train_num, X_train_cat, vae_ckpt_dir):
        # Saving latent embeddings
        with torch.no_grad():
            X_train_num = X_train_num.to(self.device)
            X_train_cat = X_train_cat.to(self.device)

            train_z = self.pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()

            np.save(os.path.join(vae_ckpt_dir, "train_z.npy"), train_z)

            print("Successfully saved pretrained embeddings on disk!")

    def load_latent_embeddings(self, vae_ckpt_dir):
        embedding_save_path = os.path.join(vae_ckpt_dir, "train_z.npy")
        train_z = torch.tensor(np.load(embedding_save_path)).float()

        # flatten embeddings
        train_z = train_z[:, 1:, :]
        B, num_tokens, token_dim = train_z.size()
        in_dim = num_tokens * token_dim

        train_z = train_z.view(B, in_dim)

        return train_z, token_dim
    
    def save_embeddings_attributes(self, vae_ckpt_dir):
        train_z, token_dim = self.load_latent_embeddings(vae_ckpt_dir)
        embedding_att = {
            'token_dim': token_dim,
            'in_dim': train_z.shape[1],
            'hid_dim': train_z.shape[1],
            'num_samples': train_z.shape[0],
            'mean_input_emb': train_z.mean(0),
            'std_input_emb': train_z.std(0)
        }
        pickle.dump(embedding_att, open(os.path.join(vae_ckpt_dir, "train_z_attributes.pkl"), "wb"))
        
    def load_embeddings_attributes(self, vae_ckpt_dir):
        embedding_att = pickle.load(open(os.path.join(vae_ckpt_dir, "train_z_attributes.pkl"), "rb"))
        return embedding_att

    def train_diffusion(self, train_loader, num_epochs, ckpt_path):
        self.dif_model.train()

        best_loss = float("inf")
        patience = 0
        start_time = time.time()
        for epoch in range(num_epochs):
            pbar = tqdm(train_loader, total=len(train_loader))
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

            batch_loss = 0.0
            len_input = 0
            for batch in pbar:
                inputs = batch.float().to(self.device)
                loss = self.dif_model(inputs)

                loss = loss.mean()

                batch_loss += loss.item() * len(inputs)
                len_input += len(inputs)

                self.dif_optimizer.zero_grad()
                loss.backward()
                self.dif_optimizer.step()

                pbar.set_postfix({"Loss": loss.item()})

            curr_loss = batch_loss / len_input
            self.dif_scheduler.step(curr_loss)

            if curr_loss < best_loss:
                best_loss = curr_loss
                patience = 0
                torch.save(
                    self.dif_model.state_dict(), os.path.join(ckpt_path, "model.pt")
                )
            else:
                patience += 1
                if patience == 500:
                    print("Early stopping")
                    break

            if epoch % 1000 == 0:
                torch.save(
                    self.dif_model.state_dict(),
                    os.path.join(ckpt_path, f"model_{epoch}.pt"),
                )

        end_time = time.time()
        print("Time: ", end_time - start_time)

    def load_model_state(self, ckpt_dir, dif_ckpt_name="model.pt"):
        dif_model_save_path = os.path.join(ckpt_dir, dif_ckpt_name)
        vae_model_save_path = os.path.join(ckpt_dir, "vae", "model.pt")
        encoder_save_path = os.path.join(ckpt_dir, "vae", "encoder.pt")
        decoder_save_path = os.path.join(ckpt_dir, "vae", "decoder.pt")

        self.dif_model.load_state_dict(torch.load(dif_model_save_path))
        self.vae_model.load_state_dict(torch.load(vae_model_save_path))
        self.pre_encoder.load_state_dict(torch.load(encoder_save_path))
        self.pre_decoder.load_state_dict(torch.load(decoder_save_path))

        print("Loaded model state from", ckpt_dir)

    def load_model_for_sampling(
        self,
        in_dim,
        hid_dim,
        d_numerical,
        categories,
        ckpt_dir,
        n_head,
        factor,
        num_layers,
        d_token,
    ):
        denoise_fn = MLPDiffusion(in_dim, 1024).to(self.device)
        model = Model(denoise_fn=denoise_fn, hid_dim=hid_dim).to(self.device)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt")))

        pre_decoder = Decoder_model(
            num_layers, d_numerical, categories, d_token, n_head=n_head, factor=factor
        )
        decoder_save_path = os.path.join(ckpt_dir, "vae", "decoder.pt")
        pre_decoder.load_state_dict(torch.load(decoder_save_path))

        self.dif_model = model
        self.pre_decoder = pre_decoder

    def sample(
        self,
        num_samples,
        in_dim,
        mean_input_emb,
        info,
        num_inverse,
        cat_inverse,
        save_path,
    ):
        """
        Generating samples
        """
        self.pre_decoder.cpu()
        info["pre_decoder"] = self.pre_decoder

        mean = mean_input_emb

        start_time = time.time()

        sample_dim = in_dim

        x_next = sample(self.dif_model.denoise_fn_D, num_samples, sample_dim)
        x_next = x_next * 2 + mean.to(self.device)

        syn_data = x_next.float().cpu().numpy()
        syn_num, syn_cat, syn_target = split_num_cat_target(
            syn_data, info, num_inverse, cat_inverse, self.device
        )

        syn_df = recover_data(syn_num, syn_cat, syn_target, info)

        idx_name_mapping = info["idx_name_mapping"]
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        syn_df.rename(columns=idx_name_mapping, inplace=True)
        syn_df.to_csv(save_path, index=False)

        end_time = time.time()
        print("Time:", end_time - start_time)

        self.pre_decoder.to(self.device)

        print("Saving sampled data to {}".format(save_path))
