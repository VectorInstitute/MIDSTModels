import os
import logging
import torch
from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader

from midst_models.single_table_TabSyn.src.data import preprocess, TabularDataset
from midst_models.single_table_TabSyn.src.tabsyn.pipeline import TabSyn
from midst_models.single_table_TabSyn.src import load_config

logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="train", version_base="1.1")
def main(cfg: DictConfig):
    logger.info(f"running experiment: {cfg.exp_name}")
    
    data_dir = cfg.data_dir
    data_name = cfg.data_name
    exp_name = cfg.exp_name
    ref_data_path = cfg.main_data_path
    processed_data_dir = os.path.join(data_dir, "processed_data", data_name)
    logger.info(f"Data directory: {data_dir}")

    # Load configuration for the task
    config_path = os.path.join(data_dir, f"{data_name}.toml")
    logger.info(f"Loading configuration from: {config_path}")
    raw_config = load_config(config_path)
    
    save_dir = os.path.join(data_dir, "workspace", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    vae_dir = os.path.join(save_dir, "vae")
    os.makedirs(vae_dir, exist_ok=True)

    # Preprocess the data
    X_num, X_cat, categories, d_numerical = preprocess(
        processed_data_dir,
        ref_dataset_path=ref_data_path,
        transforms=raw_config["transforms"],
        task_type=raw_config["task_type"],
    )

    # Separate train and test data
    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    # Convert data to float tensors
    X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_test_cat = torch.tensor(X_train_cat), torch.tensor(X_test_cat)

    # Create dataset module
    train_data = TabularDataset(X_train_num.float(), X_train_cat)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)

    # Create train DataLoader
    train_loader = DataLoader(
        train_data,
        batch_size=raw_config["train"]["vae"]["batch_size"],
        shuffle=True,
        num_workers=raw_config["train"]["vae"]["num_dataset_workers"],
    )

    # Initialize the TabSyn model
    tabsyn = TabSyn(
        train_loader,
        X_test_num,
        X_test_cat,
        num_numerical_features=d_numerical,
        num_classes=categories,
        device=device
    )

    # Instantiate and train the VAE model
    tabsyn.instantiate_vae(**raw_config["model_params"], optim_params=raw_config["train"]["optim"]["vae"])

    tabsyn.train_vae(
        **raw_config["loss_params"],
        num_epochs=raw_config["train"]["vae"]["num_epochs"],
        save_path=vae_dir,
    )

    # Save VAE embeddings
    tabsyn.save_vae_embeddings(X_train_num, X_train_cat, vae_ckpt_dir=vae_dir)

    # Load latent embeddings from VAE
    train_z, _ = tabsyn.load_latent_embeddings(vae_dir)

    # Normalize embeddings
    mean, std = train_z.mean(0), train_z.std(0)
    latent_train_data = (train_z - mean) / 2

    # Create DataLoader for latent space
    latent_train_loader = DataLoader(
        latent_train_data,
        batch_size=raw_config["train"]["diffusion"]["batch_size"],
        shuffle=True,
        num_workers=raw_config["train"]["diffusion"]["num_dataset_workers"],
    )

    # Instantiate and train the Diffusion model
    tabsyn.instantiate_diffusion(
        in_dim=train_z.shape[1], 
        hid_dim=train_z.shape[1], 
        optim_params=raw_config["train"]["optim"]["diffusion"]
    )

    tabsyn.train_diffusion(
        latent_train_loader,
        num_epochs=raw_config["train"]["diffusion"]["num_epochs"],
        ckpt_path=save_dir,
    )

if __name__ == "__main__":
    main()
