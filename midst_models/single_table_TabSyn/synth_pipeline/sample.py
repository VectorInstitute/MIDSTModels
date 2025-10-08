import os
import json
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
    data_dir = cfg.data_dir
    exp_name = cfg.exp_name
    data_name = cfg.data_name
    ref_data_path = cfg.main_data_path
    
    exp_dir = os.path.join(data_dir, "workspace", exp_name)
    latent_embeddings_dir = os.path.join(exp_dir, "vae")
    synth_dir = os.path.join(exp_dir, data_name, "_final")
    os.makedirs(synth_dir, exist_ok=True)
    synth_data_path = os.path.join(synth_dir, f"{data_name}_synthetic.csv")

    config_path = os.path.join(data_dir, f"{data_name}.toml")
    logger.info(f"Loading configuration from: {config_path}")
    raw_config = load_config(config_path)
    
    processed_data_dir = os.path.join(data_dir, "processed_data", data_name)
    info_file_path = os.path.join(processed_data_dir, "info.json")
    
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
    
    tabsyn = TabSyn(
        train_loader,
        X_test_num, 
        X_test_cat,
        num_numerical_features = d_numerical,
        num_classes = categories,
        device = device
    )

    # Instantiate VAE
    tabsyn.instantiate_vae(**raw_config["model_params"], optim_params=None)

    # Load latent embeddings of input data
    train_z, token_dim = tabsyn.load_latent_embeddings(latent_embeddings_dir)
    logger.info(f"Loaded latent embeddings with token dim: {token_dim}")

    # Instantiate diffusion model
    tabsyn.instantiate_diffusion(in_dim=train_z.shape[1], hid_dim=train_z.shape[1], optim_params=None)
    logger.info("Instantiated diffusion model")

    # Load state from checkpoint
    tabsyn.load_model_state(ckpt_dir=exp_dir, dif_ckpt_name="model.pt")
    logger.info(f"Loaded model state from {exp_dir}")
    
    with open(info_file_path, "r") as file:
        data_info = json.load(file)     
    data_info["token_dim"] = token_dim
    logger.info(f"Loaded data info from {info_file_path}")

    # Get inverse tokenizers
    _, _, categories, d_numerical, num_inverse, cat_inverse = preprocess(
        processed_data_dir,
        ref_dataset_path=ref_data_path,
        transforms=raw_config["transforms"],
        task_type=raw_config["task_type"],
        inverse=True
    )
    logger.info(f"Retrieved inverse tokenizers for {data_name}")

    # Sample data
    num_samples = train_z.shape[0]
    in_dim = train_z.shape[1]
    mean_input_emb = train_z.mean(0)
    
    tabsyn.sample(
        num_samples=num_samples,
        in_dim=in_dim,
        mean_input_emb=mean_input_emb,
        info=data_info,
        num_inverse=num_inverse,
        cat_inverse=cat_inverse,
        save_path=synth_data_path,
    )
    logger.info(f"Sampled data saved to {synth_data_path}")

if __name__ == "__main__":
    main()
