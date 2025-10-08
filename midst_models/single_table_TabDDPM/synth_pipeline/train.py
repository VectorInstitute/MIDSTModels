import os
import json
import logging
import shutil
import uuid
import hydra
from omegaconf import DictConfig

import sys
import os

from midst_models.single_table_TabDDPM.complex_pipeline import (
    clava_clustering, 
    clava_training, 
    clava_load_pretrained, 
    clava_synthesizing, 
    clava_load_synthesized_data, 
    clava_eval, 
    load_configs,
)
from midst_models.single_table_TabDDPM.pipeline_modules import load_multi_table
from midst_models.single_table_TabDDPM.gen_single_report import gen_single_report
from midst_models.single_table_TabDDPM.pipeline_utils import load_multi_table

logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="train", version_base="1.1")
def main(cfg: DictConfig):
    logger.info(f"running experiment: {cfg.exp_name}")
    
    data_dir = cfg.data_dir
    logger.info(f"Data directory: {data_dir}")

    config_path = os.path.join(data_dir, f"{cfg.data_name}.json")
    logger.info(f"Loading configuration from: {config_path}")
    configs, _ = load_configs(config_path)
    configs['general']['exp_name'] = cfg.exp_name
    
    save_dir = os.path.join(configs['general']['workspace_dir'], configs['general']['exp_name'])
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'before_matching'), exist_ok=True)

    updated_config_path = os.path.join(save_dir, f"{cfg.data_name}.json")
    with open(updated_config_path, 'w') as config_file:
        json.dump(configs, config_file, indent=4)
    logger.info(f"Exp configuration saved to: {updated_config_path}")

    tables, relation_order, dataset_meta = load_multi_table(data_dir)
    
    params_clustering = configs['clustering']
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, save_dir, configs)
    
    params_training = configs['diffusion']
    logger.info(f"Training parameters: {params_training}")
    models = clava_training(tables, relation_order, save_dir, configs)

if __name__ == "__main__":
    main()
