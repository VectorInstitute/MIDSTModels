import os
import logging
import hydra
from omegaconf import DictConfig
from midst_models.single_table_TabDDPM.complex_pipeline import (
    clava_clustering,
    clava_training,
    clava_load_pretrained,
    clava_synthesizing,
    load_configs,
)
from midst_models.single_table_TabDDPM.pipeline_modules import load_multi_table

logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="sample", version_base="1.1")
def main(cfg: DictConfig):
    data_dir = cfg.data_dir
    exp_name = cfg.exp_name
    data_name = cfg.data_name

    exp_dir = os.path.join(data_dir, "workspace", exp_name)

    config_path = os.path.join(exp_dir, f"{data_name}.json")
    logger.info(f"Loading configuration from: {config_path}")
    configs, _ = load_configs(config_path)

    tables, relation_order, dataset_meta = load_multi_table(data_dir)

    params_clustering = configs['clustering']
    logger.info(f"{'='*20} Clustering Parameters {'='*20}")
    for key, val in params_clustering.items():
        logger.info(f"{key}: {val}")

    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, exp_dir, configs)

    logger.info("Loading pretrained models from: %s", exp_dir)
    models = clava_load_pretrained(relation_order, exp_dir)

    params_sampling = configs['sampling']
    logger.info(f"{'='*20} Sampling Parameters {'='*20}")
    for key, val in params_sampling.items():
        logger.info(f"{key}: {val}")

    sample_scale = 1 if 'debug' not in configs else configs['debug']['sample_scale']
    cleaned_tables, synthesizing_time_spent, matching_time_spent = clava_synthesizing(
        # tables,
        list(tables.keys()),
        relation_order,
        exp_dir,
        all_group_lengths_prob_dicts,
        models,
        configs,
        sample_scale=sample_scale
    )

if __name__ == "__main__":
    main()
