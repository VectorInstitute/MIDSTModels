import os
import logging
import hydra
import numpy as np
import pickle
from omegaconf import DictConfig
from midst_models.multi_table_ClavaDDPM.complex_pipeline import (
    clava_clustering,
    clava_training,
    clava_load_pretrained,
    clava_synthesizing,
    load_configs,
)
from midst_models.multi_table_ClavaDDPM.pipeline_modules import load_multi_table
from midst_models.multi_table_ClavaDDPM.pipeline_utils import get_df_without_id
logger = logging.getLogger(__name__)

def save_table_info(tables, relation_order, models, save_dir):
    table_info = {}
    for parent, child in relation_order:
        result = models[(parent, child)]
        df_with_cluster = tables[child]['df']
        df_without_id = get_df_without_id(df_with_cluster)
        df_info = result['df_info']
        X_num_real = df_without_id[df_info['num_cols']].to_numpy().astype(float)
        uniq_vals_list = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            uniq_vals_list.append(uniq_vals)
        table_info[child] = {
            'uniq_vals_list': uniq_vals_list,
            'size': len(df_with_cluster),
            'columns': tables[child]['df'].columns,
            'parents': tables[child]['parents'],
            'original_cols': tables[child]['original_cols']
        }
        required_keys = ['num_numerical_features', 'is_regression', 'inverse_transform', 'empirical_class_dist', 'K']
        filtered_result = {key: result[key] for key in required_keys}
        table_info[child].update(filtered_result)
    pickle.dump(table_info, open(os.path.join(save_dir, 'synthesize_table_info.pkl'), 'wb'))

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
    # save_table_info(tables, relation_order, models, exp_dir)
    # exit()

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
