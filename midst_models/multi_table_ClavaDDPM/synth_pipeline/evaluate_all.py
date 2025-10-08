import os
import json
import torch
import hydra
import logging
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path

from midst_models.single_table_TabSyn.scripts.eval.eval_density import eval_density
from midst_models.single_table_TabSyn.scripts.eval.eval_quality import eval_quality
from midst_models.single_table_TabSyn.scripts.eval.eval_mle import eval_mle
from midst_models.single_table_TabSyn.scripts.eval.eval_dcr import eval_dcr
from midst_models.single_table_TabSyn.scripts.eval.eval_detection import eval_detection

from midst_models.multi_table_ClavaDDPM.complex_pipeline import (
    clava_clustering, 
    clava_load_pretrained, 
    clava_load_synthesized_data, 
    clava_eval, 
    load_configs
)
from midst_models.multi_table_ClavaDDPM.pipeline_utils import load_multi_table
from midst_models.multi_table_ClavaDDPM.report_utils import get_multi_metadata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# List of tables to process
TABLES = ['account', 'card', 'client', 'disp', 'district', 'loan', 'order', 'trans']

@hydra.main(config_path="config", config_name="evaluate_all", version_base="1.1")
def main(cfg: DictConfig):
    base_dir = Path(cfg.base_dir)  # Parent directory containing clavaddpm_1 to clavaddpm_70

    # Iterate through all subdirectories clavaddpm_1 to clavaddpm_70
    for i in range(92, 111):
        subdir_name = f"clavaddpm_{i}"
        subdir_path = base_dir / subdir_name
        exp_dir = subdir_path / "workspace" / cfg.exp_name

        if not subdir_path.exists() or not subdir_path.is_dir():
            raise FileNotFoundError(f"Subdirectory {subdir_path} does not exist or is not a directory. Skipping.")

        logger.info(f"Processing subdirectory: {subdir_path}")

        eval_dir = subdir_path / "eval_files"

        eval_results = {}

        for table in TABLES:
            # Paths
            synth_data_with_id = subdir_path / "workspace" / cfg.exp_name / table / "_final" / f"{table}_synthetic.csv"
            synth_df = pd.read_csv(synth_data_with_id)
            synth_df_no_id = synth_df.drop(columns=[col for col in synth_df.columns if col.endswith("_id")])
            
            info_json_path = subdir_path / "eval_files" / f"{table}_info.json"
            with open(info_json_path, 'r') as f:
                info_data = json.load(f)
            
            cat_col_idx = info_data.get("cat_col_idx", [])
            target_col_idx = info_data.get("target_col_idx", [])
            task_type = info_data.get("task_type", "")
            
            if task_type != "regression":
                cat_col_idx += target_col_idx
                
            for idx in cat_col_idx:
                col_name = synth_df_no_id.columns[idx]
                synth_df_no_id[col_name] = synth_df_no_id[col_name].apply(
                    lambda x: int(float(x)) if str(x).replace('.0', '').isdigit() else x
                )
            synth_data_no_id = subdir_path / "workspace" / cfg.exp_name / table / "_final" / f"{table}_synthetic_no_id.csv"
            synth_df_no_id.to_csv(synth_data_no_id, index=False)
            
            info_path = eval_dir / f"{table}_info.json"
            train_data_path = eval_dir / f"{table}_no_id.csv"
            synth_data_path = subdir_path / "workspace" / cfg.exp_name / table / "_final" / f"{table}_synthetic_no_id.csv"

            if not info_path.exists():
                raise FileNotFoundError(f"{info_path} does not exist. Skipping updating data_path.")
            
            with open(info_path, 'r') as f:
                info_data = json.load(f)
            logger.info(f"Loaded {info_path}")

            if cfg.eval_flags.eval_density:

                shape, trend = eval_density(str(synth_data_path), str(train_data_path), str(info_path))
                eval_results[f'density_shape_train_{table}'] = shape
                eval_results[f'density_trend_train_{table}'] = trend
                logger.info(f"Density Evaluation for {table}: Shape={shape}, Trend={trend}")
                    

            # Other Evaluations only for 'trans' table
            if table == 'trans':
                test_data_path = eval_dir / "test_no_id.csv"  # Assuming test_no_id.csv is in eval_files
                main_data_path = cfg.main_data_path
                
                if cfg.eval_flags.eval_density:
                    shape, trend = eval_density(synth_data_path, main_data_path, info_path)
                    eval_results['density_shape_all'] = shape
                    eval_results['density_trend_all'] = trend
                    logger.info(f"Shape: {shape}, Trend: {trend}")

                if cfg.eval_flags.eval_quality:
                    alpha_precision, beta_recall = eval_quality(str(synth_data_path), str(train_data_path), str(info_path))
                    eval_results['alpha_precision_trans'] = alpha_precision
                    eval_results['beta_recall_trans'] = beta_recall
                    logger.info(f"Quality Evaluation for trans: Alpha Precision={alpha_precision}, Beta Recall={beta_recall}")

                if cfg.eval_flags.eval_dcr:
                    ideal_dcr = info_data["train_num"] / (info_data["train_num"] + info_data["test_num"])
                    dcr_score = eval_dcr(str(synth_data_path), str(train_data_path), str(test_data_path), str(info_path))
                    eval_results['dcr_score_trans'] = dcr_score
                    eval_results['ideal_dcr_trans'] = ideal_dcr
                    logger.info(f"DCR Evaluation for trans: DCR Score={dcr_score}, Ideal DCR={ideal_dcr}")
                        
                if cfg.eval_flags.eval_detection:
                    detection_score = eval_detection(str(synth_data_path), str(train_data_path), str(info_path), cfg.data_name, model="clavaddpm")
                    eval_results['detection_score_train_trans'] = detection_score
                    logger.info(f"Detection Evaluation for trans (train): Detection Score={detection_score}")

                    detection_score_all = eval_detection(str(synth_data_path), str(main_data_path), str(info_path), cfg.data_name, model="clavaddpm")
                    eval_results['detection_score_all_trans'] = detection_score_all
                    logger.info(f"Detection Evaluation for trans (all): Detection Score={detection_score_all}")
                        

                if cfg.eval_flags.eval_mle:
                    mle_synth_score = eval_mle(str(synth_data_path), str(test_data_path), str(info_path))
                    mle_baseline_score = eval_mle(str(train_data_path), str(test_data_path), str(info_path))
                    eval_results['mle_synth_score_trans'] = mle_synth_score
                    eval_results['mle_baseline_score_trans'] = mle_baseline_score
                    logger.info(f"MLE Evaluation for trans: Synthetic Score={mle_synth_score}, Baseline Score={mle_baseline_score}")
                       
        tables, relation_order, dataset_meta = load_multi_table(subdir_path)
        cleaned_tables = clava_load_synthesized_data(tables.keys(), exp_dir)  
        config_path = subdir_path / f"{cfg.data_name}.json"                  
        configs, _ = load_configs(config_path)

        report = clava_eval(tables, subdir_path, configs, relation_order, cleaned_tables)
        for key, val in report.items():
            if key in ['avg_scores', 'all_avg_score']:
                if key == 'avg_scores':
                    for k, v in val.items():
                        logger.info("{:}-hop column correlation: {}".format(k, v))
                        eval_results[f"{k}_hop_column_correlation"] = v
                elif key == 'all_avg_score':
                    logger.info("{:}: {}".format(key, val))
                    eval_results[key] = val

        # Save all evaluation results to eval.json in the subdir
        eval_json_path = subdir_path / "eval.json"
        with open(eval_json_path, 'w') as eval_file:
            json.dump(eval_results, eval_file, indent=4)
         

if __name__ == "__main__":
    main()
