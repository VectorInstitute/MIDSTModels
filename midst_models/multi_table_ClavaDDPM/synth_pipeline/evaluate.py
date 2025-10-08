import os
import json
import torch
import hydra
import logging
from omegaconf import DictConfig
from pprint import pprint

from midst_models.single_table_TabSyn.scripts.eval.eval_density import eval_density
from midst_models.single_table_TabSyn.scripts.eval.eval_quality import eval_quality
from midst_models.single_table_TabSyn.scripts.eval.eval_mle import eval_mle
from midst_models.single_table_TabSyn.scripts.eval.eval_dcr import eval_dcr
from midst_models.single_table_TabSyn.scripts.eval.eval_detection import eval_detection

logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="evaluate", version_base="1.1")
def main(cfg: DictConfig):
    data_dir = cfg.data_dir
    exp_name = cfg.exp_name
    data_name = cfg.data_name
    workspace_dir = os.path.join(data_dir, "workspace", exp_name)
    
    main_data_path = cfg.main_data_path
    train_data_path = os.path.join(data_dir, "train.csv")
    test_data_path = os.path.join(data_dir, "test.csv")
    synth_data_path = os.path.join(workspace_dir, data_name, "_final", f"{data_name}_synthetic.csv")
    
    
    info_path = os.path.join(data_dir, "info.json")
    
    eval_results = {}

    # 1. Density Evaluation
    if cfg.eval_flags.eval_density:
        logger.info("Running density evaluation for train data...")
        shape, trend = eval_density(synth_data_path, train_data_path, info_path)
        eval_results['density_shape_train'] = shape
        eval_results['density_trend_train'] = trend
        logger.info(f"Shape: {shape}, Trend: {trend}")
        
        logger.info("Running density evaluation for all data...")
        shape, trend = eval_density(synth_data_path, main_data_path, info_path)
        eval_results['density_shape_all'] = shape
        eval_results['density_trend_all'] = trend
        logger.info(f"Shape: {shape}, Trend: {trend}")

    # 2. Quality Evaluation
    if cfg.eval_flags.eval_quality:
        logger.info("Running quality evaluation...")
        alpha_precision, beta_recall = eval_quality(synth_data_path, train_data_path, info_path)
        eval_results['alpha_precision'] = alpha_precision
        eval_results['beta_recall'] = beta_recall
        logger.info(f"Alpha Precision: {alpha_precision}, Beta Recall: {beta_recall}")

    # 3. DCR (Distance to Closest Record) Evaluation
    if cfg.eval_flags.eval_dcr:
        logger.info("Running DCR evaluation...")
        with open(info_path, "r") as file:
            data_info = json.load(file)
        ideal_dcr = data_info["train_num"] / (data_info["train_num"] + data_info["test_num"])
        
        dcr_score = eval_dcr(synth_data_path, train_data_path, test_data_path, info_path)
        eval_results['dcr_score'] = dcr_score
        eval_results['ideal_dcr'] = ideal_dcr
        logger.info(f"DCR Score: {dcr_score}, Ideal DCR: {ideal_dcr}")

    # 4. Detection Evaluation
    if cfg.eval_flags.eval_detection:
        logger.info("Running detection evaluation for train data...")
        detection_score = eval_detection(synth_data_path, train_data_path, info_path, data_name, model="tabddpm")
        eval_results['detection_score_train'] = detection_score
        logger.info(f"Detection Score: {detection_score}")
        
        logger.info("Running detection evaluation for all data...")
        detection_score = eval_detection(synth_data_path, main_data_path, info_path, data_name, model="tabddpm")
        eval_results['detection_score_all'] = detection_score
        logger.info(f"Detection Score: {detection_score}")

    # 5. MLE Evaluation
    if cfg.eval_flags.eval_mle:
        logger.info("Running MLE evaluation...")
        mle_synth_score = eval_mle(synth_data_path, test_data_path, info_path)
        mle_baseline_score = eval_mle(train_data_path, test_data_path, info_path)
        eval_results['mle_synth_score'] = mle_synth_score
        eval_results['mle_baseline_score'] = mle_baseline_score
        logger.info("MLE Scores:")
        logger.info(f"Synthetic: {mle_synth_score}, Baseline: {mle_baseline_score}")

    # Save evaluation results to a file
    eval_json_path = os.path.join(workspace_dir, f"{data_name}_eval.json")
    with open(eval_json_path, 'w') as eval_file:
        json.dump(eval_results, eval_file, indent=4)
    
    logger.info(f"Evaluation results saved to {eval_json_path}")
    

if __name__ == "__main__":
    main()
