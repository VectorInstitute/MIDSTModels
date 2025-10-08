import os
import json
import pandas as pd
import numpy as np
import hydra
import logging
import shutil
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from midst_models.single_table_TabSyn.scripts.process_dataset import process_data

logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="generate_subsets", version_base="1.1")
def generate_subsets(cfg: DictConfig):
    
    num_subsets = cfg.num_subsets
    subset_size = cfg.subset_size
    test_size = cfg.test_size
    challenge_size = cfg.challenge_size // 2
    
    os.makedirs(cfg.save_data_dir, exist_ok=True)
    
    main_file = os.path.join(cfg.main_data_dir, f'{cfg.data_name}.csv')
    df = pd.read_csv(main_file)
    
    main_info = json.load(open(cfg.main_info_file, 'r'))

    main_clean = df.drop(columns=[col for col in df.columns if col.endswith('_id')])
    main_clean_file = os.path.join(cfg.save_data_dir, f'{cfg.data_name}.csv')
    main_clean.to_csv(main_clean_file, index=False)
    logger.info(f"Main dataset (without _id columns) saved at {main_clean_file}")

    for subset_num in range(111, num_subsets + 111):
        subset_dir = os.path.join(cfg.save_data_dir, f'tabsyn_{subset_num}')
        os.makedirs(subset_dir, exist_ok=True)
        raw_data_dir = os.path.join(subset_dir, 'raw_data')
        os.makedirs(raw_data_dir, exist_ok=True)
        processed_data_dir = os.path.join(subset_dir, 'processed_data')
        os.makedirs(processed_data_dir, exist_ok=True)
        
        seed = subset_num * 100
        # rng = np.random.RandomState(seed=seed + 16)
        rng = np.random.RandomState(seed=seed + 28)
        # create the train set
        subset = df.sample(n=subset_size, replace=False, random_state=rng)
        train_with_id_file = os.path.join(subset_dir, f'train_with_id.csv')
        subset.to_csv(train_with_id_file, index=False)

        # create the challenge set
        # rng = np.random.RandomState(seed=seed + 1)
        rng = np.random.RandomState(seed=seed + 29)
        challenge_in_subset = subset.sample(n=challenge_size, random_state=rng)
        
        # rng = np.random.RandomState(seed=seed + 2)
        rng = np.random.RandomState(seed=seed + 30)
        challenge_out_subset = df[~df.index.isin(subset.index)].sample(
            n=challenge_size, random_state=rng
        )
        challenge_set = pd.concat([challenge_in_subset, challenge_out_subset])
        
        # create the test set
        combined_challenge_indices = challenge_set.index
        # rng = np.random.RandomState(seed=seed + 3)
        rng = np.random.RandomState(seed=seed + 31)
        test_set = df[~df.index.isin(subset.index) & ~df.index.isin(combined_challenge_indices)].sample(
            n=test_size, random_state=rng
        )
        test_with_id_file = os.path.join(subset_dir, f'test_with_id.csv')
        test_set.to_csv(test_with_id_file, index=False)

        # shuffle the challenge set
        challenge_set = challenge_set.sample(frac=1, random_state=subset_num).reset_index(drop=True)
        challenge_with_id_file = os.path.join(subset_dir, f'challenge_with_id.csv')
        challenge_set.to_csv(challenge_with_id_file, index=False)
        
        # create the challenge label file
        labels = []
        for idx, row in challenge_set.iterrows():
            if (subset == row).all(axis=1).any():
                labels.append(1)
            else:
                labels.append(0)
        
        challenge_label_df = pd.DataFrame({'is_train': labels})
        challenge_label_file = os.path.join(subset_dir, f'challenge_label.csv')
        challenge_label_df.to_csv(challenge_label_file, index=False)
        
        #  drop the _id columns and save the files
        challenge_clean = challenge_set.drop(columns=[col for col in challenge_set.columns if col.endswith('_id')])
        challenge_file = os.path.join(subset_dir, f'challenge.csv')
        challenge_clean.to_csv(challenge_file, index=False)
        
        subset_clean = subset.drop(columns=[col for col in subset.columns if col.endswith('_id')])
        train_file = os.path.join(raw_data_dir, f'train.csv')
        subset_clean.to_csv(train_file, index=False)

        test_clean = test_set.drop(columns=[col for col in test_set.columns if col.endswith('_id')])
        test_file = os.path.join(raw_data_dir, f'test.csv')
        test_clean.to_csv(test_file, index=False)

        info_path = os.path.join(cfg.main_data_dir, 'data_info', f'{cfg.data_name}.json')
        with open(info_path, 'r') as f:
            info_config = json.load(f)
        info_config['data_path'] = train_file
        info_config['test_path'] = test_file

        info_save_path = os.path.join(subset_dir, f'{cfg.data_name}.json')
        with open(info_save_path, 'w') as f:
            json.dump(info_config, f, indent=4)
        
        process_data(cfg.data_name, subset_dir, subset_dir)
        info_file = os.path.join(subset_dir, 'processed_data', cfg.data_name, 'info.json')
        info_config = json.load(open(info_file, 'r'))
        info_config['column_info'] = main_info['column_info']
        with open(info_file, 'w') as f:
            json.dump(info_config, f, indent=4)
            
        info_file = os.path.join(subset_dir, 'processed_data', cfg.data_name, 'info.json')
        info_config = json.load(open(info_file, 'r'))
        info_config['column_info'] = main_info['column_info']
        with open(info_file, 'w') as f:
            json.dump(info_config, f, indent=4)
            
        # copy the files in files_to_copy to the subset directory
        for file_name in cfg.files_to_copy:
            src_file = os.path.join(cfg.main_data_dir, file_name)
            dest_file = os.path.join(subset_dir, file_name)
            if os.path.exists(src_file):
                shutil.copy(src_file, dest_file)
                logger.info(f"Copied {file_name} to {dest_file}")
            else:
                logger.warning(f"{file_name} not found in {cfg.main_data_dir}")
            
        logger.info(f"Subset {subset_num} generated successfully in {subset_dir}")


if __name__ == "__main__":
    generate_subsets()
