import os
import json
import pandas as pd
import numpy as np
import hydra
import logging
import shutil
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

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

    main_clean = df.drop(columns=[col for col in df.columns if col.endswith('_id')])
    main_clean_file = os.path.join(cfg.save_data_dir, f'{cfg.data_name}.csv')
    main_clean.to_csv(main_clean_file, index=False)
    logger.info(f"Main dataset (without _id columns) saved at {main_clean_file}")

    for subset_num in range(111, num_subsets + 111):
        subset_dir = os.path.join(cfg.save_data_dir, f'tabddpm_{subset_num}')
        os.makedirs(subset_dir, exist_ok=True)

        # seed = subset_num * 100 + 4
        # seed = subset_num * 100 + 12
        seed = subset_num * 100 + 24
        rng = np.random.RandomState(seed=seed)
        # create the train set
        subset = df.sample(n=subset_size, replace=False, random_state=rng)
        
        train_with_id_file = os.path.join(subset_dir, f'train_with_id.csv')
        subset.to_csv(train_with_id_file, index=False)

        # create the challenge set
        # rng = np.random.RandomState(seed=seed + 5)
        # rng = np.random.RandomState(seed=seed + 13)
        rng = np.random.RandomState(seed=seed + 25)
        challenge_in_subset = subset.sample(n=challenge_size, random_state=rng)
        
        # rng = np.random.RandomState(seed=seed + 6)
        # rng = np.random.RandomState(seed=seed + 14)
        rng = np.random.RandomState(seed=seed + 26)
        challenge_out_subset = df[~df.index.isin(subset.index)].sample(
            n=challenge_size, random_state=rng
        )
        challenge_set = pd.concat([challenge_in_subset, challenge_out_subset])

        # create the test set
        combined_challenge_indices = challenge_set.index
        # rng = np.random.RandomState(seed=seed + 7)
        # rng = np.random.RandomState(seed=seed + 15)
        rng = np.random.RandomState(seed=seed + 27)
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
        subset_clean = subset.drop(columns=[col for col in subset.columns if col.endswith('_id')])
        train_file = os.path.join(subset_dir, f'train.csv')
        subset_clean.to_csv(train_file, index=False)

        challenge_clean = challenge_set.drop(columns=[col for col in challenge_set.columns if col.endswith('_id')])
        challenge_file = os.path.join(subset_dir, f'challenge.csv')
        challenge_clean.to_csv(challenge_file, index=False)

        test_clean = test_set.drop(columns=[col for col in test_set.columns if col.endswith('_id')])
        test_file = os.path.join(subset_dir, f'test.csv')
        test_clean.to_csv(test_file, index=False)

        # save the data json and info json files
        data_json_path = os.path.join(cfg.main_data_dir, f'{cfg.data_name}.json')
        with open(data_json_path, 'r') as f:
            data_config = json.load(f)
        data_config['general']['data_dir'] = subset_dir
        data_config['general']['exp_name'] = f'tabddpm_{subset_num}'
        data_config['general']['workspace_dir'] = os.path.join(subset_dir, 'workspace')
        data_config['general']['test_data_dir'] = subset_dir

        data_json_save_path = os.path.join(subset_dir, f'{cfg.data_name}.json')
        with open(data_json_save_path, 'w') as f:
            json.dump(data_config, f, indent=4)

        info_json_path = os.path.join(cfg.main_data_dir, 'info.json')
        with open(info_json_path, 'r') as f:
            info_config = json.load(f)
        info_config['data_path'] = train_file
        info_config['train_num'] = len(subset_clean)
        info_config['test_num'] = len(test_clean)

        info_json_save_path = os.path.join(subset_dir, 'info.json')
        with open(info_json_save_path, 'w') as f:
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
