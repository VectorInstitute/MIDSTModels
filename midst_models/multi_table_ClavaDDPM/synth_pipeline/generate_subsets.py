import glob
import os
import json
import pandas as pd
import numpy as np
import hydra
import logging
import shutil
from pathlib import Path
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
    
    account_df = pd.read_csv(os.path.join(cfg.main_data_dir, 'account.csv'))
    card_df = pd.read_csv(os.path.join(cfg.main_data_dir, 'card.csv'))
    client_df = pd.read_csv(os.path.join(cfg.main_data_dir, 'client.csv'))
    disp_df = pd.read_csv(os.path.join(cfg.main_data_dir, 'disp.csv'))
    district_df = pd.read_csv(os.path.join(cfg.main_data_dir, 'district.csv'))
    loan_df = pd.read_csv(os.path.join(cfg.main_data_dir, 'loan.csv'))
    order_df = pd.read_csv(os.path.join(cfg.main_data_dir, 'order.csv'))
    trans_df = pd.read_csv(os.path.join(cfg.main_data_dir, 'trans.csv'))
    
    main_clean = trans_df.drop(columns=[col for col in trans_df.columns if col.endswith('_id')])
    main_clean_file = os.path.join(cfg.save_data_dir, f'{cfg.data_name}.csv')
    main_clean.to_csv(main_clean_file, index=False)
    logger.info(f"Main dataset (without _id columns) saved at {main_clean_file}")
    
    id_columns = ['account_id', 'client_id', 'disp_id', 'trans_id', 'card_id', 'loan_id', 'order_id', 'district_id']
    for df in [account_df, card_df, client_df, disp_df, loan_df, order_df, trans_df]:
        for col in id_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)

    # Begin processing for each subset
    for subset_num in range(71, num_subsets + 71):
        print(f'Processing subset {subset_num}...')
        
        # Create a subdirectory for this subset
        subset_dir = os.path.join(cfg.save_data_dir, f'clavaddpm_{subset_num}')
        os.makedirs(subset_dir, exist_ok=True)
        
        eval_dir = os.path.join(subset_dir, 'eval_files')
        os.makedirs(eval_dir, exist_ok=True)
        
        seed = subset_num * 100 + 36
        rng = np.random.RandomState(seed=seed)
        
        # Sample uniformly 20k from trans data
        sampled_trans_df = trans_df.sample(n=subset_size, random_state=rng)
        
        # Extract corresponding account records using account_id
        sampled_account_ids = sampled_trans_df['account_id'].unique()
        sampled_account_df = account_df[account_df['account_id'].isin(sampled_account_ids)]
        
        # Extract corresponding district records from account
        sampled_district_ids = sampled_account_df['district_id'].unique()
        sampled_district_df = district_df[district_df['district_id'].isin(sampled_district_ids)]
        
        # Extract corresponding order, loan, disposition
        sampled_disp_df = disp_df[disp_df['account_id'].isin(sampled_account_ids)]
        sampled_order_df = order_df[order_df['account_id'].isin(sampled_account_ids)]
        sampled_loan_df = loan_df[loan_df['account_id'].isin(sampled_account_ids)]
        
        # From disposition extract corresponding card
        sampled_card_ids = sampled_disp_df['disp_id'].unique()
        sampled_card_df = card_df[card_df['disp_id'].isin(sampled_card_ids)]
        
        # From disposition extract corresponding client
        sampled_client_ids = sampled_disp_df['client_id'].unique()
        sampled_client_df = client_df[client_df['client_id'].isin(sampled_client_ids)]
        
        # From client extract corresponding district and add any new districts
        client_district_ids = sampled_client_df['district_id'].unique()
        additional_district_ids = np.setdiff1d(client_district_ids, sampled_district_ids)
        if additional_district_ids.size > 0:
            additional_district_df = district_df[district_df['district_id'].isin(additional_district_ids)]
            sampled_district_df = pd.concat([sampled_district_df, additional_district_df]).drop_duplicates()
        
        # Save these files in their own subdir
        sampled_trans_df.to_csv(os.path.join(subset_dir, 'trans.csv'), index=False)
        sampled_account_df.to_csv(os.path.join(subset_dir, 'account.csv'), index=False)
        sampled_district_df.to_csv(os.path.join(subset_dir, 'district.csv'), index=False)
        sampled_order_df.to_csv(os.path.join(subset_dir, 'order.csv'), index=False)
        sampled_loan_df.to_csv(os.path.join(subset_dir, 'loan.csv'), index=False)
        sampled_disp_df.to_csv(os.path.join(subset_dir, 'disp.csv'), index=False)
        sampled_card_df.to_csv(os.path.join(subset_dir, 'card.csv'), index=False)
        sampled_client_df.to_csv(os.path.join(subset_dir, 'client.csv'), index=False)
        
        # save files without id
        trans_df_no_id = sampled_trans_df.drop(columns=[col for col in sampled_trans_df.columns if col.endswith('_id')])
        trans_df_no_id.to_csv(os.path.join(eval_dir, f'trans_no_id.csv'), index=False)
        account_df_no_id = sampled_account_df.drop(columns=[col for col in sampled_account_df.columns if col.endswith('_id')])
        account_df_no_id.to_csv(os.path.join(eval_dir, f'account_no_id.csv'), index=False)
        district_df_no_id = sampled_district_df.drop(columns=[col for col in sampled_district_df.columns if col.endswith('_id')])
        district_df_no_id.to_csv(os.path.join(eval_dir, f'district_no_id.csv'), index=False)
        order_df_no_id = sampled_order_df.drop(columns=[col for col in sampled_order_df.columns if col.endswith('_id')])
        order_df_no_id.to_csv(os.path.join(eval_dir, f'order_no_id.csv'), index=False)
        loan_df_no_id = sampled_loan_df.drop(columns=[col for col in sampled_loan_df.columns if col.endswith('_id')])
        loan_df_no_id.to_csv(os.path.join(eval_dir, f'loan_no_id.csv'), index=False)
        disp_df_no_id = sampled_disp_df.drop(columns=[col for col in sampled_disp_df.columns if col.endswith('_id')])
        disp_df_no_id.to_csv(os.path.join(eval_dir, f'disp_no_id.csv'), index=False)
        card_df_no_id = sampled_card_df.drop(columns=[col for col in sampled_card_df.columns if col.endswith('_id')])
        card_df_no_id.to_csv(os.path.join(eval_dir, f'card_no_id.csv'), index=False)
        client_df_no_id = sampled_client_df.drop(columns=[col for col in sampled_client_df.columns if col.endswith('_id')])
        client_df_no_id.to_csv(os.path.join(eval_dir, f'client_no_id.csv'), index=False)
                                               
        # create the challenge set
        rng = np.random.RandomState(seed=seed + 37)
        challenge_in_subset = sampled_trans_df.sample(n=challenge_size, random_state=rng)
        
        rng = np.random.RandomState(seed=seed + 38)
        challenge_out_subset = trans_df[~trans_df.index.isin(sampled_trans_df.index)].sample(
            n=challenge_size, random_state=rng
        )
        challenge_set = pd.concat([challenge_in_subset, challenge_out_subset])

        # create the test set
        combined_challenge_indices = challenge_set.index
        rng = np.random.RandomState(seed=seed + 39)
        test_set = trans_df[~trans_df.index.isin(sampled_trans_df.index) & ~trans_df.index.isin(combined_challenge_indices)].sample(
            n=test_size, random_state=rng
        )
        test_with_id_file = os.path.join(subset_dir, f'test.csv')
        test_set.to_csv(test_with_id_file, index=False)
        
        # shuffle the challenge set
        challenge_set = challenge_set.sample(frac=1, random_state=subset_num).reset_index(drop=True)
        challenge_with_id_file = os.path.join(subset_dir, f'challenge.csv')
        challenge_set.to_csv(challenge_with_id_file, index=False)

        # create the challenge label file
        labels = []
        for idx, row in challenge_set.iterrows():
            if (sampled_trans_df == row).all(axis=1).any():
                labels.append(1)
            else:
                labels.append(0)
        
        challenge_label_df = pd.DataFrame({'is_train': labels})
        challenge_label_file = os.path.join(subset_dir, f'challenge_label.csv')
        challenge_label_df.to_csv(challenge_label_file, index=False)

        #  drop the _id columns and save the files
        challenge_clean = challenge_set.drop(columns=[col for col in challenge_set.columns if col.endswith('_id')])
        challenge_file = os.path.join(subset_dir, f'challenge_no_id.csv')
        challenge_clean.to_csv(challenge_file, index=False)

        test_clean = test_set.drop(columns=[col for col in test_set.columns if col.endswith('_id')])
        test_file = os.path.join(eval_dir, f'test_no_id.csv')
        test_clean.to_csv(test_file, index=False)

        # save the data json and info json files
        info_files = Path(cfg.info_dir).glob("*_info.json")
        for info_file in info_files:
            dest_info = os.path.join(eval_dir, os.path.basename(info_file))
            shutil.copy(str(info_file), str(dest_info))
            logging.info(f"Copied {info_file} to {dest_info}")
            
            table_name = info_file.stem.replace('_info', '')
            csv_file = Path(eval_dir) / f"{table_name}_no_id.csv"
            
            df = pd.read_csv(csv_file)
            train_size = len(df)
            
            with open(dest_info, 'r') as f:
                info_data = json.load(f)

            # Update fields
            info_data['data_path'] = str(csv_file.resolve())
            info_data['train_num'] = train_size
            
            if table_name == "trans":
                info_data['test_num'] = len(test_clean)
            else:
                info_data['test_num'] = 0
            
            with open(dest_info, 'w') as f:
                json.dump(info_data, f, indent=4)
        
        data_json_path = os.path.join(cfg.main_data_dir, f'{cfg.data_name}.json')
        with open(data_json_path, 'r') as f:
            data_config = json.load(f)
        data_config['general']['data_dir'] = subset_dir
        data_config['general']['exp_name'] = f'train_1'
        data_config['general']['workspace_dir'] = os.path.join(subset_dir, 'workspace')
        data_config['general']['test_data_dir'] = subset_dir

        data_json_save_path = os.path.join(subset_dir, f'{cfg.data_name}.json')
        with open(data_json_save_path, 'w') as f:
            json.dump(data_config, f, indent=4)            

        # copy the files in files_to_copy to the subset directory
        file_patterns = ['*.pkl', '*.txt', '*.json']
        for pattern in file_patterns:
        # Use glob to find all files matching the pattern in the source directory recursively
            pattern_path = os.path.join(cfg.main_data_dir, '**', pattern)
            matching_files = glob.glob(pattern_path, recursive=True)

            print(f"Copying files matching {pattern}: {len(matching_files)} files found.")

            for file_path in matching_files:
                file_name = os.path.basename(file_path)
                if file_name in cfg.files_to_exclude:
                    print(f"Skipping excluded file: {file_name}")
                    continue
                # Copy each file to the target directory, preserving the directory structure
                target_file_path = os.path.join(subset_dir, file_name)
                if os.path.exists(target_file_path):
                    print(f"File {target_file_path} already exists. Skipping to prevent overwrite.")
                    continue
                shutil.copy2(file_path, target_file_path)
                logger.info(f"Copied {file_path} to {target_file_path}")
            
        logger.info(f"Subset {subset_num} generated successfully in {subset_dir}")


if __name__ == "__main__":
    generate_subsets()