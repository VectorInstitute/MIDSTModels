#!/bin/bash

main_dataset_dir="/projects/aieng/midst_competition/data/tabsyn_round3"
log_dir="/projects/aieng/midst_competition/logs/tabsyn_round3"

datasets=($(ls -d $main_dataset_dir/*/ | grep -v "all_data" | sort))

num_gpus=7
batch_size=7
num_datasets=${#datasets[@]}

for (( i=0; i<$num_datasets; i+=batch_size )); do
    batch=("${datasets[@]:i:batch_size}")
    job_ids=()
    for dataset in "${batch[@]}"; do
        dataset_name=$(basename $dataset)
        output_log="${log_dir}/${dataset_name}_%j.log"
        echo "Submitting job for dataset: $dataset_name"
        echo "Dataset path: $dataset"
        job_id=$(sbatch --job-name="${dataset_name}" \
               --output=$output_log \
               --export=DATASET_DIR=$dataset \
               tabsyn.slrm &)
        job_ids+=($job_id)
        if [[ $(jobs -r -p | wc -l) -ge $num_gpus ]]; then
            wait -n
        fi
    done
    echo "Sleeping for 50 minutes before submitting the next batch..."
    sleep 3000

    # echo "Waiting for batch $((i/batch_size + 1)) to complete..."
    # for job_id in "${job_ids[@]}"; do
    #     while squeue -j $job_id > /dev/null 2>&1; do
    #         echo "Job $job_id is still running. Checking again in 60 seconds."
    #         sleep 60
    #     done
    # done
    echo "Batch $((i/batch_size + 1)) complete."    
done
