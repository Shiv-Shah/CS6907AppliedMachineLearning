#!/bin/bash

# Note: to be run from inside scripts folder.
cd ../src || { echo "Could not change directory, aborting..."; exit 1; }


# variables
base_name="csaw-m_multi_hot_final"
loss_type="multi_hot"
n_repeats=5  # how many runs on training we want to have

train_folder="/Users/trs/Desktop/ML_Workspace/14687271/images/preprocessed/train"
train_csv="/Users/trs/Desktop/ML_Workspace/14687271/labels/CSAW-M_train.csv"
checkpoints_path="/Users/trs/Desktop/ML_Workspace/14687271/CSAW-M-review_checkpoints"
n_epochs=40

set -e
source /Users/trs/Desktop/ML_Workspace/.venv/bin/activate

run_train() {
    echo "Training model" $model_name on GPU: $gpu" => started"
    python3 main.py --train \
                --model_name $model_name \
                --loss_type $loss_type \
                --train_folder $train_folder \
                --train_csv $train_csv \
                --checkpoints_path $checkpoints_path \
                --n_epochs $n_epochs \
                --gpu_id $gpu
}

for (( i=1; i<=$n_repeats; i++ ))
do
#    gpu=$(($i-1))
    gpu=7
    model_name="${base_name}_run_${i}"
    run_train
done
deactivate

