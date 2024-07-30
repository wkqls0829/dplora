#!/bin/bash

num_client=4
data_path=~/lora/FederatedScope/data/1613/
data_names=(678 695 696 114)
data_name=0
lora_r=64
num_rounds=10
client_epochs=1
learning_rate=1e-6
model=google-bert/bert-base-cased
# model=datajuicer/LLaMA-1B-dj-refine-150B
mode=dplora
projection_type=gradient

tid=10100

nohup python -u server.py \
    --num_client $num_client --data_name $data_name --rank 0 \
    --num_rounds $num_rounds --client_epochs $client_epochs --client_ckpt $model \
    --mode $mode --lora_r $lora_r --client_lr $learning_rate \
    > outputs/${tid}.out 2>&1 &

for client in 0 1 2 3
do
    device=$((client+4))
    data_name=${data_names[$client]}
    local_r=16
    nohup python -u dpl-client.py \
        --num_client $num_client --data_path $data_path --data_name $data_name --rank $client \
        --num_rounds $num_rounds --client_epochs $client_epochs --client_ckpt $model \
        --mode $mode --lora_r $lora_r --local_r $local_r --client_lr $learning_rate \
        --device $device --projection_type $projection_type \
        > outputs/client/${tid}_${client}.log 2>&1 &
done
