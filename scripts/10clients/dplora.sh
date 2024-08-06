#!/bin/bash

num_client=10
data_path=~/lora/FederatedScope/data/1613/
data_names=(390_0 390_1 396_0 396_1 399_0 399_1 400_0 400_1 401_0 401_1)
data_name=10clients
lora_r=64
num_rounds=20
client_epochs=1
learning_rate=1e-4

model=datajuicer/LLaMA-1B-dj-refine-150B
mode=dplora
projection_type=gradient
local_ranks=(16 16 16 16 16 16 16 16 16 16)

tid=19300

#export CUDA_VISIBLE_DEVICES=4
nohup python -u server.py \
    --num_client $num_client --data_name $data_name --rank 0 \
    --num_rounds $num_rounds --client_epochs $client_epochs --client_ckpt $model \
    --mode $mode --lora_r $lora_r --client_lr $learning_rate --tid $tid \
    > outputs/${tid}.log 2>&1 &

for client in 0 1 2 3 4 5 6 7 8 9
do
    export CUDA_VISIBLE_DEVICES=$((client % 5))
    device=0
    data_name=${data_names[$client]}
    # local_r=16
    local_r=${local_ranks[$client]}
    nohup python -u dpl-client.py \
        --num_client $num_client --data_path $data_path --data_name $data_name --rank $client \
        --num_rounds $num_rounds --client_epochs $client_epochs --client_ckpt $model \
        --mode $mode --lora_r $lora_r --local_r $local_r --client_lr $learning_rate \
        --device $device --projection_type $projection_type --tid $tid \
        > outputs/client/${tid}_${client}.log 2>&1 &
done
