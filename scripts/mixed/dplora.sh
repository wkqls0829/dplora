#!/bin/bash

num_client=4
data_path=~/lora/FederatedScope/data/1613/
data_names=(390 399 400 407)
data_name=0
lora_r=32
num_rounds=20
client_epochs=1
learning_rate=5e-4

#model=google-bert/bert-base-cased
model=datajuicer/LLaMA-1B-dj-refine-150B
mode=dplora
projection_type=gradient

tid=10310

nohup python -u server.py \
    --num_client $num_client --data_name $data_name --rank 0 \
    --num_rounds $num_rounds --client_epochs $client_epochs --client_ckpt $model \
    --mode $mode --lora_r $lora_r --client_lr $learning_rate --tid $tid \
    > outputs/${tid}.log 2>&1 &

for client in 0 1 2 3
do
    export CUDA_VISIBLE_DEVICES=$((client+4))
    device=0 #$((client+4)))
    data_name=${data_names[$client]}
    local_r=8
    nohup python -u dpl-client.py \
        --num_client $num_client --data_path $data_path --data_name $data_name --rank $client \
        --num_rounds $num_rounds --client_epochs $client_epochs --client_ckpt $model \
        --mode $mode --lora_r $lora_r --local_r $local_r --client_lr $learning_rate \
        --device $device --projection_type $projection_type --tid $tid \
        > outputs/client/${tid}_${client}.log 2>&1 &
done
