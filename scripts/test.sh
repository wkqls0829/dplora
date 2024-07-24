#!/bin/bash

num_client=2
data_path=~/FederatedScope/data/1613/
data_names=(eval_1 eval_2 eval_3 eval_4)
data_name=task001
lora_r=64
num_rounds=3
client_epochs=1
model=datajuicer/LLaMA-1B-dj-refine-150B
mode=dplora
projection_type=gradient

tid=00000

nohup python -u server.py \
    --num_client $num_client --data_name $data_name --rank 0 \
    --num_rounds $num_rounds --client_epochs $client_epochs --client_ckpt $model \
    --mode $mode --lora_r $lora_r \
    > test.out 2>&1 &

for client in 0 1
do
    device=$((client+6))
    data_name=${data_names[$client]}
    local_r=4
    nohup python -u dpl-client.py \
        --num_client $num_client --data_path $data_path --data_name $data_name --rank $client \
        --num_rounds $num_rounds --client_epochs $client_epochs --client_ckpt $model \
        --mode $mode --lora_r $lora_r --local_r $local_r \
        --device $device --projection_type $projection_type \
        > outputs/test_${client}.out 2>&1 &
done
