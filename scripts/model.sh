#!/bin/bash

num_client=4
data_name=cola
rank=4
num_rounds=10
model=FacebookAI/roberta-large

tid=10101

nohup python -u server.py \
    --num_client $num_client --data_name $data_name --rank 4 \
    --num_rounds $num_rounds --client_ckpt $model \
    > outputs/${tid}.out 2>&1 &

for device in 1 2 3 4
do
    nohup python -u lora-client.py \
        --num_client $num_client --data_name $data_name --rank 4 \
        --num_rounds $num_rounds --client_ckpt $model \
        --device $device \
        > outputs/client/${tid}_${device}.out 2>&1 &
done
