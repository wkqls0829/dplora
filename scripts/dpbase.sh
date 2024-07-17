#!/bin/bash

num_client=4
data_name=sst2
lora_r=64
num_rounds=50
client_epochs=1
model=distilbert-base-uncased
mode=dplora
projection_type="fixed"

tid=40001

nohup python -u server.py \
    --num_client $num_client --data_name $data_name --rank 0 \
    --num_rounds $num_rounds --client_epochs $client_epochs --client_ckpt $model \
    --mode $mode --lora_r $lora_r \
    > outputs/${tid}.out 2>&1 &

for client in 1 2 3 4
do
    device=$((client + 3))
    local_r=8
    nohup python -u lora-client.py \
        --num_client $num_client --data_name $data_name --rank $client \
        --num_rounds $num_rounds --client_epochs $client_epochs --client_ckpt $model \
        --mode $mode --lora_r $lora_r --local_r $local_r \
        --device $device --projection_type $projection_type \
        > outputs/client/${tid}_${client}.out 2>&1 &
done
