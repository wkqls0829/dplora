#!/bin/bash

num_client=2
data_name=sst2
lora_r=8
num_rounds=10
client_epochs=10
model=distilbert-base-uncased
mode=svdlora

tid=39000

nohup python -u server.py \
    --num_client $num_client --data_name $data_name --rank 0 \
    --num_rounds $num_rounds --client_epochs $client_epochs --client_ckpt $model \
    --mode $mode --lora_r $lora_r \
    > outputs/${tid}.out 2>&1 &

for client in 1 2
do
    device=$((client + 5))
    local_r=$((client * 4))
    nohup python -u lora-client.py \
        --num_client $num_client --data_name $data_name --rank $client \
        --num_rounds $num_rounds --client_epochs $client_epochs --client_ckpt $model \
        --mode $mode --lora_r $lora_r --local_r $local_r \
        --device $device \
        > outputs/client/${tid}_${client}.out 2>&1 &
done
