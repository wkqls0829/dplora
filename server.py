import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from transformers import AutoModelForCausalLM
from arguments import parse_args, pretty_print_args
from output_result import save_run_as_json
import time
from pprint import pprint
import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

from peft import (
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
)

import numpy as np
import datetime
import wandb

args = parse_args()
pretty_print_args(args)
NUM_CLIENTS = args.num_clients
NUM_SPLITS = NUM_CLIENTS + 1

CHECKPOINT = args.client_ckpt
DEVICE = "cpu"

net = AutoModelForCausalLM.from_pretrained(
        args.client_ckpt,
        trust_remote_code=True
    )

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False, 
    #target_modules=["query", "key", "value"],
    target_modules=["q_proj", "v_proj", "k_proj"], #, "o_proj", "gate_proj", "down_proj", "up_proj"],
    r=args.lora_r, 
    lora_alpha=16, 
    lora_dropout=0.1)

net = get_peft_model(net, peft_config)
param_keys = get_peft_model_state_dict(net).keys()

del net

class DPLoRA(fl.server.strategy.FedAvg):

    def configure_fit(self, server_round, parameters, client_manager):

        config = {"round": server_round}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = fl.common.FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, fit_ins) for client in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):

        config = {"round": server_round}

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        evaluate_ins = fl.common.EvaluateIns(parameters, config)
        return [(client_proxy, evaluate_ins) for client_proxy in client_manager.sample(
            num_clients=sample_size
        )]
    
    def aggregate_fit(self, rnd, results, failures):
        aggregated_params = []
        
        if results:
            weights_results = []
            num_examples = []
            for _, fit_res in results:
                weights_results.append(parameters_to_ndarrays(fit_res.parameters))
                num_examples.append(fit_res.num_examples)
            pprint(num_examples)

            # print(len(weights_results))
            # print(len(weights_results[0]))
            
            ts = time.time()

            for i, key in enumerate(param_keys):
                if "lora_A" in key:
                    
                    ### dW = B * A
                    for j, (wr, ne) in enumerate(zip(weights_results, num_examples)):
                        if not j:
                            lora_dW = ne * (np.dot(wr[i + 1], wr[i]))
                        else:
                            lora_dW += ne * (np.dot(wr[i + 1], wr[i]))
                    lora_dW /= sum(num_examples)
                    
                    ### decompose
                    #B_dist, E_dist, A_dist = np.linalg.svd(lora_dW, full_matrices=True)
                    
                    import torch

                    lora_dW_gpu = torch.tensor(lora_dW, device='cuda')
                    U, S, V = torch.svd(lora_dW_gpu)
                    B_dist, E_dist, A_dist = U.cpu().numpy(), S.cpu().numpy(), V.cpu().numpy()

                    aggregated_params.append(A_dist[:args.lora_r, :])
                    aggregated_params.append(B_dist[:, :args.lora_r] * E_dist[:args.lora_r])


                elif ("weight" in key or "bias" in key) and ("lora_B" not in key) :
                    print(key)
                    
                    ### Same as FedAvg
                    for j, (wr, ne) in enumerate(zip(weights_results, num_examples)):
                        if not j:
                            dW = ne * wr[i]
                        else:
                            dW += ne * wr[i]
                    dW /= sum(num_examples)
                    aggregated_params.append(dW)
            
            te = time.time()
            print(te - ts)

            # for i in aggregated_params:
            #     print(i.shape)
        
        return ndarrays_to_parameters(aggregated_params), {}

def weighted_average(metrics):
    global current_round

    # Sort metrics based on client number
    metrics_sorted = sorted(metrics, key=lambda x: x[1]["client"])
    print(metrics_sorted)
    
    total_examples = sum(num_examples for num_examples, _ in metrics_sorted)
    weighted_sums = {"eval_rouge1": 0, "eval_rouge2": 0, "eval_rougeL": 0, "eval_rougeLsum": 0}
    
    for num_examples, m in metrics_sorted:
        for key in weighted_sums.keys():
            weighted_sums[key] += num_examples * m[key]
        client = m["client"]
        dataset_name = str(m["dataset"]).zfill(4)
        print(f"eval_rouge1 of client {client} is {m['eval_rouge1']}")
        print(f"eval_rouge2 of client {client} is {m['eval_rouge2']}")
        print(f"eval_rougeL of client {client} is {m['eval_rougeL']}")
        print(f"eval_rougeLsum of client {client} is {m['eval_rougeLsum']}")
        wandb.log({"Client {} / Dataset {} / Eval Rouge 1".format(client, dataset_name): m['eval_rouge1'],
                   "Client {} / Dataset {} / Eval Rouge 2".format(client, dataset_name): m['eval_rouge2'],
                   "Client {} / Dataset {} / Eval Rouge L".format(client, dataset_name): m['eval_rougeL'],
                   "Client {} / Dataset {} / Eval Rouge Lsum".format(client, dataset_name): m['eval_rougeLsum']},
                   step=current_round)
    
    result = {key: weighted_sums[key] / total_examples for key in weighted_sums}
    wandb.log({"Global Eval Rouge 1": result['eval_rouge1'],
               "Global Eval Rouge 2": result['eval_rouge2'],
               "Global Eval Rouge L": result['eval_rougeL'],
               "Global Eval Rouge Lsum": result['eval_rougeLsum']},
               step=current_round)
    print(result)

    current_round += 1
    
    return {key: weighted_sums[key] / total_examples for key in weighted_sums}

# def weighted_average(metrics):
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]
#     for i in range(len(accuracies)):
#         print(f"accuracy of client i is {accuracies[i] / examples[i]}")

#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}

if __name__ == "__main__":

    ### WandB Logging

    wandb.init(project="DPLoRA_" + args.data_name,
            config=vars(args),
            )
    wandb.run.name = args.tid + "_" + datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%T")
    wandb.config.update({"task_id": args.tid})

    current_round = 0

    # from flwr.common import GRPC_MAX_MESSAGE_LENGTH
    # GRPC_MAX_MESSAGE_LENGTH = 2000000000
    
    # Define strategy
    if args.mode == "dplora":
        # strategy = DPLoRA( #fl.server.strategy.FedAvg(
        #     fraction_fit=1.0,
        #     fraction_evaluate=1.0,
        #     evaluate_metrics_aggregation_fn=weighted_average,
        #     min_available_clients=args.num_clients,
        #     min_fit_clients=args.num_clients,
        # )
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            evaluate_metrics_aggregation_fn=weighted_average,
            min_available_clients=args.num_clients,
            min_fit_clients=args.num_clients,
            )
    else:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            evaluate_metrics_aggregation_fn=weighted_average,
            min_available_clients=args.num_clients,
            min_fit_clients=args.num_clients,
            )

    # Start server
    t1 = time.perf_counter()
    history = fl.server.start_server(
        server_address="0.0.0.0:8090",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds), #, GRPC_MAX_MESSAGE_LENGTH=2000000000),
        strategy=strategy,
        grpc_max_message_length = 2000000000,
    )
    t2 = time.perf_counter()
    extra_data = dict(
        elapsed_time_secs=t2 - t1
    )
    save_run_as_json(args, history, extra_data=extra_data)
    
