import flwr as fl
from arguments import parse_args, pretty_print_args
from output_result import save_run_as_json
import time
import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

import datetime
import wandb

args = parse_args()
pretty_print_args(args)
NUM_CLIENTS = args.num_clients
NUM_SPLITS = NUM_CLIENTS + 1

def weighted_average(metrics):
    global current_round

    # Sort metrics based on client number
    metrics_sorted = sorted(metrics, key=lambda x: x[1]["client"])
    
    total_examples = sum(num_examples for num_examples, _ in metrics_sorted)
    weighted_sums = {"eval_rouge1": 0, "eval_rouge2": 0, "eval_rougeL": 0, "eval_rougeLsum": 0}
    
    for num_examples, m in metrics_sorted:
        for key in weighted_sums.keys():
            weighted_sums[key] += num_examples * m[key]
        client = m["client"]
        print(f"eval_rouge1 of client {client} is {m['eval_rouge1']}")
        print(f"eval_rouge2 of client {client} is {m['eval_rouge2']}")
        print(f"eval_rougeL of client {client} is {m['eval_rougeL']}")
        print(f"eval_rougeLsum of client {client} is {m['eval_rougeLsum']}")
        wandb.log({"Client {} Eval Rouge 1".format(client): m['eval_rouge1'],
                   "Client {} Eval Rouge 2".format(client): m['eval_rouge2'],
                   "Client {} Eval Rouge L".format(client): m['eval_rougeL'],
                   "Client {} Eval Rouge Lsum".format(client): m['eval_rougeLsum']},
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

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=weighted_average,
        min_available_clients=args.num_clients,
    )

    # Start server
    t1 = time.perf_counter()
    history = fl.server.start_server(
        server_address="0.0.0.0:8090",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
    t2 = time.perf_counter()
    extra_data = dict(
        elapsed_time_secs=t2 - t1
    )
    save_run_as_json(args, history, extra_data=extra_data)
