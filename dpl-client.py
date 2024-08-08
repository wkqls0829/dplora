import warnings

import flwr as fl
import torch
import torch.nn as nn
import os
import numpy as np

from copy import deepcopy

import evaluate
from evaluate import load as load_metric

import datasets
from datasets import load_dataset
from utils.prompter import Prompter
from utils.data_utils import tokenize

from peft import (
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
)

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

import transformers
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer
from arguments import parse_args
from data_loader import load_data

datasets.utils.logging.set_verbosity_error()

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

os.environ['WANDB_DISABLED'] = 'true'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

args = parse_args()
RANK = args.rank
NUM_CLIENTS = args.num_clients
NUM_SPLITS = NUM_CLIENTS + 1  # teacher also has a split

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print(f"current device is {device}")

DEVICE = torch.cuda.current_device()
CHECKPOINT = args.client_ckpt

# wandb.init(project="DPLoRA_" + args.data_name,
#             config=vars(args),
#             )
# wandb.run.name = args.tid + "_" + datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%T")
# wandb.config.update({"task_id": args.tid})

# def train(net, trainloader, epochs, lr):
#     optimizer = AdamW(net.parameters(), lr=lr, no_deprecation_warning=True)
#     net.train()
#     if args.mode in ["ffalora", "dplora"]:
#         for name, module in net.named_modules():
#             if "_A" in name:
#                 module.requires_grad = False
#                 module.eval()


#     criterion = nn.NLLLoss()

#     for i in range(epochs):
#         for batch in trainloader:
#             batch = {k: v.to(DEVICE) for k, v in batch.items()}
#             outputs = net(**batch)
#             loss = outputs.loss
            
#             if args.mode == "hetlora":
#                 loss = HetLoRALoss(net, criterion, outputs, batch["labels"])
#             else:
#                 loss = outputs.loss


#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()

def HetLoRALoss(net, criterion, outputs, target):
    loss = outputs.loss
    for name, module in net.named_modules():
        if "_A" in name:
            for params in module.parameters():
                loss += args.lda * torch.norm(params[int(args.local_r // args.gamma):args.local_r, :])
        elif "_B" in name:
            for params in module.parameters():
                loss += args.lda * torch.norm(params[:, int(args.local_r // args.gamma):args.local_r])
    return loss



#def test(net, testloader):
#    metric = load_metric("accuracy")
#    loss = 0
#    net.eval()
#    for batch in testloader:
#        
#        batch = {k: v.to(DEVICE) for k, v in batch.items()}
#        with torch.no_grad():
#            outputs = net(**batch)
#        logits = outputs.logits
#        loss += outputs.loss.item()
#        predictions = torch.argmax(logits, dim=-1)
#        metric.add_batch(predictions=predictions, references=batch["labels"])
#    loss /= len(testloader.dataset)
#    accuracy = metric.compute()["accuracy"]
#    return loss, accuracy

def build_local_trainer(net,
                        local_train_dataset,
                        local_eval_dataset,
                        optim,
                        tokenizer,
                        local_micro_batch_size,
                        gradient_accumulation_steps,
                        local_num_epochs,
                        local_learning_rate,
                        group_by_length,
                        warmup=0,
                        density=None,
                        lambd=None,
                        reg=None):
    class reg_Trainer(transformers.Trainer):
        def compute_loss(net, inputs, return_outputs=False):
            outputs = net(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            regularizer = 0
            count = 0
            loss += lambd * regularizer
            return (loss, outputs) if return_outputs else loss

    # def compute_metrics(pred):
    #     labels_ids = pred.label_ids
    #     pred_ids = np.argmax(pred.predictions,axis=-1)
    #     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    #     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    #     rouge = evaluate.load('./evaluate/metrics/rouge/rouge.py')
    #     rouge_output = rouge.compute(predictions=pred_str, references=label_str, use_aggregator=True)
    #     return {
    #         'rouge1': round(rouge_output["rouge1"], 4),
    #         'rouge2': round(rouge_output["rouge2"], 4),
    #         'rougeL': round(rouge_output["rougeL"], 4),
    #         'rougeLsum': round(rouge_output["rougeLsum"], 4),
    #     }

    def compute_metrics(pred):
        pred_ids, labels_ids = pred
        labels_ids = np.where(labels_ids != -100, labels_ids, tokenizer.pad_token_id)
        pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
        pred_ids = np.argmax(pred_ids, axis=-1)
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        rouge = evaluate.load('./evaluate/metrics/rouge/rouge.py')
        rouge_output = rouge.compute(predictions=pred_str, references=label_str, use_aggregator=True)
        return {
            'rouge1': round(rouge_output["rouge1"], 4),
            'rouge2': round(rouge_output["rouge2"], 4),
            'rougeL': round(rouge_output["rougeL"], 4),
            'rougeLsum': round(rouge_output["rougeLsum"], 4),
        }
    
    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=local_micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup,
        num_train_epochs=local_num_epochs,
        learning_rate=local_learning_rate,
        do_train=True,
        do_eval=True,
        logging_steps=1,
        optim=optim,
        evaluation_strategy="epoch",
        save_strategy="no",
        output_dir=args.output_dir,
        ddp_find_unused_parameters=False,
        group_by_length=False,
        dataloader_drop_last=False,
        report_to="none"
    )
    local_trainer = transformers.Trainer(model=net,
                                         train_dataset=local_train_dataset,
                                         eval_dataset=local_eval_dataset,
                                         args=train_args,
                                         data_collator=transformers.DataCollatorForSeq2Seq(
                                             tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                                         ),
                                         compute_metrics=compute_metrics,
                                        )
    return local_trainer


def test(net, tokenizer, test_data, epoch, local_micro_batch_size):
    test_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        do_train=False,
        do_eval=True,
        # fp16=True,
        per_device_eval_batch_size=local_micro_batch_size,
        dataloader_drop_last=False,
        eval_accumulation_steps=4,
    )

    def compute_metrics(pred):
        pred_ids, labels_ids = pred
        labels_ids = np.where(labels_ids != -100, labels_ids, tokenizer.pad_token_id)
        pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
        pred_ids = np.argmax(pred_ids, axis=-1)
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        rouge = evaluate.load('./evaluate/metrics/rouge/rouge.py')
        rouge_output = rouge.compute(predictions=pred_str, references=label_str, use_aggregator=True)


        return {
            'rouge1': round(rouge_output["rouge1"], 4),
            'rouge2': round(rouge_output["rouge2"], 4),
            'rougeL': round(rouge_output["rougeL"], 4),
            'rougeLsum': round(rouge_output["rougeLsum"], 4),
        }

    # def compute_metrics(pred):
    #     labels_ids = pred.label_ids
    #     labels_ids[labels_ids == -100] = 1829
    #     # pred_ids = pred.predictions
    #     pred_ids = np.argmax(pred.predictions, axis=-1)
    #     # all unnecessary tokens are removed
    #     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    #     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    #     rouge = evaluate.load('./evaluate/metrics/rouge/rouge.py')
    #     rouge_output = rouge.compute(predictions=pred_str, references=label_str, use_aggregator=True)
    #     return {
    #         'rouge1': round(rouge_output["rouge1"], 4),
    #         'rouge2': round(rouge_output["rouge2"], 4),
    #         'rougeL': round(rouge_output["rougeL"], 4),
    #         'rougeLsum': round(rouge_output["rougeLsum"], 4)
    #     }

    # init trainer
    tester = transformers.Trainer(
        model=net,
        args=test_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=compute_metrics
    )
    # test_dataset = self.test_data["train"].shuffle().map(self.generate_and_tokenize_prompt)
    # test_dataset = self.local_test_dataset
    eval_dataset = test_data
    # test_results = tester.evaluate(test_dataset)
    eval_results = tester.evaluate(eval_dataset)
    # logging.info('For client ' + str( self.client_id) + ', the test result is:')
    # logging.info(test_results)
    print('For client ' + str(RANK) + ', the eval result is:')
    print(eval_results)
    return eval_results



def main():
    logging.info(f"Started client {RANK}")
    net = AutoModelForCausalLM.from_pretrained(
        args.client_ckpt,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"
    # net = AutoModelForSequenceClassification.from_pretrained(
        # CHECKPOINT, num_labels=2
    # ).to(DEVICE)
    prompter = Prompter()
    

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False, 
        #target_modules=["query", "key", "value"],
        target_modules=["q_proj", "v_proj", "k_proj"], #, "o_proj", "gate_proj", "down_proj", "up_proj"],
        r=args.lora_r, 
        lora_alpha=16, 
        lora_dropout=0.1)


    net = get_peft_model(net, peft_config)

    net.print_trainable_parameters()

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(tokenizer, full_prompt, cutoff_len=512, add_eos_token=True)
        return tokenized_full_prompt

    data_names = os.listdir(args.data_path)
    
    if "_" in args.data_name:
        args.data_name, args.filter = args.data_name.split("_")
    
    for dn in data_names:
        if "training_" + args.data_name + ".json" in dn:
            train_data_name = dn
        if "eval_" + args.data_name + ".json" in dn:
            eval_data_name = dn
            test_data_name = dn
        if "test_" + args.data_name + ".json" in dn:
            test_data_name = dn

    print(f"loading data {train_data_name}")

    train_path = os.path.join(args.data_path, train_data_name)
    eval_path = os.path.join(args.data_path, eval_data_name)
    test_path = os.path.join(args.data_path, test_data_name)
    
    if args.filter is not None:
        if args.filter == "0":
            train_data = load_dataset("json", data_files=train_path, split="train[:50%]")
            eval_data = load_dataset("json", data_files=eval_path, split="train[:50%]")
            test_data = load_dataset("json", data_files=test_path, split="train[:50%]")
            # train_data["train"] = train_data["train"][:len(train_data)]
            # eval_data["train"] = eval_data["train"][:len(train_data)]
            # test_data["train"] = test_data["train"][:len(train_data)]
        elif args.filter == "1":
            train_data = load_dataset("json", data_files=train_path, split="train[50%:]")
            eval_data = load_dataset("json", data_files=eval_path, split="train[50%:]")
            test_data = load_dataset("json", data_files=test_path, split="train[50%:]")
        else:
            raise ValueError("More than 2 splits are not available.")
    else:
        train_data = load_dataset("json", data_files=train_path, split="train")
        eval_data = load_dataset("json", data_files=eval_path, split="train")
        test_data = load_dataset("json", data_files=test_path, split="train")

    # loaded_data = load_dataset("json", data_files=train_path, field=["Categories", "Definition", "Input_language", "Output_language", "Positive Examples"])
    # print(f'loading data from {loaded_data["Categories"]} with language {loaded_data["Input_language"]} to {loaded_data["Output_language"]}')
    # train_data = loaded_data["Positive Examples"]
    # train_data["instruction"] = loaded_data["Definition"] 
    # train_eval_data = train_data["train"].train_test_split(test_size=0.2)
    train_data = train_data.shuffle().map(generate_and_tokenize_prompt)
    eval_data = eval_data.shuffle().map(generate_and_tokenize_prompt)
    test_data = test_data.shuffle().map(generate_and_tokenize_prompt)
    train_data.set_format("torch")
    eval_data.set_format("torch")
    test_data.set_format("torch")


    print("train data is")
    print(train_data)
    print(len(train_data))

    local_trainer=build_local_trainer(net=net,
                                      local_train_dataset=train_data,
                                      local_eval_dataset=eval_data,
                                      optim="adamw_torch",
                                      tokenizer=tokenizer,
                                      local_micro_batch_size=args.micro_batch_size,
                                      gradient_accumulation_steps=args.batch_size//args.micro_batch_size,
                                      local_num_epochs=args.client_epochs,
                                      local_learning_rate=args.client_lr,
                                      group_by_length=False,
                                      warmup=0,
                                     )

    # trainloader, testloader = load_data(args.data_path, args.data_name, RANK, NUM_SPLITS, CHECKPOINT, args.teacher_data_pct)
    peft_state_dict_keys = get_peft_model_state_dict(net).keys()
    non_lora_keys = [k for k in peft_state_dict_keys if "lora" not in k]
    lora_keys = [k for k in peft_state_dict_keys if "lora" in k]


    # Flower client
    class Client(fl.client.NumPyClient):
        def get_parameters(self, config=None):
            state_dict = get_peft_model_state_dict(net)
            return [val.cpu().numpy() for _, val in state_dict.items()]

        def set_parameters(self, parameters):
            params_dict = zip(peft_state_dict_keys, parameters)
            state_dict = {k: torch.Tensor(v) for k, v in params_dict}
            set_peft_model_state_dict(net, state_dict)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            if args.mode in ["ffalora", "dplora"]:
                for name, module in net.named_modules():
                    if "_A" in name:
                        module.requires_grad = False
                        module.eval()
            local_trainer=build_local_trainer(net=net,
                                              local_train_dataset=train_data,
                                              local_eval_dataset=eval_data,
                                              optim="adamw_torch",
                                              tokenizer=tokenizer,
                                              local_micro_batch_size=args.micro_batch_size,
                                              gradient_accumulation_steps=args.batch_size//args.micro_batch_size,
                                              local_num_epochs=args.client_epochs,
                                              local_learning_rate=args.client_lr,
                                              group_by_length=False,
                                              warmup=0,
                                             )
            logging.info(f"Client {RANK} Training Started...")
            result = local_trainer.train()
            # print(f"trained on {len(train_data)} number of dataset")
            print(local_trainer.state.log_history[-2])
            print(local_trainer.state.log_history[-1])
            print(result.metrics)
            return self.get_parameters(), len(train_data), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            eval_results = test(net, tokenizer, test_data, 1, args.micro_batch_size)
            print(eval_results)
            loss = eval_results["eval_loss"]
            eval_rouge1 = eval_results["eval_rouge1"]
            eval_rouge2 = eval_results["eval_rouge2"]
            eval_rougeL = eval_results["eval_rougeL"]
            eval_rougeLsum = eval_results["eval_rougeLsum"]
            result_dict = {"eval_rouge1": float(eval_results["eval_rouge1"]),
                           "eval_rouge2": float(eval_results["eval_rouge2"]),
                           "eval_rougeL": float(eval_results["eval_rougeL"]),
                           "eval_rougeLsum": float(eval_results["eval_rougeLsum"]),
                           "client": RANK,
                           "dataset": args.data_name}
            return float(loss), len(test_data), result_dict #{"accuracy": float(accuracy)}

    # HetLoRA client
    class HetLoRA_Client(fl.client.NumPyClient):
        def get_parameters(self, config=None):
            state_dict = get_peft_model_state_dict(net)
            try:
                for k, v in state_dict.items():
                    if "lora_A" in k:
                        v[args.local_r:, :] = self.par_mem[k]
                    elif "lora_B" in k:
                        v[:, args.local_r:] = self.par_mem[k]
            except AttributeError:
                print(f"parameter not set yet")

            return [val.cpu().numpy() for _, val in state_dict.items()]

        def set_parameters(self, parameters):
            params_dict = zip(peft_state_dict_keys, parameters)
            state_dict = {k: torch.Tensor(v) for k, v in params_dict}
            self.par_mem = {}
            for k, v in state_dict.items():
                if "lora_A" in k:
                    self.par_mem[k] = v[args.local_r:, :]
                    v[args.local_r:, :] = 0
                elif "lora_B" in k:
                    self.par_mem[k] = v[:, args.local_r:]
                    v[:, args.local_r:] = 0

            set_peft_model_state_dict(net, state_dict)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            local_trainer=build_local_trainer(net=net,
                                              local_train_dataset=train_data,
                                              local_eval_dataset=eval_data,
                                              optim="adamw_torch",
                                              tokenizer=tokenizer,
                                              local_micro_batch_size=args.micro_batch_size,
                                              gradient_accumulation_steps=args.batch_size//args.micro_batch_size,
                                              local_num_epochs=args.client_epochs,
                                              local_learning_rate=args.client_lr,
                                              group_by_length=False,
                                              warmup=0,
                                             )
            logging.info(f"Client {RANK} Training Started...")
            result = local_trainer.train()
            print(f"trained on {len(train_data)} number of dataset")
            print(local_trainer.state.log_history[-2])
            print(local_trainer.state.log_history[-1])
            print(result.metrics)
            return self.get_parameters(), len(train_data), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            eval_results = test(net, tokenizer, test_data, 1, args.micro_batch_size)
            # print(eval_results)
            loss = eval_results["eval_loss"]
            eval_rouge1 = eval_results["eval_rouge1"]
            eval_rouge2 = eval_results["eval_rouge2"]
            eval_rougeL = eval_results["eval_rougeL"]
            eval_rougeLsum = eval_results["eval_rougeLsum"]
            result_dict = {"eval_rouge1": float(eval_results["eval_rouge1"]),
                           "eval_rouge2": float(eval_results["eval_rouge2"]),
                           "eval_rougeL": float(eval_results["eval_rougeL"]),
                           "eval_rougeLsum": float(eval_results["eval_rougeLsum"]),
                           "client": RANK,
                           "dataset": args.data_name}
            return float(loss), len(test_data), result_dict #{"accuracy": float(accuracy)}


    class DPLoRA_Client(fl.client.NumPyClient):
        def __init__(self):
            self.basis = {}

        def get_parameters(self, config=None):
            state_dict = get_peft_model_state_dict(net)

        # if config and config.get("transmit_subset", True):
            try:
                projection_basis = self.get_projection_basis()

            except AttributeError:
                projection_basis = [range(args.local_r) for _ in range(len(peft_state_dict_keys)//2)]
                print(f"parameter not set yet")
            
            i = 0
            for k, v in state_dict.items():
                if "lora_A" in k:
                    state_dict[k] = v.clone()
                    for j in range(v.shape[0]):
                        if j not in projection_basis[i]:
                            state_dict[k][j,:] = 0
                elif "lora_B" in k:
                    state_dict[k] = v.clone()
                    for j in range(v.shape[1]):
                        if j not in projection_basis[i]:
                            state_dict[k][:,j] = 0
                    i += 1
        
            return [val.cpu().numpy() for _, val in state_dict.items()]

        def get_projection_basis(self):
            projection_basis = []
            if args.projection_type == "fixed":
                fixed_half = args.local_r//2
                projection_basis = [list(range(fixed_half)) + list(range(fixed_half + args.rank-1, fixed_half + args.num_clients*fixed_half, args.num_clients)) for _ in range(len(peft_state_dict_keys)//2)]
                # projection_basis = [range(args.rank - 1,  args.num_clients*args.local_r ,args.client_num) for _ in range(len(peft_state_dict_keys)//2)]
            elif args.projection_type == "gradient":
                state_dict = get_peft_model_state_dict(net)
                prev_state_dict = self.state_dict_mem
                if not prev_state_dict:
                    return [range(args.local_r) for _ in range(len(peft_state_dict_keys)//2)]
                for k, v in state_dict.items():
                    if "lora_B" in k:
                        _basis = list(torch.topk(torch.norm(prev_state_dict[k].cpu() - v.cpu(), dim=0)[:args.lora_r], args.local_r).indices)
                        projection_basis.append(_basis)
                        self.basis[k] = _basis #"_".join([str(tensor.item()) for tensor in _basis])
            elif "BA_mag" in args.projection_type:
                state_dict = get_peft_model_state_dict(net)
                prev_state_dict = self.state_dict_mem
                if not prev_state_dict:
                    return [range(args.local_r) for _ in range(len(peft_state_dict_keys)//2)]
                for k, v in state_dict.items():
                    if "lora_A" in k:
                        A_v = v
                    elif "lora_B" in k:
                        dB = prev_state_dict[k].cpu() - v.cpu()
                        dBA_mag = []
                        for i in range(dB.shape[1]):
                            if "1" in args.projection_type:
                                dBA_mag.append(torch.linalg.matrix_norm(dB[:,i].unsqueeze(dim=1) @ A_v[i,:].unsqueeze(dim=0).cpu()), ord=1)
                            if "2" in args.projection_type:
                                dBA_mag.append(torch.linalg.matrix_norm(dB[:,i].unsqueeze(dim=1) @ A_v[i,:].unsqueeze(dim=0).cpu()), ord=2)
                            else:
                                dBA_mag.append(torch.linalg.matrix_norm(dB[:,i].unsqueeze(dim=1) @ A_v[i,:].unsqueeze(dim=0).cpu()))
                        _basis = list(torch.topk(torch.Tensor(dBA_mag), args.local_r).indices)
                        projection_basis.append(_basis)
                        self.basis[k] = _basis #"_".join([str(tensor.item()) for tensor in _basis])



            else:
                projection_basis = [range(args.local_r) for _ in range(len(peft_state_dict_keys)//2)]
            
            # print(projection_basis)
            print(self.basis)

            return projection_basis

        def set_parameters(self, parameters):
            from copy import deepcopy

            params_dict = zip(peft_state_dict_keys, parameters)
            state_dict = {k: torch.Tensor(v) for k, v in params_dict}
            self.state_dict_mem = deepcopy(state_dict)
            set_peft_model_state_dict(net, state_dict)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            if args.projection_type in ("gradient"):
                for name, param in net.named_parameters():
                    if "_A" in name:
                        param.requires_grad = False

            local_trainer=build_local_trainer(net=net,
                                              local_train_dataset=train_data,
                                              local_eval_dataset=eval_data,
                                              optim="adamw_torch",
                                              tokenizer=tokenizer,
                                              local_micro_batch_size=args.micro_batch_size,
                                              gradient_accumulation_steps=args.batch_size//args.micro_batch_size,
                                              local_num_epochs=args.client_epochs,
                                              local_learning_rate=args.client_lr,
                                              group_by_length=False,
                                              warmup=0,
                                             )
            logging.info(f"Client {RANK} Training Started...")
            result = local_trainer.train()
            print(f"trained on {len(train_data)} number of dataset")

            return self.get_parameters(), len(train_data), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            eval_results = test(net, tokenizer, test_data, 1, args.micro_batch_size)
            print(eval_results)
            loss = eval_results["eval_loss"]

            result_dict = {"eval_rouge1": float(eval_results["eval_rouge1"]),
                           "eval_rouge2": float(eval_results["eval_rouge2"]),
                           "eval_rougeL": float(eval_results["eval_rougeL"]),
                           "eval_rougeLsum": float(eval_results["eval_rougeLsum"]),
                           "client": RANK,
                           "dataset": args.data_name,
                           }
            return float(loss), len(test_data), result_dict #{"accuracy": float(accuracy)}


    # Start client
    if args.mode == "hetlora":
        print("starting hetlora")
        fl.client.start_client(server_address="127.0.0.1:8090", client=HetLoRA_Client().to_client())
    elif args.mode == "svdlora":
        print("starting svdlora")
        fl.client.start_client(server_address="127.0.0.1:8090", client=SVDLoRA_Client().to_client())
    elif args.mode == "dplora":
        print("starting dplora")
        fl.client.start_client(server_address="127.0.0.1:8090", client=DPLoRA_Client().to_client())
    elif args.mode == "dplora2":
        print("starting dplora2")
        fl.client.start_client(server_address="127.0.0.1:8090", client=DPLoRA_Client().to_client())
    else:
        print("starting normal lora")
        fl.client.start_client(server_address="127.0.0.1:8090", client=Client().to_client())


if __name__ == "__main__":
    main()
