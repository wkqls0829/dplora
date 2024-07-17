import warnings

import flwr as fl
import torch
import torch.nn as nn

from evaluate import load as load_metric

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
from arguments import parse_args
from data_loader import load_data

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

args = parse_args()
RANK = args.rank
NUM_CLIENTS = args.num_clients
NUM_SPLITS = NUM_CLIENTS + 1  # teacher also has a split

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print(f"current device is {device}")

DEVICE = torch.cuda.current_device()
CHECKPOINT = args.client_ckpt

def train(net, trainloader, epochs, lr):
    optimizer = AdamW(net.parameters(), lr=lr, no_deprecation_warning=True)
    net.train()
    if args.mode in ["ffalora", "dplora"]:
        for name, module in net.named_modules():
            if "_A" in name:
                module.requires_grad = False



    criterion = nn.NLLLoss()

    for i in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            
            if args.mode == "hetlora":
                loss = HetLoRALoss(net, criterion, outputs, batch["labels"])
            else:
                loss = outputs.loss


            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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



def test(net, testloader):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy

def main():
    logging.info(f"Started client {RANK}")
    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    ).to(DEVICE)

    peft_config = LoraConfig(
        task_type="SEQ_CLS", 
        inference_mode=False, 
        target_modules=["q_lin", "v_lin"],
        r=args.lora_r, 
        lora_alpha=16, 
        lora_dropout=0.1)


    net = get_peft_model(net, peft_config)

    net.print_trainable_parameters()


    trainloader, testloader = load_data(args.data_path, args.data_name, RANK, NUM_SPLITS, CHECKPOINT, args.teacher_data_pct)
    peft_state_dict_keys = get_peft_model_state_dict(net).keys()
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
            logging.info(f"Client {RANK} Training Started...")
            train(net, trainloader, epochs=args.client_epochs, lr=args.client_lr)
            return self.get_parameters(), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), len(testloader), {"accuracy": float(accuracy)}

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
            logging.info(f"Client {RANK} Training Started...")
            train(net, trainloader, epochs=args.client_epochs, lr=args.client_lr)
            return self.get_parameters(), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), len(testloader), {"accuracy": float(accuracy)}

    class SVDLoRA_Client(fl.client.NumPyClient):
        def get_parameters(self, config=None):
            state_dict = get_peft_model_state_dict(net)
            
            delta_w = []
            for k, v in state_dict.items():
                if "lora_A" in k:
                    param_A = v[args.local_r:, :]
                elif "lora_B" in k:
                    dw = torch.mm(v[:, args.local_r:], param_A).cpu().numpy()
                    delta_w.append(dw)

                    
            parameters = [val.cpu().numpy() for _, val in state_dict.items()] + delta_w

            return parameters

        def set_parameters(self, parameters):
            params_dict = zip(peft_state_dict_keys, parameters[:len(peft_state_dict_keys)])
            delta_w = parameters[len(peft_state_dict_keys):]
            new_params = []

            for dw in delta_w:
                u, s, v = torch.svd(torch.Tensor(dw).to(device))
                s = torch.diag(s)
                B = torch.mm(u[:, :args.lora_r], s[:args.lora_r,:args.lora_r])
                A = v[:args.lora_r, :]
                new_params.append(A)
                new_params.append(B)

            for i, (k, v) in enumerate(params_dict):
                if "lora" in k:
                    v = new_params[i]
            state_dict = {k: torch.tensor(v).to(device) for k, v in params_dict}
            self.state_dict_mem = state_dict

            set_peft_model_state_dict(net, state_dict)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            logging.info(f"Client {RANK} Training Started...")
            train(net, trainloader, epochs=args.client_epochs, lr=args.client_lr)
            return self.get_parameters(), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), len(testloader), {"accuracy": float(accuracy)}

    class DPLoRA_Client(fl.client.NumPyClient):
        def get_parameters(self, config=None):
            state_dict = get_peft_model_state_dict(net)
            
            try:
                projection_basis = self.get_projection_basis()
            except AttributeError:
                projection_basis = [range(args.local_r) for _ in range(len(peft_state_dict_keys)//2)]
                print(f"parameter not set yet")


            delta_w = []
            i = 0
            for k, v in state_dict.items():
                if "lora_A" in k:
                    param_A = v[projection_basis[i], :]
                elif "lora_B" in k:
                    dw = torch.mm(v[:, projection_basis[i]], param_A).cpu().numpy()
                    delta_w.append(dw)
                    i += 1
        
            parameters = [val.cpu().numpy() for _, val in state_dict.items()] + delta_w

            return parameters

        def get_projection_basis(self):
            projection_basis = []
            if args.projection_type == "fixed":
                fixed_half = args.local_r//2
                projection_basis = [list(range(fixed_half)) + list(range(fixed_half + args.rank-1, fixed_half + args.num_clients*fixed_half, args.num_clients)) for _ in range(len(peft_state_dict_keys)//2)]
                # projection_basis = [range(args.rank - 1,  args.num_clients*args.local_r ,args.client_num) for _ in range(len(peft_state_dict_keys)//2)]
            elif args.projection_type == "diff":
                state_dict = get_peft_model_state_dict(net)
                prev_state_dict = self.state_dict_mem
                if not prev_state_dict:
                    return [range(args.local_r) for _ in range(len(peft_state_dict_keys)//2)]
                for k, v in state_dict.items():
                    if "lora_B" in k:
                        projection_basis.append(list(torch.topk(torch.norm(prev_state_dict[k] - v, dim=1)[:args.lora_r], args.local_r).indices))


            else:
                projection_basis = [range(args.local_r) for _ in range(len(peft_state_dict_keys)//2)]

            print(projection_basis)

            return projection_basis


        def set_parameters(self, parameters):
            params_dict = zip(peft_state_dict_keys, parameters[:len(peft_state_dict_keys)])
            delta_w = parameters[len(peft_state_dict_keys):]
            new_params = []

            for dw in delta_w:
                u, s, v = torch.svd(torch.Tensor(dw).to(device))
                s = torch.diag(s)
                B = torch.mm(u[:, :args.lora_r], s[:args.lora_r,:args.lora_r])
                A = v[:args.lora_r, :]
                new_params.append(A)
                new_params.append(B)

            for i, (k, v) in enumerate(params_dict):
                if "lora" in k:
                    v = new_params[i]
            params_dict = zip(peft_state_dict_keys, parameters[:len(peft_state_dict_keys)])
            state_dict = {k: torch.tensor(v).to(device) for k, v in params_dict}
            self.state_dict_mem = state_dict

            set_peft_model_state_dict(net, state_dict)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            logging.info(f"Client {RANK} Training Started...")
            train(net, trainloader, epochs=args.client_epochs, lr=args.client_lr)
            return self.get_parameters(), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), len(testloader), {"accuracy": float(accuracy)}
    # Start client
    if args.mode == "hetlora":
        print("starting hetlora")
        fl.client.start_numpy_client(server_address="127.0.0.1:8090", client=HetLoRA_Client())
    elif args.mode == "svdlora":
        print("starting svdlora")
        fl.client.start_numpy_client(server_address="127.0.0.1:8090", client=SVDLoRA_Client())
    elif args.mode == "dplora":
        print("starting dplora")
        fl.client.start_numpy_client(server_address="127.0.0.1:8090", client=DPLoRA_Client())
    else:
        print("starting normal lora")
        fl.client.start_numpy_client(server_address="127.0.0.1:8090", client=Client())


if __name__ == "__main__":
    main()
