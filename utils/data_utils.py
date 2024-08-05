import random
import numpy as np
import os
import torch
import peft
from tqdm import tqdm

# def tokenize(tokenizer, prompt, cutoff_len=512, add_eos_token=True):
#     result = tokenizer(
#         prompt,
#         truncation=True,
#         max_length=cutoff_len,
#         padding=True,
#         return_tensors=None,
#     )
#     if (
#             result["input_ids"][-1] != tokenizer.cls_token_id
#             and len(result["input_ids"]) < cutoff_len
#             and add_eos_token
#     ):
#         result["input_ids"].append(tokenizer.cls_token_id)
#         result["attention_mask"].append(1)
#         result["token_type_ids"].append(0)

#     result["labels"] = result["input_ids"].copy()

#     return result

def tokenize(tokenizer, prompt, cutoff_len=512, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=True,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
        #result["token_type_ids"].append(0)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point, prompter, tokenizer):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(tokenizer, full_prompt, cutoff_len=512, add_eos_token=True)
    return tokenized_full_prompt

