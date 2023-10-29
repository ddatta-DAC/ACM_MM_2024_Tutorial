import argparse
import os
from time import time

import evaluate
import lightning as pl
import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
import transformers
from colorama import Back, Fore, Style
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import Linear, Sequential
from torch.nn import functional as TF
from torch.optim import SGD, AdamW, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import F1Score, MulticlassAccuracy
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

accuracy = evaluate.load("accuracy")
os.environ["TOKENIZERS_PARALLELISM"] = "True"

np.random.seed(100)


def print_info(msg):
    print(Back.BLUE + Fore.YELLOW + msg)
    print(Style.RESET_ALL)


def write_to_file(result: dict, result_path="results_v1.csv"):
    df = None
    if os.path.exists(result_path):
        df = pd.read_csv(result_path, index_col=None)
    _df = pd.DataFrame(result, index=[0])
    if df is None:
        df = _df
    else:
        df = pd.concat([df, _df])
    df.to_csv(result_path, index=False)
    print_info(f"Results recorded | {result['model_id']}")
    return


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def get_data(tokenizer_obj, train_size=5000):
    """
    Create a subset of data
    """

    def preprocess_function(examples):
        return tokenizer_obj(examples["text"], truncation=True)
    
    rng = np.random.default_rng(12345)
    val_size = train_size // 10
    test_size = train_size // 10

    dataset_dict = load_dataset("imdb")
    
    idx = rng.choice(
        np.arange(len(dataset_dict["train"])), train_size + val_size, replace=False
    )
    idx_train = idx[:train_size]
    idx_val = idx[train_size:]

    train_dataset = dataset_dict["train"].select(idx_train)
    val_dataset = dataset_dict["train"].select(idx_val)
    
    idx = rng.choice(
        np.arange(len(dataset_dict["test"])), test_size, replace=False
    )
    dataset_dict["test"] = dataset_dict["test"].select(idx)
    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "validation": val_dataset,
            "test": dataset_dict["test"],
        }
    )

    dataset_dict = dataset_dict.map(preprocess_function, batched=True)
    return dataset_dict


def main(
    model_id: str,
    max_epochs: int = 1,
    learning_rate=1e-5,
    train_batch_size: int = 1,
    eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 0.005,
):
    """
    Main function
    """
    print_info(
        model_id + f"  Epochs: {max_epochs}" +  f"LR: {learning_rate}" + f"Weight Decay: {weight_decay}" + f"Grad Acc: {gradient_accumulation_steps}"
    )

    tokenizer_obj = AutoTokenizer.from_pretrained(model_id)
    dataset_dict = get_data(tokenizer_obj)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer_obj)
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=max_epochs,
        weight_decay=0.01,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer_obj,
    )
    t1 = time()
    trainer.train()
    t2 = time()
    training_time = t2 - t1
    results = trainer.predict(
        test_dataset=dataset_dict["test"],
    )
    y_pred = np.argmax(results.predictions, axis=-1)
    y_true = dataset_dict["test"]["label"]
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    result = {
        "model_id": model_id,
        "num_epochs": max_epochs,
        "train_batch_size": train_batch_size,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "weight_decay": weight_decay,
        "train_time": training_time,
        "accuracy": acc,
        "f1": f1,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="distilbert-base-uncased")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)

    args = parser.parse_args()
    model_id = args.model_id
    gradient_accumulation_steps = args.gradient_accumulation_steps
    weight_decay = args.weight_decay
    train_batch_size = args.train_batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    
    result = main(
        model_id=model_id,
        max_epochs=num_epochs,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
    )
    write_to_file(result)