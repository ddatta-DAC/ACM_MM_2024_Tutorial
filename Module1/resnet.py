"""
ResNet Model for Image Classification
"""

import torch
import os
import sys
from glob import glob
from typing import *

import lightning as pl
import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from torch.nn import Linear, Sequential
from torch.nn import functional as TF
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ResNetModel
from torch.optim import SGD, AdamW
from colorama import Fore, Back, Style
from time import time 

def print_info(msg):
    print(Back.BLUE + Fore.YELLOW  + msg)
    print(Style.RESET_ALL)
    
def verify_image(path):
    """
    Ensure image path points to a valid image file
    """
    try:
        Image.open(path)
    except IOError:
        return False
    return True


def get_data_split(split="train"):
    """
    Read data b split name
    """
    assert split in ["train", "test", "validation"]
    transform = transforms.Compose(
        [
            transforms.Resize(
                [224, 224]
            ),  # Resizing the image as the VGG only take 224 x 244 as input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = ImageFolder(root=f"./CIFAR_10_subset/{split}", is_valid_file=verify_image, transform=transform)
    return ds


class CIFAR_Module(pl.LightningModule):
    def __init__(self, model_obj, num_classes, optimizer_name="AdamW", optimizer_hparams=None):
        """
        Inputs:
            model_obj - The pre
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        self.model = model_obj
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.acc_metric = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, inputs):
        return model_obj(inputs)

    def configure_optimizers(self):
        if self.optimizer_name == "AdamW":
            optimizer = AdamW(self.parameters(), **self.optimizer_hparams)
        else:
            optimizer = SGD(self.parameters(), **self.optimizer_hparams)

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self(inputs)
        acc = self.acc_metric(preds, labels)
        loss = TF.cross_entropy(preds, labels.view(-1))
        self.log("train_loss", loss)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self(inputs)
        acc = self.acc_metric(preds, labels)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self(inputs)
        acc = self.acc_metric(preds, labels)
        self.log("test_acc", acc)


def get_ResNet(num_classes=2):
    """
    Obtain a pre-trained model from Pytorch hub
    """
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", weights="ResNet18_Weights.DEFAULT")
    for param in model.parameters():
        param.requires_grad = False

    fc = list(model.children())[-1:]
    inp_features = fc[0].in_features
    model.fc = Linear(inp_features, num_classes)
    return model


"""
Main execution code
"""

if __name__ == "__main__":
    num_classes = 10
    batch_size = 16
    num_epochs = 10
    model_obj = get_ResNet(num_classes)
    print_info('Model Initialized')
    print_info(f'Number of classes {num_classes}')
    
    pl_model_container = CIFAR_Module(
        model_obj=model_obj,
        num_classes=10,
        optimizer_name="AdamW",
        optimizer_hparams={"lr": 0.0001, "weight_decay": 0.001},
    )
    
    ds_train = get_data_split(split="train")
    ds_val = get_data_split(split="validation")
    print_info(f'Length of training set {len(ds_train)}')
    print_info(f'Length of validation set {len(ds_val)}')
    print_info(f'Batch Size {batch_size}')
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=20)
    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=20)
    
    print_info('Start of training')
    trainer = pl.Trainer(devices="auto", strategy="ddp_notebook", accelerator="gpu", max_epochs=num_epochs)
    t1 = time()
    trainer.fit(model=pl_model_container, train_dataloaders=dl_train, val_dataloaders=dl_val)
    t2 = time()
    print_info('End of training')
    print_info( f'Time taken:: {t2-t1} seconds')