from logsy import LogsyRunner
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler

torch.random.seed = 0
np.random.seed(0)
from datasets import *
from tokenizer import LogTokenizer
from utils import get_padded_data
from model.loss_function import SimpleLossCompute
from model.model import LogsyModel
from model.trainer import run_train, run_test
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

aux_size = 300000
bgl_dataset = BGLDataset(aux=True, aux_size=aux_size)
every_n = 100
aux_size = 100000
spirit_dataset = SpiritDataset(every_n=every_n, aux_size=aux_size,aux=True)
intrepid_dataset = InterpidScrubbedDataset(aux=True)
max_lines = 5000000
tbird_dataset = ThunderbirdDataset(max_lines=max_lines)

dataset = tbird_dataset
aux_datasets = [bgl_dataset, spirit_dataset, intrepid_dataset]

model_params = {
    "tgt_vocab": 2,
    "n_layers": 2,
    "in_features": 16,
    "out_features": 16,
    "num_heads": 2,
    "dropout": 0.05,
    "max_len": 50

}

split_size = 0.8
batch_size = 2048
pad_len = 50
loss_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 1.0]).cuda())  #
model_path = '../output/models/'
epochs = 30

logsy_run = LogsyRunner(dataset=dataset,
                       aux_datasets=aux_datasets,
                       model_params=model_params,
                       split_size=split_size,
                       batch_size=batch_size,
                       pad_len=pad_len,
                       loss_criterion=loss_criterion,
                       model_path=model_path,
                       epochs=epochs)

logsy_run.run_logsy()