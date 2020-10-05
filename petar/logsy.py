import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import os
from tqdm import trange


from tokenizer import LogTokenizer
from utils import get_padded_data
from model.loss_function import SimpleLossCompute
from model.model import LogsyModel
from model.trainer import run_train, run_test

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

torch.random.seed = 0
np.random.seed(0)

torch.cuda.set_device(0)


class LogsyRunner:
    def __init__(self, dataset, aux_datasets=None, model_params=None, split_size=0.8, batch_size=2048, pad_len=50,
                 loss_criterion=None, model_path='../output/models/', epochs=30):

        self.load_test = self.log_payload = self.true_labels = self.load_train = \
            self.labels_test = self.labels_train = self.max_auc = self.max_distances = None
        self.src_vocab = self.model = self.preds = None

        self.dataset = dataset
        self.aux_datasets = aux_datasets or []
        self.split_size = split_size
        self.dataset_size = 0  # this value will be filled by _init_data

        # data loaders
        self.batch_size = batch_size
        self.pad_len = pad_len

        #
        self.epochs = epochs

        self.model_params = model_params or {
            "tgt_vocab": 2,
            "n_layers": 2,
            "in_features": 16,
            "out_features": 16,
            "num_heads": 2,
            "dropout": 0.05,
            "max_len": 50

        }

        if loss_criterion is None:
            raise Exception("Please provide a loss function")
        self.optimizer = None
        self.loss_criterion = loss_criterion
        self.model_path = model_path

    def run_logsy(self):
        print("[*] Initializing dataset")
        self.log_payload, self.true_labels = self.init_data(self.dataset, self.aux_datasets)

        # tokenized payload
        print("[*] Tokenizing data")
        self.log_payload = self.tokenize_data(self.log_payload)
        # split dataset
        print("[*] Splitting dataset")
        self.load_train, self.labels_train, self.load_test, self.labels_test = self.train_test_split(self.log_payload,
                                                                                                     self.true_labels)

        print("[*] Creating data loaders")
        train_dataloader, test_dataloader = self.create_data_loaders(self.load_train, self.labels_train, self.load_test,
                                                                     self.labels_test)
        print("[*] Running optimizer")
        self.run_optimizer(train_dataloader, test_dataloader)

    def run_optimizer(self, train_dataloader, test_dataloader):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.001)

        max_auc = 0.0
        max_distances = 0
        best_preds = None
        for epoch in range(self.epochs):
            self.model.train()
            print("Epoch", epoch)
            run_train(train_dataloader, self.model,
                      SimpleLossCompute(self.model, self.loss_criterion, self.optimizer), step_size=100)
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            torch.save(self.model.state_dict(), os.path.join(self.model_path, 'model_' + str(epoch) + '.pt'))

            # test
            self.model.eval()
            preds, distances = run_test(test_dataloader, self.model,
                                        SimpleLossCompute(self.model, self.loss_criterion, None, is_test=True),
                                        step_size=100)

            preds = np.array(preds)
            auc = roc_auc_score(self.labels_test.astype(np.int32), distances)
            print("AUC:", auc)
            if auc > max_auc:
                best_preds = preds
                max_auc = auc
                fpr, tpr, thresholds = roc_curve(self.labels_test.astype(np.int32), distances, pos_label=1)
                np.save(str((len(self.log_payload) - self.dataset_size)) + '_without8020.npy', [fpr, tpr, thresholds])
                print(roc_auc_score(self.labels_test.astype(np.int32), distances))
                max_distances = distances

        self.max_distances = max_distances
        self.max_auc = max_auc
        self.preds = best_preds

    def init_data(self, dataset, aux_datasets):
        log_payload, true_labels = dataset.get_data()
        self.dataset_size = len(log_payload)

        for i in range(len(aux_datasets)):
            normal, anomalies = aux_datasets[i].get_data()
            log_payload = np.append(log_payload.reshape(-1, 1), anomalies.reshape(-1, 1), axis=0)

        true_labels = np.append(true_labels, np.ones(len(log_payload) - self.dataset_size).reshape(-1, 1),
                                axis=0).flatten()
        return log_payload, true_labels.flatten()

    def tokenize_data(self, log_payload):
        tokenizer = LogTokenizer()
        data_tokenized = []
        for i in trange(0, len(log_payload)):
            tokenized = tokenizer.tokenize(log_payload[i][0])
            data_tokenized.append(tokenized)

        # need for model creation
        self.src_vocab = tokenizer.n_words
        self.model = LogsyModel(src_vocab=self.src_vocab, **self.model_params).get_model()
        torch.cuda.set_device(0)
        self.model.cuda()
        return np.asanyarray(data_tokenized)

    def train_test_split(self, log_payload, true_labels):

        train_size = round(self.dataset_size * self.split_size)

        load_test = log_payload[train_size:self.dataset_size]
        load_train = np.append(log_payload[:train_size][true_labels[:train_size] == 0],
                               log_payload[self.dataset_size:], axis=0)

        labels_test = true_labels[train_size:self.dataset_size]
        labels_train = np.append(true_labels[:train_size][true_labels[:train_size] == 0].flatten(),
                                 true_labels[self.dataset_size:].flatten(), axis=0)

        return load_train, labels_train, load_test, labels_test

    def create_data_loaders(self, load_train, labels_train, load_test, labels_test):
        # transform_to_tensor = transforms.Lambda(lambda lst: torch.tensor(lst))
        train_data = TensorDataset(torch.tensor(get_padded_data(load_train, pad_len=self.pad_len), dtype=torch.long),
                                   torch.tensor(labels_train.astype(np.int32), dtype=torch.long))
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        test_data = TensorDataset(torch.tensor(get_padded_data(load_test, pad_len=self.pad_len), dtype=torch.long),
                                  torch.tensor(labels_test.astype(np.int32).flatten(), dtype=torch.long))
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size)
        return train_dataloader, test_dataloader
