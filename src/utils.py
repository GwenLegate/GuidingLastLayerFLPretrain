#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# utils for use in the examples and tutorials
import os
import math
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from flsim.data.data_provider import IFLDataProvider, IFLUserData
from flsim.data.data_sharder import FLDataSharder, SequentialSharder
from flsim.interfaces.data_loader import IFLDataLoader
from flsim.interfaces.metrics_reporter import Channel
from flsim.interfaces.model import IFLModel
from tensorboard_metrics_reporter import FLMetricsReporter
from flsim.utils.data.data_utils import batchify
from flsim.utils.simple_batch_metrics import FLBatchMetrics
from torch import nn
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.flowers102 import Flowers102
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertModel
#from torchvision.datasets.stanford_cars import StanfordCars
from torchvision.datasets.eurosat import EuroSAT
from femnist_dataset import FEMNIST
from cub2011_dataset import Cub2011
from stanford_cars import StanfordCars
import ssl
from dirichlet_sharder import DirichletSharder
import random
import wandb
from options import args_parser

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class DistillBERTClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.distill_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.out = torch.nn.Linear(768, 2)

    def forward(self, ids, mask):
        distilbert_output = self.distill_bert(ids, mask)
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        output = self.out(pooled_output)
        return output
        
def tokenization(example):
    return tokenizer(example["text"], padding=True, truncation=True, max_length=30)

def validata_dataset_params(cfg):
    if cfg.dataset.name == 'femnist':
        if cfg.dataset.num_classes != 62:
            cfg.dataset.num_classes = 62
    elif cfg.dataset.name == 'flowers':
        if cfg.dataset.num_classes != 102:
            cfg.dataset.num_classes = 102
    elif cfg.dataset.name == 'cub':
        if cfg.dataset.num_classes != 200:
            cfg.dataset.num_classes = 200
    elif cfg.dataset.name == 'cifar' or cfg.dataset.name == 'eurosat':
        if cfg.dataset.num_classes != 10:
            cfg.dataset.num_classes = 10
    elif cfg.dataset.name == 'cars':
        if cfg.dataset.num_classes != 196:
            cfg.dataset.num_classes = 196
    else:
        raise Exception(f'{cfg.dataset.name} is not a valid dataset')


def collate_fn(batch: Tuple) -> Dict[str, Any]:
    try:
        # HF dataset
        feature = batch['input_ids']
        label = batch['label']
        mask = batch['attention_mask']

        return {"features": feature, "labels": label, 'attention_mask': mask}
    except:
        feature, label = batch
        return {"features": feature, "labels": label, 'attention_mask': None}
    

class DataLoader(IFLDataLoader):
    SEED = 2137
    random.seed(SEED)

    def __init__(
        self,
        train_dataset: VisionDataset,
        eval_dataset: VisionDataset,
        test_dataset: VisionDataset,
        sharder: FLDataSharder,
        batch_size: int,
        drop_last: bool = False,
        collate_fn=collate_fn,
    ):
        assert batch_size > 0, "Batch size should be a positive integer."
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sharder = sharder
        self.collate_fn = collate_fn

    def fl_train_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        rank = kwargs.get("rank", 0)
        world_size = kwargs.get("world_size", 1)
        yield from self._batchify(self.train_dataset, self.drop_last, world_size, rank)

    def fl_eval_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.eval_dataset, drop_last=False)

    def fl_test_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.test_dataset, drop_last=False)

    def _batchify(
        self,
        dataset: VisionDataset,
        drop_last: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ) -> Generator[Dict[str, Generator], None, None]:
        # pyre-fixme[16]: `VisionDataset` has no attribute `__iter__`.
        data_rows: List[Dict[str, Any]] = [self.collate_fn(batch) for batch in dataset]
        for _, (_, user_data) in enumerate(self.sharder.shard_rows(data_rows)):
            batch = {}
            keys = user_data[0].keys()
            for key in keys:
                attribute = {
                    key: batchify(
                        [row[key] for row in user_data],
                        self.batch_size,
                        drop_last,
                    )
                }
                batch = {**batch, **attribute}
            yield batch


class UserData(IFLUserData):
    def __init__(self, user_data: Dict[str, Generator], eval_split: float = 0.0):
        self._train_batches = []
        self._num_train_batches = 0
        self._num_train_examples = 0

        self._eval_batches = []
        self._num_eval_batches = 0
        self._num_eval_examples = 0

        self._eval_split = eval_split

        user_features = list(user_data["features"])
        user_labels = list(user_data["labels"])
        try:
            user_masks = list(user_data["attention_mask"])
        except: 
            user_masks = None
        self.user_labels = user_labels
        total = sum(len(batch) for batch in user_labels)

        for features, labels in zip(user_features, user_labels):
            if self._num_eval_examples < int(total * self._eval_split):
                self._num_eval_batches += 1
                self._num_eval_examples += UserData.get_num_examples(labels)
                self._eval_batches.append(UserData.fl_training_batch(features, labels, user_masks))
            else:
                self._num_train_batches += 1
                self._num_train_examples += UserData.get_num_examples(labels)
                self._train_batches.append(UserData.fl_training_batch(features, labels, user_masks))

    def num_train_examples(self) -> int:
        """
        Returns the number of train examples
        """
        return self._num_train_examples

    def num_eval_examples(self):
        """
        Returns the number of eval examples
        """
        return self._num_eval_examples

    def num_train_batches(self):
        """
        Returns the number of train batches
        """
        return self._num_train_batches

    def num_eval_batches(self):
        """
        Returns the number of eval batches
        """
        return self._num_eval_batches

    def train_data(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterator to return a user batch data for training
        """
        for batch in self._train_batches:
            yield batch

    def eval_data(self):
        """
        Iterator to return a user batch data for evaluation
        """
        for batch in self._eval_batches:
            yield batch

    @staticmethod
    def get_num_examples(batch: List) -> int:
        return len(batch)

    @staticmethod
    def fl_training_batch(
        features: List[torch.Tensor], labels: List[float], user_masks: List[torch.Tensor]=None
    ) -> Dict[str, torch.Tensor]:
        return {"features": torch.stack(features), "labels": torch.Tensor(labels), "attention_mask": user_masks}


class LEAFDataLoader(IFLDataLoader):
    SEED = 2137
    random.seed(SEED)

    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        drop_last: bool = False,
    ):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def fl_train_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.train_dataset, self.drop_last)

    def fl_eval_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.eval_dataset, drop_last=False)

    def fl_test_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.test_dataset, drop_last=False)

    def _batchify(
        self, dataset: Dataset, drop_last=False
    ) -> Generator[Dict[str, Generator], None, None]:
        # pyre-fixme[16]: `Dataset` has no attribute `__iter__`.
        for one_user_inputs, one_user_labels in dataset:
            data = list(zip(one_user_inputs, one_user_labels))
            random.shuffle(data)
            one_user_inputs, one_user_labels = zip(*data)
            batch = {
                "features": batchify(one_user_inputs, self.batch_size, drop_last),
                "labels": batchify(one_user_labels, self.batch_size, drop_last),
            }
            yield batch


class DataProvider(IFLDataProvider):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._train_users = self._create_fl_users(
            data_loader.fl_train_set(), eval_split=0.0
        )
        self._eval_users = self._create_fl_users(
            data_loader.fl_eval_set(), eval_split=1.0
        )
        self._test_users = self._create_fl_users(
            data_loader.fl_test_set(), eval_split=1.0
        )

    def train_user_ids(self) -> List[int]:
        return list(self._train_users.keys())

    def num_train_users(self) -> int:
        return len(self._train_users)

    def get_train_user(self, user_index: int) -> IFLUserData:
        if user_index in self._train_users:
            return self._train_users[user_index]
        else:
            raise IndexError(
                f"Index {user_index} is out of bound for list with len {self.num_train_users()}"
            )

    def train_users(self) -> Iterable[IFLUserData]:
        for user_data in self._train_users.values():
            yield user_data

    def eval_users(self) -> Iterable[IFLUserData]:
        for user_data in self._eval_users.values():
            yield user_data

    def test_users(self) -> Iterable[IFLUserData]:
        for user_data in self._test_users.values():
            yield user_data

    def _create_fl_users(
        self, iterator: Iterator, eval_split: float = 0.0
    ) -> Dict[int, IFLUserData]:
        return {
            user_index: UserData(user_data, eval_split=eval_split)
            # for user_index, user_data in tqdm(
            #     enumerate(iterator), desc="Creating FL User", unit="user"
            # )
            for user_index, user_data in enumerate(iterator)
        }

def get_data_subset(ds_name, num_train, train_dataset, num_classes, num_test=None, test_dataset=None):
    train_idxs = []
    if num_test is not None:
        test_idxs = []
        num_samples = num_train + num_test
    else:
        num_samples = num_train

    for i in range(num_classes):
        if ds_name in ['flowers']:
            idx_for_label = np.where(np.asarray(train_dataset._labels) == i)[0]
        elif ds_name in ['cub', 'cars']:
            idx_for_label = np.where(np.asarray(train_dataset.labels) == i)[0]
        else:
            idx_for_label = np.where(np.asarray(train_dataset.targets) == i)[0]

        selected_idxs = np.random.choice(idx_for_label, math.floor(num_samples/num_classes), False)
        if num_test is not None:
            train_idxs.extend(selected_idxs[:math.floor(num_train/num_classes)])
            test_idxs.extend(selected_idxs[math.floor(num_train/num_classes):])
        else:
           train_idxs.extend(selected_idxs)

    # ensure right amount of samples selected sometimes off for low samples sizes due to floor function
    diff = num_train - len(train_idxs)
    diffs = [diff]
    if num_test is not None:
        diff_test = num_test - len(test_idxs)
        diffs.append(diff_test)

    for num, diff in enumerate(diffs):
        if diff > 0:
            # add additional idxs not already selected
            while diff != 0:
                if ds_name in ['flowers']:
                    if num == 1:
                        ds_len = len(test_dataset._labels)
                    else:
                        ds_len = len(train_dataset._labels)
                elif ds_name in ['cub', 'cars']:
                    if num == 1:
                        ds_len = len(test_dataset.labels)
                    else:
                        ds_len = len(train_dataset.labels)
                else:
                    if num == 1:
                        ds_len = len(test_dataset.targets)
                    else:
                        ds_len = len(train_dataset.targets)
                if num == 1:
                    add_test_idxs = np.random.choice(np.array([i for i in range(ds_len)]), diff, False)
                    add_test_idxs = [idx for idx in add_test_idxs if idx not in test_idxs]
                    test_idxs.extend(add_train_idxs)
                    diff = diff - len(add_test_idxs)
                else:
                    add_train_idxs = np.random.choice(np.array([i for i in range(ds_len)]), diff, False)
                    add_train_idxs = [idx for idx in add_train_idxs if idx not in train_idxs]
                    train_idxs.extend(add_train_idxs)
                    diff = diff - len(add_train_idxs)
        elif diff < 0:
            # remove additional idxs
            if num == 1:
                test_idxs = list(np.random.choice(np.array(test_idxs), len(test_idxs) + diff, False))
            else:
                train_idxs = list(np.random.choice(np.array(train_idxs), len(train_idxs) + diff, False))
        else:
            pass

    train_dataset = torch.utils.data.Subset(train_dataset, train_idxs)
    if num_test is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, test_idxs)
        return train_dataset, test_dataset
    else:
        return train_dataset

def build_data_provider(local_batch_size, num_clients, dataset, num_classes, alpha, num_client_samples, drop_last: bool = False):
    # load dataset
    if dataset == 'femnist':
        data_dir = '../data/femnist/'
        train_dataset = FEMNIST(root=data_dir, train=True, download=True)
        mean = train_dataset.train_data.float().mean()
        std = train_dataset.train_data.float().std()

        apply_transform = transforms.Compose([
            transforms.RandomCrop(24, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        train_dataset = FEMNIST(data_dir, train=True, download=False, transform=apply_transform)
        test_dataset = FEMNIST(data_dir, train=False, download=False, transform=test_transform)
        if num_client_samples is not None:
            train_dataset = get_data_subset(ds_name='femnist',
                                            num_train=int(num_client_samples*num_clients),
                                                          train_dataset=train_dataset,
                                                          num_classes=num_classes,)

    elif dataset == 'flowers':
        train_dir = '../data/flowers/train/'
        test_dir = '../data/flowers/test/'
        train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                    ])
        test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                   ])

        train_dataset = Flowers102(train_dir, split='train', transform=train_transforms, download=True)
        test_dataset = Flowers102(test_dir, split='test', transform=test_transforms, download=True)
        if num_client_samples is not None:
            train_dataset = get_data_subset(ds_name='flowers',
                                            num_train=int(num_client_samples*num_clients),
                                            train_dataset=train_dataset,
                                            num_classes=num_classes)

    elif dataset == 'cub':
        data_dir = '../data/cub2011/'
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        train_dataset = Cub2011(data_dir, train=True, transform=train_transform, download=True)
        test_dataset = Cub2011(data_dir, train=False, transform=test_transform, download=True)
        if num_client_samples is not None:
            train_dataset = get_data_subset(ds_name='cub',
                                            num_train=int(num_client_samples*num_clients),
                                                          train_dataset=train_dataset,
                                                          num_classes=num_classes,)

    elif dataset == 'cars':
        data_dir = '../data/stanford_cars/'
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        train_dataset = StanfordCars(data_dir, split="train", transform=train_transform, download=True)
        test_dataset = StanfordCars(data_dir, split="test", transform=test_transform, download=True)
        if num_client_samples is not None:
            train_dataset = get_data_subset(ds_name='cars',
                                            num_train=int(num_client_samples*num_clients),
                                                          train_dataset=train_dataset,
                                                          num_classes=num_classes,)

    elif dataset == 'eurosat':
        data_dir = '../data/eurosat/'
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ssl._create_default_https_context = ssl._create_unverified_context
        train_dataset = EuroSAT(data_dir, transform=train_transform, download=True)
        test_dataset = EuroSAT(data_dir, transform=test_transform, download=True)

        if num_client_samples is not None:
            train_dataset, test_dataset = get_data_subset(ds_name='eurosat',
                                                          num_train=int(num_client_samples*num_clients),
                                                          train_dataset=train_dataset,
                                                          num_classes=num_classes,
                                                          num_test=1000,
                                                          test_dataset=test_dataset)
        else:
            # split randomly into 5000 train, 1000 test images (iid)
            train_dataset, test_dataset = get_data_subset(ds_name='eurosat',
                                                          num_train=5000,
                                                          train_dataset=train_dataset,
                                                          num_classes=num_classes,
                                                          num_test=1000,
                                                          test_dataset=test_dataset)

    elif dataset == 'cifar':
        train_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224),
                                     (0.247032237587, 0.243485133253, 0.261587846975))
            ]
        )

        train_dataset = CIFAR10(
            root="../data/cifar10/", train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(
            root="../data/cifar10/", train=False, download=True, transform=test_transform)
    elif dataset == 'rotten_tomatoes':

        dataset = load_dataset("rotten_tomatoes", split="train")
        dataset = dataset.map(tokenization, batched=True)
        dataset.set_format(type="torch", columns=['text', 'label', 'input_ids', 'attention_mask'])
        dataset.format['type']
        
        split_dataset = dataset.train_test_split(test_size=0.3)
        train_dataset = split_dataset['train']
        test_dataset = split_dataset['test']

    elif dataset == 'ag_news':
        dataset_size = 8400
        dataset = load_dataset("ag_news", split=f"train[:{dataset_size}]")
        dataset = dataset.map(tokenization, batched=True)
        dataset.set_format(type="torch", columns=['text', 'label', 'input_ids', 'attention_mask'])
        dataset.format['type']

        split_dataset = dataset.train_test_split(test_size=0.3)
        train_dataset = split_dataset['train']
        test_dataset = split_dataset['test']

    else:
        raise Exception(f'{dataset} is not a valid dataset')

    sharder = DirichletSharder(num_shards=num_clients,
                               alpha=alpha,
                               num_classes=num_classes,)
    fl_data_loader = DataLoader(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        test_dataset=test_dataset,
        sharder=sharder,
        batch_size=local_batch_size,
        drop_last=drop_last,
    )

    data_provider = DataProvider(fl_data_loader)
    return fl_data_loader, data_provider


class SimpleConvNet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0):
        super(SimpleConvNet, self).__init__()
        self.out_channels = 32
        self.stride = 1
        self.padding = 2
        self.layers = []
        in_dim = in_channels
        for _ in range(4):
            self.layers.append(
                nn.Conv2d(in_dim, self.out_channels, 3, self.stride, self.padding)
            )
            in_dim = self.out_channels
        self.layers = nn.ModuleList(self.layers)

        self.gn_relu = nn.Sequential(
            nn.GroupNorm(self.out_channels, self.out_channels, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        num_features = (
            self.out_channels
            * (self.stride + self.padding)
            * (self.stride + self.padding)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        for conv in self.layers:
            x = self.gn_relu(conv(x))

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(self.dropout(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class FLModel(IFLModel):
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device

    def fl_forward(self, batch, user_labels = None) -> FLBatchMetrics:
        features = batch["features"]  # [B, C, 28, 28]
        batch_label = batch["labels"]
        batch_mask = batch["attention_mask"]
        batch_mask = torch.stack(batch_mask[0], dim=0).to(self.device)

        stacked_label = batch_label.view(-1).long().clone().detach()

        if self.device is not None:
            features = features.to(self.device)

        if batch_mask is not None:
            if features.shape[0] < 64:
                batch_mask = batch_mask[0:features.shape[0]]

            output = self.model(features, mask=batch_mask)
        else:
            output = self.model(features)

        if user_labels:
            wsm_logits = self.weighted_log_softmax(output, user_labels).to(self.device)

        if self.device is not None:
            try:
                output, batch_label, stacked_label = (
                    output.to(self.device),
                    batch_label.to(self.device),
                    stacked_label.to(self.device),
                )
            except:
                output, batch_label, stacked_label = (
                    output.logits.to(self.device),
                    batch_label.to(self.device),
                    stacked_label.to(self.device),
                )

        if user_labels:
            loss = torch.nn.functional.nll_loss(wsm_logits, stacked_label)
            wsm_logits.detach().cpu()
        else:
            loss = F.cross_entropy(output, stacked_label)
        num_examples = self.get_num_examples(batch)
        output = output.detach().cpu()
        stacked_label = stacked_label.detach().cpu()
        del features
        return FLBatchMetrics(
            loss=loss,
            num_examples=num_examples,
            predictions=output,
            targets=stacked_label,
            model_inputs=[],
        )
    
    def weighted_log_softmax(self, logits, user_labels):
        """
        computes softmax weighted by class proportions from the client
        Args:
            logits: logits for the mini batch under consideration
        Returns:
            softmax weighted by class proportion
        """
        alphas = torch.tensor(self.client_data_proportions(user_labels), requires_grad=True)

        log_alphas = alphas.log().clamp_(min=-1e9)
        log_alphas = log_alphas.to(self.device)
        deno = torch.logsumexp(log_alphas.to(self.device) + logits.to(self.device), dim=-1, keepdim=True)
        
        return log_alphas + logits - deno

    def client_data_proportions(self, user_labels):
        client_labels = np.asarray([item for sublist in user_labels for item in sublist])
        print(f'reshape labels {client_labels}')
        #client_labels = user_labels

        #count_labels = len(client_labels)
        count_labels = client_labels.shape[0]
        print(f'num labels {count_labels}')

        count_client_labels = []
        for i in range(10):
            count_client_labels.append(int(np.argwhere(client_labels == i).shape[0]))
            print(f'count client labels updates {count_client_labels}')
        count_labels = np.array(count_labels)
        count_client_labels = np.array(count_client_labels)
        print(f'label couut {count_client_labels}, norm label count {count_client_labels / count_labels}')

        return count_client_labels / count_labels

    def fl_create_training_batch(self, **kwargs):
        features = kwargs.get("features", None)
        labels = kwargs.get("labels", None)
        return UserData.fl_training_batch(features, labels)

    def fl_get_module(self) -> nn.Module:
        return self.model

    def fl_cuda(self) -> None:
        self.model = self.model.to(self.device)  # pyre-ignore

    def get_eval_metrics(self, batch) -> FLBatchMetrics:
        with torch.no_grad():
            return self.fl_forward(batch)

    def get_num_examples(self, batch) -> int:
        return UserData.get_num_examples(batch["labels"])


class MetricsReporter(FLMetricsReporter):
    ACCURACY = "Accuracy"

    def __init__(
        self,
        channels: List[Channel],
        target_eval: float = 0.0,
        window_size: int = 5,
        average_type: str = "sma",
        log_dir: Optional[str] = None,
        wandb_dict: dict = None
    ):
        super().__init__(channels, wandb_dict, log_dir)
        self.set_summary_writer(log_dir=log_dir)
        self._round_to_target = float(1e10)

    def compare_metrics(self, eval_metrics, best_metrics):
        print(f"Current eval accuracy: {eval_metrics}%, Best so far: {best_metrics}%")
        if self.wandb_dict['activate']:
            wandb.log({f"Current eval accuracy": eval_metrics})
        if best_metrics is None:
            return True

        current_accuracy = eval_metrics.get(self.ACCURACY, float("-inf"))
        best_accuracy = best_metrics.get(self.ACCURACY, float("-inf"))
        return current_accuracy > best_accuracy

    def compute_scores(self) -> Dict[str, Any]:
        # compute accuracy
        correct = torch.Tensor([0])
        for i in range(len(self.predictions_list)):
            all_preds = self.predictions_list[i]
            pred = all_preds.data.max(1, keepdim=True)[1]

            assert pred.device == self.targets_list[i].device, (
                f"Pred and targets moved to different devices: "
                f"pred >> {pred.device} vs. targets >> {self.targets_list[i].device}"
            )
            if i == 0:
                correct = correct.to(pred.device)

            correct += pred.eq(self.targets_list[i].data.view_as(pred)).sum()

        # total number of data
        total = sum(len(batch_targets) for batch_targets in self.targets_list)

        accuracy = 100.0 * correct.item() / total
        return {self.ACCURACY: accuracy}

    def create_eval_metrics(
        self, scores: Dict[str, Any], total_loss: float, **kwargs
    ) -> Any:
        accuracy = scores[self.ACCURACY]
        return {self.ACCURACY: accuracy}

def wandb_setup(cfg):
    if cfg.wandb.run_name:
        os.environ['WANDB_NAME'] = cfg.wandb.run_name
        os.environ['WANDB_START_METHOD'] = "thread"
    if cfg.wandb.offline:
        os.environ["WANDB_MODE"] = "offline"

    # need to set wandb run_dir to something we can access to avoid permission denied error.
    # See https://github.com/wandb/client/issues/3437
    wandb_path = f'/scratch/{os.environ.get("USER","glegate")}/wandb'
    #wandb_path = f'wandb'
    if not os.path.isdir(wandb_path):
        os.makedirs(wandb_path, mode=0o755, exist_ok=True)

    # if using wandb check project and entity are set
    assert not cfg.wandb.project == '' and not cfg.wandb.entity == ''
    wandb.login()
    wandb.init(dir=wandb_path, project=cfg.wandb.project, entity=cfg.wandb.entity)
    general_args = {
        "algorithm": cfg.trainer.algorithm,
        "fl_algorithm": cfg.trainer.fl_algorithm,
        "client_lr": cfg.trainer.client.optimizer.lr,
        "server_lr": cfg.trainer.server.server_optimizer.lr,
        "rounds": cfg.trainer.epochs,
        "dataset": cfg.dataset.name,
        "num_clients": cfg.trainer.total_users,
        "clients_per_round": cfg.trainer.users_per_round,
        "local_epochs": cfg.trainer.client.epochs,
        "local_bs": cfg.trainer.client.local_bs,
        "dirichlet_alpha": cfg.dataset.alpha,
        "optimizer": cfg.set_optimizer,
        "pretrained": cfg.trainer.pretrained,
        "NCM": cfg.trainer.ncm_init,
        "momentum" : cfg.trainer.server.server_optimizer.momentum,
        "alpha": cfg.dataset.alpha,
        "model": cfg.trainer.model,
    }
    wandb.config.update(general_args)

def set_cfg_from_cl(cfg):
    args = args_parser()
    print(args)
    for arg in vars(args):
        if getattr(args, arg) is not None:
            if arg == 'wandb':
                cfg.wandb.activate = args.wandb
            elif arg == 'wandb_project':
                cfg.wandb.project = args.wandb_project
            elif arg == 'wandb_entity':
                cfg.wandb.entity = args.wandb_entity
            elif arg == 'wandb_run_name':
                cfg.wandb.run_name = args.wandb_run_name
            elif arg == 'model':
                cfg.trainer.model = args.model
            elif arg == 'ncm':
                assert args.ncm == 1 or args.ncm == 0, f'value passed for --ncm was {args.ncm}, it must be 1 or 0'
                if args.ncm == 0:
                    cfg.trainer.ncm_init = False
                else:
                    cfg.trainer.ncm_init = True
            elif arg == 'algorithm':
                cfg.trainer.algorithm = args.algorithm
            elif arg == 'fl_algorithm':
                cfg.trainer.fl_algorithm = args.fl_algorithm
            elif arg == 'epochs':
                cfg.trainer.epochs = args.epochs
            elif arg == 'num_clients':
                cfg.trainer.total_users = args.num_clients
            elif arg == 'clients_per_round':
                cfg.trainer.users_per_round = args.clients_per_round
            elif arg == 'local_ep':
                cfg.trainer.client.epochs = args.local_ep
            elif arg == 'local_bs':
                cfg.trainer.client.local_bs = args.local_bs
            elif arg == 'client_lr':
                cfg.trainer.client.optimizer.lr = args.client_lr
            elif arg == 'server_lr':
                cfg.trainer.server.server_optimizer.lr = args.server_lr
            elif arg == 'momentum':
                cfg.trainer.server.server_optimizer.momentum = args.momentum
            elif arg == 'dataset':
                cfg.dataset.name = args.dataset
            elif arg == 'optimizer':
                cfg.set_optimizer = args.optimizer
            elif arg == 'alpha':
                cfg.dataset.alpha = args.alpha
            elif arg == 'num_client_samples':
                cfg.dataset.num_client_samples = args.num_client_samples
            elif arg == 'pretrained':
                assert args.pretrained == 1 or args.pretrained == 0, \
                    f'value passed for --pretrained was {args.pretrained}, it must be 1 or 0'
                if args.pretrained == 0:
                    cfg.trainer.pretrained = False
                else:
                    cfg.trainer.pretrained = True
            else:
                print(f'this should never happen arg: {arg}')
                exit(1)

def inference(model, dataloader, device):
    """
    Returns the inference accuracy and loss.
    """
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    loss = loss / batch_idx + 1
    return accuracy, loss



