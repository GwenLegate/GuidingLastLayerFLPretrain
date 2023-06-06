
from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, TypeVar

import numpy as np
from flsim.utils.config_utils import fullclassname, init_self_cfg
from omegaconf import MISSING
from torch.utils.data import Dataset
from flsim.data.data_sharder import FLDataSharder, FLDataSharderConfig

Shardable = TypeVar("Shardable", Iterable[Dict[str, Any]], Dataset)

class DirichletSharder(FLDataSharder):
    """Splits training data along dirichlet distribution. num_shards, num_classes and alpha should be specified"""

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=DirichletSharderConfig,
            **kwargs,
        )
        super().__init__(**kwargs)
        self._shard_assignments = None
        self._all_labels = None

    def shard_for_row(self, csv_row: Dict[Any, Any]) -> List[int]:
        """
            Dtermine which shard a row should belong to.
        """
        # distributions are only used once, re-distribute every time for train, validation, test
        if self._shard_assignments is None:
            self._shard_assignments = \
                self.noniid_dirichlet(self._all_labels, self.cfg.alpha, self.cfg.num_shards, self.cfg.num_classes)

        sample_idx = csv_row['features']
        # data is divided equally so there are sometimes leftover samples that are unused, need to bypass these
        try:
            assigned_client = self._shard_assignments[sample_idx]
        except KeyError:
            assigned_client = None
        return [assigned_client]
        #return [self._shard_assignments[sample_idx]]


    def shard_rows(self, data_rows: Shardable) -> Iterable[Tuple[str, Any]]:
        """Partition a set of rows into multiple sets using a sharding strategy.

        Args:
            data_rows: Iterable[Dict[str, Any]]]: iterable over dictionary mapping column
            name to value.
        """
        if self._all_labels is None:
            self._all_labels = self.get_labels(data_rows)

        shards = defaultdict(list)
        idx = 0
        # pyre-fixme[16]: `Dataset` has no attribute `__iter__`.
        count_none = 0
        for one_row in data_rows:
            # sorry about this, it's kind of an ugly work around to make the dirichlet code work with the FLSim framework.
            # I need the row index to find a samples client assignment otherwise I'd have to store all the features in
            # label assignment dict and search the entire dict for matching sample by its features every time.
            # Instead I'm modifying one_row to replace 'features' with sample index which meets all the type definitions
            # and saves a lot of compute -GL
            one_row_for_dirichlet = copy.deepcopy(one_row)
            one_row_for_dirichlet['features'] = idx
            for shard_id in self.shard_for_row(one_row_for_dirichlet):
                #shards[str(shard_id)].append(one_row)
                if shard_id is not None:
                    shards[str(shard_id)].append(one_row)
                else:
                    count_none += 1

            idx += 1
        # reset labels and distributions to None
        self._shard_assignments = None
        self._all_labels = None
        return shards.items()

    def get_labels(self, data_rows):
        labels = []
        for one_row in data_rows:
            labels.append(one_row['labels'])
        return np.asarray(labels, dtype=int)

    def noniid_dirichlet(self, labels, alpha, num_clients, num_classes):
        """Splits samples among shards according to a dirichlet distribution
         parameterized by alpha
        Args:
            alpha: Parameter of Dirichlet distribution. Each client
            samples from this Dirichlet to get a multinomial distribution over
            classes. It controls the data heterogeneity of clients.
            num_clients: The number of clients the examples are going to be partitioned on.
            num_classes: The number of unique classes in the dataset
            labels: list of class labels in order that they appear in data_rows
        Returns:
            a dict where keys are sample index and values are shards (client indices).
            """
        print(f'distributing data as dirichlet distribution with alpha={alpha}')
        dict_label_assignment = {}
        multinomial_vals = []
        examples_per_label = []

        for i in range(num_classes):
            examples_per_label.append(int(np.argwhere(labels == i).shape[0]))

        # Each client has a multinomial distribution over classes drawn from a Dirichlet.
        for i in range(num_clients):
            proportion = np.random.dirichlet(alpha * np.ones(num_classes))
            multinomial_vals.append(proportion)

        multinomial_vals = np.array(multinomial_vals)
        example_indices = []

        for k in range(num_classes):
            label_k = np.where(labels == k)[0]
            np.random.shuffle(label_k)
            example_indices.append(label_k)
        example_indices = np.array(example_indices, dtype=object)

        client_samples = [[] for _ in range(num_clients)]
        count = np.zeros(num_classes).astype(int)

        examples_per_client = int(labels.shape[0] / num_clients)

        max_fail = 0
        for k in range(num_clients):
            for i in range(examples_per_client):
                sampled_label = np.argwhere(np.random.multinomial(1, multinomial_vals[k, :]) == 1)[0][0]

                label_indices = example_indices[sampled_label]

                # very sketchy work around for low dataset sample sizes
                try:
                    client_samples[k].append(label_indices[count[sampled_label]])
                except IndexError:
                    i -= 1
                    max_fail += 1
                    if max_fail >1000:
                        print('(line 142) Dirichlet Sharder too many fails')
                        exit(1)
                    continue

                count[sampled_label] += 1
                if count[sampled_label] == examples_per_label[sampled_label]:
                    multinomial_vals[:, sampled_label] = 0
                    multinomial_vals = (
                            multinomial_vals /
                            multinomial_vals.sum(axis=1)[:, None])

        for i in range(num_clients):
            for idx in client_samples[i]:
                dict_label_assignment[idx] = i
        return dict_label_assignment

@dataclass
class DirichletSharderConfig(FLDataSharderConfig):
    _target_: str = fullclassname(DirichletSharder)
    num_shards: int = MISSING
    alpha: float = MISSING
    num_classes: int = MISSING