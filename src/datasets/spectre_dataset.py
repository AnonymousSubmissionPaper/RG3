import os
import pathlib

import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class SpectreGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.sbm_file = 'sbm_200.pt'
        self.planar_file = 'planar_64_200.pt'
        self.comm20_file = 'community_12_21_100.pt'
        self.dataset_name = dataset_name
        self.split = split
        self.num_graphs = 200
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        if self.dataset_name == 'sbm':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt'
        elif self.dataset_name == 'planar':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt'
        elif self.dataset_name == 'comm20':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt'
        elif self.dataset_name == "ego":
            raw_url = "https://raw.githubusercontent.com/tufts-ml/graph-generation-EDGE/main/graphs/Ego.pkl"
        else:
            raise ValueError(f'Unknown dataset {self.dataset_name}')
        file_path = download_url(raw_url, self.raw_dir)


        if self.dataset_name == 'ego':
            import pickle as pkl
            from networkx import to_numpy_array
            networks = pkl.load(open(file_path, 'rb'))
            adjs = [torch.Tensor(to_numpy_array(network)).fill_diagonal_(0) for network in networks]
        else:
            adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(file_path)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)

        self.num_graphs = len(adjs)

        if self.dataset_name == 'ego':
            test_len = int(round(self.num_graphs * 0.2))
            train_len = int(round(self.num_graphs * 0.8))
            val_len = int(round(self.num_graphs * 0.2))
            indices = torch.randperm(self.num_graphs, generator=g_cpu)
            train_indices = indices[:train_len]
            val_indices = indices[:val_len]
            test_indices = indices[train_len:]
            print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
        else:
            test_len = int(round(self.num_graphs * 0.2))
            train_len = int(round((self.num_graphs - test_len) * 0.8))
            val_len = self.num_graphs - train_len - test_len
            indices = torch.randperm(self.num_graphs, generator=g_cpu)
            print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
            train_indices = indices[:train_len]
            val_indices = indices[train_len : train_len + val_len]
            test_indices = indices[train_len + val_len :]

        train_data = []
        val_data = []
        test_data = []

        for i, adj in enumerate(adjs):
            if i in train_indices:
                train_data.append(adj)
            if i in val_indices:
                val_data.append(adj)
            if i in test_indices:
                test_data.append(adj)


        print(f"Train data: {len(train_data)}, Val data: {len(val_data)}, Test data: {len(test_data)}")

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])


    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        for adj in raw_dataset:
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float()
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=num_nodes)
            data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])

class Comm20Dataset(SpectreGraphDataset):
    def __init__(self):
        super().__init__('community_12_21_100.pt')


class SBMDataset(SpectreGraphDataset):
    def __init__(self):
        super().__init__('sbm_200.pt')


class PlanarDataset(SpectreGraphDataset):
    def __init__(self):
        super().__init__('planar_64_200.pt')


class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path),
                    'val': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='val', root=root_path),
                    'test': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]

class Comm20DataModule(SpectreGraphDataModule):
    def prepare_data(self):
        graphs = Comm20Dataset()
        return super().prepare_data(graphs)


class SBMDataModule(SpectreGraphDataModule):
    def prepare_data(self):
        graphs = SBMDataset()
        return super().prepare_data(graphs)


class PlanarDataModule(SpectreGraphDataModule):
    def prepare_data(self):
        graphs = PlanarDataset()
        return super().prepare_data(graphs)

class EgoDataModule(SpectreGraphDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)

class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

