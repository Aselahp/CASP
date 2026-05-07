import os.path as osp
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikipediaNetwork, HeterophilousGraphDataset, WebKB, Actor, CitationFull, DBLP, NELL
import torch
from torch_geometric.data import Data
import numpy as np
import os


def load_filtered_wiki(name, root='./data'):
    import urllib.request, os
    path = osp.join(root, f'{name}_filtered')
    os.makedirs(path, exist_ok=True)
    npz_path = osp.join(path, f'{name}_filtered.npz')
    if not osp.exists(npz_path):
        url = f'https://github.com/yandex-research/heterophilous-graphs/raw/main/data/{name}_filtered.npz'
        print(f'Downloading {name}_filtered from {url}')
        urllib.request.urlretrieve(url, npz_path)

    raw = np.load(npz_path)
    x = torch.tensor(raw['node_features'], dtype=torch.float)
    y = torch.tensor(raw['node_labels'], dtype=torch.long)
    N = x.shape[0]

    edges = raw['edges']
    # Make sure edge_index is [2, E]
    if edges.shape[0] == 2:
        edge_index = torch.tensor(edges, dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).T

    def _fix_mask(m):
        m = torch.tensor(m, dtype=torch.bool)
        # Want shape [N, num_splits]
        if m.dim() == 1:
            m = m.unsqueeze(1)
        elif m.shape[0] != N and m.shape[1] == N:
            m = m.T
        return m.contiguous()

    train_mask = _fix_mask(raw['train_masks'])
    val_mask   = _fix_mask(raw['val_masks'])
    test_mask  = _fix_mask(raw['test_masks'])

    data = Data(x=x, y=y, edge_index=edge_index,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    class _FilteredDataset:
        def __init__(self, data):
            self._data = data
            self.num_classes = int(data.y.max().item()) + 1
            self.num_features = data.x.shape[1]
            self.num_node_features = data.x.shape[1]
        def __getitem__(self, idx): return self._data
        def __len__(self): return 1

    return _FilteredDataset(data)


def DataLoader(name):
    name = name.lower()
    
    # Standard PyTorch Geometric datasets
    if name in ['cora', 'citeseer', 'pubmed']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Planetoid(path, name, split="geom-gcn", transform=None)
    elif name in ['cora1', 'citeseer1', 'pubmed', 'cora_ml', 'dblp']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = CitationFull(path, name, transform=None)
    elif name in ['cs', 'physics']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Coauthor(path, name, transform=None)
    elif name in ['computers', 'photo', 'physics']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Amazon(path, name, transform=None)
    # Filtered chameleon / squirrel (Platonov et al.)
    elif name in ['chameleon', 'squirrel']:
        dataset = load_filtered_wiki(name, root='./data')
    elif name in ['crocodile']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = WikipediaNetwork(path, name, transform=None)
    elif name in ['cornell', 'texas', 'wisconsin']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = WebKB(path, name)
    elif name in ['film']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Actor(path)
    elif name in ['nell']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = NELL(path)
    # New heterophilous datasets
    elif name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = HeterophilousGraphDataset(path, name, transform=None)
    elif name in ["cora_full"]:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = CitationFull(path, name="Cora")
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')
    
    return dataset