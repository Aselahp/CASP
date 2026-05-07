from __future__ import division, print_function

import math
import time
import argparse
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx

from utils import *
from models import (
    CASP_GCN, CASP_APPNP, CASP_MixHop, CASP_DirGNN, CASP_FAGCN,
    louvain_partition, louvain_modularity,
    build_casp_edge_sets, normalise_casp_edges,
    compute_delta, compute_p,
)
from dataset_utils import DataLoader

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

parser = argparse.ArgumentParser()
parser.add_argument('--seed',           type=int,   default=42)
parser.add_argument('--epochs',         type=int,   default=1000)
parser.add_argument('--lr',             type=float, default=0.01)
parser.add_argument('--weight_decay',   type=float, default=5e-4)
parser.add_argument('--hidden',         type=int,   default=256)
parser.add_argument('--early_stopping', type=int,   default=200)
parser.add_argument('--train_rate',     type=float, default=0.6)
parser.add_argument('--val_rate',       type=float, default=0.2)
parser.add_argument('--dropout',        type=float, default=0.1)
parser.add_argument('--layers',         type=int,   default=2)
parser.add_argument('--dataset',        default='citeseer')
parser.add_argument('--resolution',     type=float, default=1.0,
                    help="Louvain resolution parameter (higher = more communities)")
args = parser.parse_args("")

np.random.seed(args.seed)
torch.manual_seed(args.seed)

permute_masks = random_planetoid_splits


def compute_degree(edge_index, num_nodes=None):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_edges_from(zip(edge_index[0], edge_index[1]))
    if num_nodes is not None:
        G.add_nodes_from(range(num_nodes))
    degrees = dict(G.degree())
    return torch.tensor([degrees.get(i, 0) for i in range(max(G.nodes) + 1)])


def copy_data(data):
    import copy
    return copy.deepcopy(data)


dataset    = DataLoader(args.dataset)
data_orig  = dataset[0]
percls_trn = int(round(args.train_rate * len(data_orig.y) / dataset.num_classes))
val_lb     = int(round(args.val_rate   * len(data_orig.y)))


def create_random_split(data, run_seed):
    np.random.seed(args.seed + run_seed)
    torch.manual_seed(args.seed + run_seed)
    data_split = permute_masks(copy_data(data_orig),
                               dataset.num_classes, percls_trn, val_lb)
    np.random.seed(args.seed + run_seed)
    torch.manual_seed(args.seed + run_seed)
    A, X, labels, idx_train, idx_val, idx_test = load_citation_data(data_split)
    features = torch.FloatTensor(X)
    labels   = torch.LongTensor(labels)
    degrees  = compute_degree(A)
    return data_split, features, labels, A, idx_train, idx_val, idx_test, degrees


print(f"\nRunning CASP preprocessing (resolution={args.resolution}) ...")
t0 = time.time()

data_split_ref, features_ref, labels_ref, adj_ref, \
    idx_train_ref, _, _, _ = create_random_split(data_orig, 0)

n          = features_ref.size(0)
edge_index = adj_ref if isinstance(adj_ref, torch.Tensor) else torch.tensor(adj_ref)

comm = louvain_partition(edge_index, n, resolution=args.resolution)
Q    = louvain_modularity(edge_index, n, comm)

edge_intra_raw, edge_inter_raw = build_casp_edge_sets(edge_index, comm)

ei_intra_norm, ew_intra, \
ei_inter_norm, ew_inter = normalise_casp_edges(
                              edge_intra_raw, edge_inter_raw, n,
                              dtype=features_ref.dtype)

p = compute_p(edge_intra_raw, edge_inter_raw)

train_mask_ref = torch.zeros(n, dtype=torch.bool)
train_mask_ref[idx_train_ref] = True
delta = compute_delta(edge_intra_raw, edge_inter_raw, labels_ref, train_mask_ref)

r_init  = math.exp(1.0 * delta)
ai_init = r_init / (r_init * p + (1 - p))
ao_init = 1.0    / (r_init * p + (1 - p))

print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Q             = {Q:.4f}")
print(f"  p             = {p:.4f}")
print(f"  δ             = {delta:.4f}")
print(f"  r_0           = {r_init:.4f}")
print(f"  α_intra(init) = {ai_init:.4f},  α_inter(init) = {ao_init:.4f}")
print(f"  |E_intra|     = {edge_intra_raw.size(1)},  |E_inter| = {edge_inter_raw.size(1)}")
print()


def train(epoch, model, optimizer, features, adj, labels, idx_train, train_mask):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj, labels=labels, train_mask=train_mask)
    l_task = F.nll_loss(output[idx_train].squeeze(-1), labels[idx_train])
    loss   = l_task
    acc    = accuracy(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_task: {:.4f}'.format(l_task.item()),
          'acc: {:.4f}'.format(acc.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss.item(), acc.item()


def test(model, features, adj, labels, data_split):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
    accs, losses, preds = [], [], []
    for _, mask in data_split('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc  = accuracy(logits[mask], labels[mask])
        loss = F.nll_loss(logits[mask].squeeze(-1), labels[mask])
        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())
    return accs, preds, losses


Results0  = []
run_stats = []

for i in range(10):
    print(f"\n{'='*50}\nRun {i+1}/10\n{'='*50}")

    data, features, labels, adj, idx_train, idx_val, idx_test, degrees = \
        create_random_split(data_orig, i)

    train_mask_i = torch.zeros(n, dtype=torch.bool)
    train_mask_i[idx_train] = True

    delta_i = compute_delta(edge_intra_raw, edge_inter_raw, labels, train_mask_i)

    model = CASP_GCN(
        in_features     = features.shape[1],
        hidden_features = args.hidden,
        out_features    = int(labels.max()) + 1,
        dropout         = args.dropout,
        t_max           = args.layers,
        ei_intra_norm   = ei_intra_norm,
        ew_intra        = ew_intra,
        ei_inter_norm   = ei_inter_norm,
        ew_inter        = ew_inter,
        ei_intra_raw    = edge_intra_raw,
        ei_inter_raw    = edge_inter_raw,
        p               = p,
        delta           = delta_i,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc  = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train(
            epoch, model, optimizer, features, adj, labels, idx_train, train_mask_i)

        ([_, val_acc, tmp_test_acc], preds, [_, val_loss, _]) = \
            test(model, features, adj, labels, data)

        val_loss_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            test_acc      = tmp_test_acc
            best_epoch    = epoch

        if args.early_stopping > 0 and epoch > args.early_stopping:
            tmp = torch.tensor(val_loss_history[-(args.early_stopping + 1):-1])
            if val_loss > tmp.mean().item():
                print(f"Early stopping at epoch {epoch}")
                break

    Results0.append([test_acc, best_val_acc])
    run_stats.append({
        'run':          i,
        'best_epoch':   best_epoch,
        'best_val_acc': best_val_acc.item(),
        'test_acc':     test_acc.item(),
    })
    print(f"Run {i+1}: Test={test_acc:.4f}  Val={best_val_acc:.4f}")


test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100
test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100

print(f"\n{'='*50}")
print(f"Final Results ({args.dataset}, resolution={args.resolution})")
print(f"{'='*50}")
print(f"Test accuracy:       {test_acc_mean:.2f}% ± {test_acc_std:.2f}%")
print(f"Validation accuracy: {val_acc_mean:.2f}%")
print(f"p = {p:.4f}  |  δ = {delta:.4f}")