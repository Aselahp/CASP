from __future__ import division, print_function

import math
import time
import argparse
import copy
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]  = ""

import torch
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import remove_self_loops, to_undirected

from models import (
    CASP_GCN, CASP_APPNP, CASP_MixHop, CASP_FAGCN, CASP_DirGNN,
    louvain_partition, louvain_modularity,
    build_casp_edge_sets, normalise_casp_edges,
    compute_delta, compute_p,
)
from dataset_utils import DataLoader

import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--seed',           type=int,   default=42)
parser.add_argument('--epochs',         type=int,   default=1000)
parser.add_argument('--lr',             type=float, default=0.01)
parser.add_argument('--weight_decay',   type=float, default=5e-4)
parser.add_argument('--hidden',         type=int,   default=256)
parser.add_argument('--early_stopping', type=int,   default=200)
parser.add_argument('--dropout',        type=float, default=0.1)
parser.add_argument('--layers',         type=int,   default=2)
parser.add_argument('--dataset',        default='tolokers',
                    choices=['minesweeper', 'tolokers', 'questions'])
parser.add_argument('--train_rate',     type=float, default=0.6)
parser.add_argument('--val_rate',       type=float, default=0.2)
parser.add_argument('--num_runs',       type=int,   default=10)
parser.add_argument('--resolution', type=float, default=1.0,
                    help="Louvain resolution parameter")
args = parser.parse_args("")

np.random.seed(args.seed)
torch.manual_seed(args.seed)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def roc_auc(output, labels):
    """ROC-AUC for binary classification. output is log-softmax [N, 2]."""
    with torch.no_grad():
        probs = output.exp()[:, 1].detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    if len(np.unique(y_true)) < 2:
        return torch.tensor(0.5)
    return torch.tensor(roc_auc_score(y_true, probs))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def copy_data(data):
    return copy.deepcopy(data)


def stratified_split(y, train_rate, val_rate, seed):
    """
    Stratified 60/20/20 (or whatever) split — samples per class so that
    train/val/test all reflect the class distribution. This guarantees
    both classes appear in every mask, which is essential for ROC-AUC
    on imbalanced datasets like Questions (97/3).
    """
    rng = np.random.RandomState(seed)
    N = y.shape[0]
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)

    for c in y.unique().tolist():
        class_idx = (y == c).nonzero(as_tuple=True)[0].cpu().numpy()
        rng.shuffle(class_idx)
        n_c       = len(class_idx)
        n_train   = int(round(train_rate * n_c))
        n_val     = int(round(val_rate   * n_c))
        train_ids = class_idx[:n_train]
        val_ids   = class_idx[n_train : n_train + n_val]
        test_ids  = class_idx[n_train + n_val :]
        train_mask[train_ids] = True
        val_mask[val_ids]     = True
        test_mask[test_ids]   = True

    return train_mask, val_mask, test_mask


def prepare_data_with_split(data_full, run_seed):
    """
    Clean edge_index and build a stratified 60/20/20 split using run_seed.
    """
    d = copy_data(data_full)

    ei = d.edge_index
    ei = remove_self_loops(ei)[0]
    ei = to_undirected(ei)

    features = d.x.float()
    labels   = d.y.long()

    tr, va, te = stratified_split(labels, args.train_rate, args.val_rate,
                                  seed=args.seed + run_seed)
    return features, labels, ei, tr, va, te


# ─────────────────────────────────────────────────────────────────────────────
# Dataset & CASP preprocessing
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nLoading {args.dataset} ...")
dataset   = DataLoader(args.dataset)
data_orig = dataset[0]
num_classes = int(data_orig.y.max().item()) + 1
n = data_orig.x.shape[0]

class_counts = torch.bincount(data_orig.y).tolist()
print(f"  |V| = {n},  |E| = {data_orig.edge_index.shape[1]},  classes = {num_classes}")
print(f"  Class balance: {class_counts}  "
      f"(minority: {min(class_counts)/n*100:.2f}%)")
print(f"  Split: {args.train_rate*100:.0f}/{args.val_rate*100:.0f}/"
      f"{(1-args.train_rate-args.val_rate)*100:.0f}  (stratified)")

print(f"\nRunning CASP preprocessing ...")
t0 = time.time()

# Use split 0 as reference for CASP preprocessing
features_ref, labels_ref, edge_index_ref, tr_ref, _, _ = prepare_data_with_split(data_orig, 0)

comm = lcomm = louvain_partition(edge_index_ref, n, resolution=args.resolution)
Q    = louvain_modularity(edge_index_ref, n, comm)

edge_intra_raw, edge_inter_raw = build_casp_edge_sets(edge_index_ref, comm)

ei_intra_norm, ew_intra, ei_inter_norm, ew_inter = normalise_casp_edges(
    edge_intra_raw, edge_inter_raw, n, dtype=features_ref.dtype)

p = compute_p(edge_intra_raw, edge_inter_raw)
delta = compute_delta(edge_intra_raw, edge_inter_raw, labels_ref, tr_ref)

r_init = math.exp(1.0 * delta)
ai_init = r_init / (r_init * p + (1 - p))
ao_init = 1.0    / (r_init * p + (1 - p))

print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Q             = {Q:.4f}")
print(f"  p             = {p:.4f}  (intra-edge fraction)")
print(f"  δ (fast)      = {delta:.4f}  (community–label alignment)")
print(f"  θ_0 = 1.0  →  r_0 = {r_init:.4f}")
print(f"  α_intra(init) = {ai_init:.4f},  α_inter(init) = {ao_init:.4f}")
print(f"  |E_intra| = {edge_intra_raw.size(1)},  |E_inter| = {edge_inter_raw.size(1)}")


# ─────────────────────────────────────────────────────────────────────────────
# Train / test
# ─────────────────────────────────────────────────────────────────────────────

def train(epoch, model, optimizer, features, adj, labels, train_mask, verbose=False):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features, adj, labels=labels, train_mask=train_mask)

    l_task = F.nll_loss(output[train_mask].squeeze(-1), labels[train_mask])
    loss   = l_task
    auc    = roc_auc(output[train_mask], labels[train_mask])

    loss.backward()
    optimizer.step()

    if verbose:
        print(f'Epoch: {epoch+1:04d}  '
          f'loss: {loss.item():.4f}  '
          f'auc: {auc.item():.4f}  '
          f'time: {time.time()-t:.3f}s')
    return loss.item(), auc.item()


def evaluate(model, features, adj, labels, train_mask, val_mask, test_mask):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)

    aucs, losses = [], []
    for mask in [train_mask, val_mask, test_mask]:
        auc  = roc_auc(logits[mask], labels[mask])
        loss = F.nll_loss(logits[mask].squeeze(-1), labels[mask])
        aucs.append(auc)
        losses.append(loss.detach().cpu())
    return aucs, losses


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

Results   = []
run_stats = []

for i in range(args.num_runs):
    print(f"\n{'='*60}\nRun {i+1}/{args.num_runs}\n{'='*60}")

    features, labels, adj, tr_mask, va_mask, te_mask = prepare_data_with_split(data_orig, i)

    # Sanity: confirm both classes present in every mask
    for name, m in [('train', tr_mask), ('val', va_mask), ('test', te_mask)]:
        nclass = labels[m].unique().numel()
        assert nclass >= 2, f"{name} mask has only {nclass} class — stratification broken"

    print(f"  Split sizes: train={tr_mask.sum().item()}  "
          f"val={va_mask.sum().item()}  test={te_mask.sum().item()}")
    print(f"  Positive class in masks: "
          f"train={labels[tr_mask].float().mean().item():.3f}  "
          f"val={labels[va_mask].float().mean().item():.3f}  "
          f"test={labels[te_mask].float().mean().item():.3f}")

    delta_i = compute_delta(edge_intra_raw, edge_inter_raw, labels, tr_mask)

    np.random.seed(args.seed + i)
    torch.manual_seed(args.seed + i)

    model = CASP_GCN(
        in_features     = features.shape[1],
        hidden_features = args.hidden,
        out_features    = num_classes,
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

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    best_val_auc  = 0.0
    best_test_auc = 0.0
    best_val_loss = float('inf')
    val_loss_history = []
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss, train_auc = train(
            epoch, model, optimizer, features, adj, labels, tr_mask,
            verbose=True)

        (aucs, losses) = evaluate(
            model, features, adj, labels, tr_mask, va_mask, te_mask)
        _, val_auc, test_auc = aucs
        _, val_loss, _       = losses

        val_loss_history.append(val_loss)

        if val_auc > best_val_auc:
            best_val_auc  = val_auc
            best_test_auc = test_auc
            best_val_loss = val_loss
            best_epoch    = epoch

        if args.early_stopping > 0 and epoch > args.early_stopping:
            tmp = torch.tensor(val_loss_history[-(args.early_stopping + 1):-1])
            if val_loss > tmp.mean().item():
                print(f"  Early stopping at epoch {epoch}")
                break

    #stats = model.casp_stats()
    Results.append([best_test_auc, best_val_auc])
    run_stats.append({
        'run':          i,
        'best_epoch':   best_epoch,
        'best_val_auc': best_val_auc.item() if torch.is_tensor(best_val_auc) else best_val_auc,
        'test_auc':     best_test_auc.item() if torch.is_tensor(best_test_auc) else best_test_auc,
        #**stats,
    })

    print(f"  Best epoch: {best_epoch}  |  "
          f"Test AUC={best_test_auc:.4f}  Val AUC={best_val_auc:.4f}  ")
          #f"θ={stats['theta']:.4f}  r={stats['r']:.4f}  "
          #f"α_intra={stats['alpha_intra']:.4f}  α_inter={stats['alpha_inter']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Final results
# ─────────────────────────────────────────────────────────────────────────────

Results_np = np.array([[float(t), float(v)] for t, v in Results])
test_mean, val_mean = Results_np.mean(axis=0) * 100
test_std = Results_np.std(axis=0)[0] * 100

print(f"\n{'='*60}\nFinal Results ({args.dataset}, metric: ROC-AUC)\n{'='*60}")
print(f"Split: {args.train_rate*100:.0f}/{args.val_rate*100:.0f}/"
      f"{(1-args.train_rate-args.val_rate)*100:.0f}  (stratified, {args.num_runs} runs)")
print(f"Test AUC:       {test_mean:.2f}% ± {test_std:.2f}%")
print(f"Validation AUC: {val_mean:.2f}%")
print(f"p = {p:.4f}  |  δ (ref) = {delta:.4f}  |  Q = {Q:.4f}")

print("\nPer-run results:")
'''for i, stats in enumerate(run_stats):
    print(f"  Run {i+1}: Test AUC={stats['test_auc']*100:.2f}%  "
          f"Val AUC={stats['best_val_auc']*100:.2f}%  "
          f"θ={stats['theta']:.4f}  r={stats['r']:.4f}  "
          f"ai={stats['alpha_intra']:.4f}  ao={stats['alpha_inter']:.4f}")'''
