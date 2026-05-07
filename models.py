import math
import pathlib
import importlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP, MixHopConv, GCNConv, GCN2Conv, FAConv
from torch_geometric.nn.conv import DirGNNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import remove_self_loops

_louvain_best_partition = None
_nx = None
_THIS_DIR = pathlib.Path(__file__).parent.resolve()


def _ensure_louvain():
    global _louvain_best_partition, _nx
    if _louvain_best_partition is not None:
        return
    original_path = sys.path[:]
    sys.path = [p for p in sys.path if pathlib.Path(p).resolve() != _THIS_DIR]
    _cached = sys.modules.pop('community', None)
    try:
        community_mod = importlib.import_module('community')
        if not hasattr(community_mod, 'best_partition'):
            raise ImportError("Rename CASP/community.py to avoid shadowing python-louvain.")
        _louvain_best_partition = community_mod.best_partition
        import networkx as nx
        _nx = nx
    except ImportError as exc:
        if _cached is not None:
            sys.modules['community'] = _cached
        sys.path = original_path
        raise ImportError(f"python-louvain could not be imported: {exc}") from exc
    finally:
        sys.path = original_path


# ── ECI ───────────────────────────────────────────────────────────────────────

def _mad(h, src, dst):
    if src.numel() == 0:
        return h.new_zeros(())
    return (h[src] - h[dst]).norm(dim=-1).mean()


def compute_eci(h, intra_src, intra_dst, inter_src, inter_dst, eps=1e-8):
    mad_inter = _mad(h, inter_src, inter_dst)
    mad_intra = _mad(h, intra_src, intra_dst)
    return (mad_inter / (mad_intra + eps)).item()


# ── Louvain ───────────────────────────────────────────────────────────────────

def louvain_partition(edge_index, n, random_state=42, resolution=1.0):
    _ensure_louvain()
    G = _nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edge_index.t().tolist())
    part = _louvain_best_partition(G, random_state=random_state, resolution=resolution)
    return torch.tensor([part[i] for i in range(n)], dtype=torch.long)


def louvain_modularity(edge_index, n, labels):
    _ensure_louvain()
    G = _nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edge_index.t().tolist())
    communities = [
        {i for i in range(n) if labels[i].item() == k}
        for k in labels.unique().tolist()
    ]
    return _nx.community.modularity(G, communities)


# ── Edge splitting ────────────────────────────────────────────────────────────

def build_casp_edge_sets(edge_index, community_labels):
    ei, _ = remove_self_loops(edge_index)
    src, dst = ei
    same = community_labels[src] == community_labels[dst]
    return ei[:, same], ei[:, ~same]


def normalise_casp_edges(edge_intra, edge_inter, num_nodes,
                         dtype=torch.float32, improved=False):
    def _norm(ei):
        if ei.size(1) == 0:
            idx = torch.arange(num_nodes, device=ei.device)
            ei_sl = torch.stack([idx, idx])
            ew = torch.zeros(num_nodes, dtype=dtype, device=ei.device)
            return ei_sl, ew
        return gcn_norm(ei, edge_weight=None, num_nodes=num_nodes,
                        improved=improved, add_self_loops=True,
                        flow='source_to_target', dtype=dtype)
    ei_i, ew_i = _norm(edge_intra)
    ei_o, ew_o = _norm(edge_inter)
    return ei_i, ew_i, ei_o, ew_o


# ── δ and p ───────────────────────────────────────────────────────────────────

def compute_delta(edge_intra_raw, edge_inter_raw, y, train_mask):
    def _homophily_on_split(ei):
        src, dst = ei
        both_train = train_mask[src] & train_mask[dst]
        src_tr, dst_tr = src[both_train], dst[both_train]
        if src_tr.numel() == 0:
            return 0.5
        return (y[src_tr] == y[dst_tr]).float().mean().item()
    return _homophily_on_split(edge_intra_raw) - _homophily_on_split(edge_inter_raw)


def compute_p(edge_intra_raw, edge_inter_raw):
    n_intra = edge_intra_raw.size(1)
    n_inter = edge_inter_raw.size(1)
    total = n_intra + n_inter
    return n_intra / total if total > 0 else 0.5


# ── CASP Mixin ────────────────────────────────────────────────────────────────

class CASPMixin(nn.Module):

    def _init_casp(self, ei_intra_norm, ew_intra, ei_inter_norm, ew_inter,
                   ei_intra_raw, ei_inter_raw, p, delta=0.0):
        self.register_buffer('ei_intra_norm', ei_intra_norm)
        self.register_buffer('ew_intra',      ew_intra)
        self.register_buffer('ei_inter_norm', ei_inter_norm)
        self.register_buffer('ew_inter',      ew_inter)
        self.register_buffer('ei_intra_raw',  ei_intra_raw)
        self.register_buffer('ei_inter_raw',  ei_inter_raw)
        self.register_buffer('p',     torch.tensor(float(p)))
        self.register_buffer('delta', torch.tensor(float(delta)))
        init_raw = math.log(math.exp(1.0) - 1.0)
        self.raw_theta = nn.Parameter(torch.tensor(init_raw))

    @property
    def theta(self):
        return F.softplus(self.raw_theta)

    def _get_alphas(self):
        r = torch.exp(self.theta * self.delta)
        denom = r * self.p + (1.0 - self.p)
        return r / denom, 1.0 / denom

    def _casp(self, conv, h, alpha_intra, alpha_inter):
        return (alpha_intra * conv(h, self.ei_intra_norm, self.ew_intra) +
                alpha_inter * conv(h, self.ei_inter_norm, self.ew_inter))

    def casp_stats(self):
        with torch.no_grad():
            r = torch.exp(self.theta * self.delta)
            denom = r * self.p + (1.0 - self.p)
            return {
                'theta':       self.theta.item(),
                'delta':       self.delta.item(),
                'r':           r.item(),
                'p':           self.p.item(),
                'alpha_intra': (r / denom).item(),
                'alpha_inter': (1.0 / denom).item(),
            }


def _gcn_layer(in_ch, out_ch):
    return GCNConv(in_ch, out_ch, add_self_loops=False, normalize=False)


def _casp_init_args(ei_intra_norm, ew_intra, ei_inter_norm, ew_inter,
                    ei_intra_raw, ei_inter_raw, p, delta=0.0):
    return dict(
        ei_intra_norm=ei_intra_norm, ew_intra=ew_intra,
        ei_inter_norm=ei_inter_norm, ew_inter=ew_inter,
        ei_intra_raw=ei_intra_raw,   ei_inter_raw=ei_inter_raw,
        p=p, delta=delta,
    )


# ── CASP Models ───────────────────────────────────────────────────────────────

class CASP_GCN(CASPMixin):
    def __init__(self, in_features, hidden_features, out_features, dropout,
                 ei_intra_norm, ew_intra, ei_inter_norm, ew_inter,
                 ei_intra_raw, ei_inter_raw, p, delta=0.0, t_max=8):
        super().__init__()
        self._init_casp(**_casp_init_args(
            ei_intra_norm, ew_intra, ei_inter_norm, ew_inter,
            ei_intra_raw, ei_inter_raw, p, delta=delta))
        self.dropout = dropout
        self.layers = nn.ModuleList([
            _gcn_layer(in_features if i == 0 else hidden_features, hidden_features)
            for i in range(t_max)])
        self.proj = nn.Linear(hidden_features, out_features)

    def forward(self, x, edge_index, labels=None, train_mask=None):
        ai, ao = self._get_alphas()
        h = x
        for t, layer in enumerate(self.layers):
            h = self._casp(layer, h, ai, ao)
            h = F.relu(h)
            if t < len(self.layers) - 1:
                h = F.dropout(h, self.dropout, training=self.training)
        return F.log_softmax(self.proj(h), dim=-1)


class CASP_APPNP(CASPMixin):
    def __init__(self, in_features, hidden_features, out_features, dropout,
                 ei_intra_norm, ew_intra, ei_inter_norm, ew_inter,
                 ei_intra_raw, ei_inter_raw, p, delta=0.0, K=2, alpha_ppr=0.0, t_max=None):
        super().__init__()
        self._init_casp(**_casp_init_args(
            ei_intra_norm, ew_intra, ei_inter_norm, ew_inter,
            ei_intra_raw, ei_inter_raw, p, delta=delta))
        self.dropout = dropout
        self.lin1 = nn.Linear(in_features, hidden_features)
        self.lin2 = nn.Linear(hidden_features, out_features)
        self.prop_intra = APPNP(K=K, alpha=alpha_ppr)
        self.prop_inter = APPNP(K=K, alpha=alpha_ppr)

    def forward(self, x, edge_index, labels=None, train_mask=None):
        ai, ao = self._get_alphas()
        h = F.dropout(x, self.dropout, training=self.training)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.lin2(h)
        h = (ai * self.prop_intra(h, self.ei_intra_norm) +
             ao * self.prop_inter(h, self.ei_inter_norm))
        return F.log_softmax(h, dim=-1)


class CASP_MixHop(CASPMixin):
    def __init__(self, in_features, hidden_features, out_features, dropout,
                 ei_intra_norm, ew_intra, ei_inter_norm, ew_inter,
                 ei_intra_raw, ei_inter_raw, p, delta=0.0, powers=None, t_max=2):
        super().__init__()
        self._init_casp(**_casp_init_args(
            ei_intra_norm, ew_intra, ei_inter_norm, ew_inter,
            ei_intra_raw, ei_inter_raw, p, delta=delta))
        self.dropout = dropout
        self.powers = powers or [0, 1, 2]
        out_per_layer = len(self.powers) * hidden_features
        self.layers_intra = nn.ModuleList([
            MixHopConv(in_features if i == 0 else out_per_layer,
                       hidden_features, powers=self.powers)
            for i in range(t_max)])
        self.layers_inter = nn.ModuleList([
            MixHopConv(in_features if i == 0 else out_per_layer,
                       hidden_features, powers=self.powers)
            for i in range(t_max)])
        self.proj = nn.Linear(out_per_layer, out_features)

    def forward(self, x, edge_index, labels=None, train_mask=None):
        ai, ao = self._get_alphas()
        h = x
        for t, (l_intra, l_inter) in enumerate(zip(self.layers_intra, self.layers_inter)):
            h = (ai * l_intra(h, self.ei_intra_norm) +
                 ao * l_inter(h, self.ei_inter_norm))
            h = F.relu(h)
            if t < len(self.layers_intra) - 1:
                h = F.dropout(h, self.dropout, training=self.training)
        return F.log_softmax(self.proj(h), dim=-1)


class CASPDirConv(nn.Module):
    def __init__(self, in_ch, out_ch, alpha_dir=0.5):
        super().__init__()
        self.dir_intra = DirGNNConv(GCNConv(in_ch, out_ch))
        self.dir_inter = DirGNNConv(GCNConv(in_ch, out_ch))

    def forward(self, h, ei_intra, ew_intra, ei_inter, ew_inter, alpha_intra, alpha_inter):
        return (alpha_intra * self.dir_intra(h, ei_intra) +
                alpha_inter * self.dir_inter(h, ei_inter))



class CASP_DirGNN(CASPMixin):
    def __init__(self, in_features, hidden_features, out_features, dropout,
                 ei_intra_norm, ew_intra, ei_inter_norm, ew_inter,
                 ei_intra_raw, ei_inter_raw, p, delta=0.0, alpha_dir=0.5, t_max=2):
        super().__init__()
        self._init_casp(**_casp_init_args(
            ei_intra_norm, ew_intra, ei_inter_norm, ew_inter,
            ei_intra_raw, ei_inter_raw, p, delta=delta))
        self.dropout = dropout
        self.layers = nn.ModuleList([
            CASPDirConv(in_features if i == 0 else hidden_features, hidden_features)
            for i in range(t_max)])
        self.proj = nn.Linear(hidden_features, out_features)

    def forward(self, x, edge_index, labels=None, train_mask=None):
        ai, ao = self._get_alphas()
        h = x
        for t, layer in enumerate(self.layers):
            h = layer(h, self.ei_intra_norm, self.ew_intra,
                      self.ei_inter_norm, self.ew_inter, ai, ao)
            h = F.relu(h)
            if t < len(self.layers) - 1:
                h = F.dropout(h, self.dropout, training=self.training)
        return F.log_softmax(self.proj(h), dim=-1)
    




class CASP_FAGCN(CASPMixin):
    def __init__(self, in_features, hidden_features, out_features, dropout,
                 ei_intra_norm, ew_intra, ei_inter_norm, ew_inter,
                 ei_intra_raw, ei_inter_raw, p, delta=0.0, eps=0.3, t_max=4):
        super().__init__()
        self._init_casp(**_casp_init_args(
            ei_intra_norm, ew_intra, ei_inter_norm, ew_inter,
            ei_intra_raw, ei_inter_raw, p, delta=delta))
        self.dropout = dropout
        self.lin_in = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            FAConv(hidden_features, eps=eps, dropout=dropout)
            for _ in range(t_max)])
        self.proj = nn.Linear(hidden_features, out_features)

    def _casp_fa(self, conv, h, x_0, alpha_intra, alpha_inter):
        return (alpha_intra * conv(h, x_0, self.ei_intra_raw) +
                alpha_inter * conv(h, x_0, self.ei_inter_raw))

    def forward(self, x, edge_index, labels=None, train_mask=None):
        ai, ao = self._get_alphas()
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.lin_in(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x_0 = x
        h = x
        for t, layer in enumerate(self.layers):
            h = self._casp_fa(layer, h, x_0, ai, ao)
            if t < len(self.layers) - 1:
                h = F.dropout(h, self.dropout, training=self.training)
        return F.log_softmax(self.proj(h), dim=-1)