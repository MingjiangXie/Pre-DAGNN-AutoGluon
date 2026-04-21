#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import dgl
from dgl.nn import NNConv

from rdkit import Chem, RDLogger
from sklearn.model_selection import StratifiedKFold

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


# =========================================================
# basic utils
# =========================================================
E_DIM = 6
HYB = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 82, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        cublas_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
        if torch.cuda.is_available() and cublas_cfg not in {":4096:8", ":16:8"}:
            warnings.warn(
                "Strict deterministic mode requested, but CUBLAS_WORKSPACE_CONFIG is not set. "
                "Falling back to safe non-strict mode. To force strict mode, launch with "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 python ... --strict-deterministic"
            )
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass
            return
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def _norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def pick_col(df, candidates):
    mapping = {_norm(c): c for c in df.columns}
    if isinstance(candidates, str):
        candidates = [candidates]
    flat = []
    for c in candidates:
        if isinstance(c, (list, tuple)):
            flat.extend(c)
        else:
            flat.append(c)
    for c in flat:
        k = _norm(c)
        if k in mapping:
            return mapping[k]
    for k, raw in mapping.items():
        if any(_norm(c) in k for c in flat):
            return raw
    return None


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(1.0 - np.mean((y_true - y_pred) ** 2) / (np.var(y_true) + 1e-12))
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def format_mean_std(mean_val, std_val, ndigits=3):
    return f"{mean_val:.{ndigits}f} ± {std_val:.{ndigits}f}"


def make_strat_bins(y, n_bins=10):
    y = pd.Series(y).reset_index(drop=True)
    try:
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        if pd.Series(bins).nunique() >= 2:
            return pd.Series(bins).astype(int)
    except Exception:
        pass
    try:
        bins = pd.cut(y, bins=n_bins, labels=False, duplicates="drop")
        if pd.Series(bins).nunique() >= 2:
            return pd.Series(bins).astype(int)
    except Exception:
        pass
    return pd.Series(np.zeros(len(y), dtype=int))


# =========================================================
# graph featurization
# =========================================================
def atom_feat(a):
    Z = a.GetAtomicNum()
    deg = a.GetTotalDegree()
    fc = a.GetFormalCharge()
    arom = 1.0 if a.GetIsAromatic() else 0.0
    ring = 1.0 if a.IsInRing() else 0.0
    hyb = HYB.get(a.GetHybridization(), 5)
    base = [
        min(Z, 100) / 100.0,
        min(deg, 5) / 5.0,
        max(min(fc, 3), -3) / 3.0,
        arom,
        ring,
    ]
    oh = [0] * 6
    oh[min(hyb, 5)] = 1
    return torch.tensor(base + oh, dtype=torch.float32)


def bond_feat(b):
    t = b.GetBondType()
    arom = b.GetIsAromatic() if hasattr(b, "GetIsAromatic") else b.IsAromatic()
    conj = (
        b.GetIsConjugated() if hasattr(b, "GetIsConjugated")
        else (b.IsConjugated() if hasattr(b, "IsConjugated") else False)
    )
    one = [
        1.0 if t == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if t == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if t == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if arom else 0.0,
    ]
    return torch.tensor(
        one + [1.0 if conj else 0.0, 1.0 if b.IsInRing() else 0.0],
        dtype=torch.float32,
    )


def mol_to_graph(mol, add_self_loops=True):
    n = mol.GetNumAtoms()
    atom_x = [atom_feat(mol.GetAtomWithIdx(i)) for i in range(n)]
    src, dst, efeat = [], [], []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_feat(b)
        src += [u, v]
        dst += [v, u]
        efeat += [bf, bf]
    if add_self_loops:
        for i in range(n):
            src.append(i)
            dst.append(i)
            efeat.append(torch.zeros(E_DIM))
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=n)
    g.ndata["h"] = torch.stack(atom_x, dim=0)
    g.edata["e"] = torch.stack(efeat, dim=0)
    return g


# =========================================================
# dataset
# =========================================================
class GraphDescDataset(Dataset):
    def __init__(
        self,
        df,
        selected_features,
        y_mean=None,
        y_std=None,
        d_mean=None,
        d_std=None,
        fit=False,
    ):
        smi_col = pick_col(df, ["SMILES", "smiles"])
        y_col = pick_col(df, [["logBCF", "log_bcf", "logbcf"], ["log_BCF"], ["BCF_log"]])
        if smi_col is None or y_col is None:
            raise ValueError("CSV必须包含 SMILES 与 logBCF 列")

        self.selected_features = list(selected_features)
        self.smiles_raw = df[smi_col].astype(str).tolist()
        self.row_ids_raw = df["row_id"].tolist() if "row_id" in df.columns else list(range(len(df)))
        self.y_raw = df[y_col].astype(float).values

        G, keep, D = [], [], []
        for i, s in enumerate(self.smiles_raw):
            m = Chem.MolFromSmiles(str(s).strip())
            if m is None:
                continue
            g = mol_to_graph(m)
            if g.num_nodes() == 0:
                continue

            desc_values = []
            for feat in self.selected_features:
                if feat not in df.columns:
                    raise SystemExit(f"CSV缺少描述符列：{feat}")
                desc_values.append(df[feat].iloc[i])

            desc = np.array(desc_values, dtype=np.float32)
            G.append(g)
            keep.append(i)
            D.append(desc)

        self.G = G
        self.smiles = [self.smiles_raw[i] for i in keep]
        self.row_ids = [self.row_ids_raw[i] for i in keep]
        self.y_raw = self.y_raw[keep]
        D = np.asarray(D, dtype=np.float32)

        D[~np.isfinite(D)] = np.nan
        med = np.nanmedian(D, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        idx = np.where(np.isnan(D))
        if idx[0].size > 0:
            D[idx] = np.take(med, idx[1])
        if not np.isfinite(D).all():
            D[~np.isfinite(D)] = 0.0

        if fit:
            self.y_mean = float(self.y_raw.mean())
            self.y_std = float(self.y_raw.std() + 1e-8)
            self.d_mean = D.mean(axis=0)
            self.d_std = D.std(axis=0) + 1e-8
        else:
            self.y_mean = float(y_mean)
            self.y_std = float(max(y_std, 1e-8))
            self.d_mean = np.array(d_mean, dtype=np.float32)
            self.d_std = np.array(d_std, dtype=np.float32)

        self.y = torch.tensor((self.y_raw - self.y_mean) / self.y_std, dtype=torch.float32)
        self.D = torch.tensor((D - self.d_mean) / self.d_std, dtype=torch.float32)

    def __len__(self):
        return len(self.G)

    def __getitem__(self, i):
        return self.G[i], self.y[i], self.smiles[i], self.row_ids[i], self.D[i]


def collate(batch):
    gs, ys, smiles, row_ids, Ds = zip(*batch)
    return dgl.batch(gs), torch.stack(ys), list(smiles), list(row_ids), torch.stack(Ds)


# =========================================================
# EMA
# =========================================================
class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].data)

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n].data)
        self.backup = {}


# =========================================================
# model
# =========================================================
class NNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, e_dim=E_DIM, edge_h=64):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(e_dim, edge_h),
            nn.ReLU(),
            nn.Linear(edge_h, in_dim * out_dim),
        )
        self.nnconv = NNConv(in_dim, out_dim, self.edge_mlp, aggregator_type="mean")
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, g, h):
        h = self.nnconv(g, h, g.edata["e"])
        return self.bn(torch.relu(h))


class Encoder(nn.Module):
    def __init__(self, in_dim=11, hid=128, layers=4, dropout=0.12, edge_h=64):
        super().__init__()
        self.layers = nn.ModuleList(
            [NNLayer(in_dim, hid, edge_h=edge_h)]
            + [NNLayer(hid, hid, edge_h=edge_h) for _ in range(layers - 1)]
        )
        self.dropout = nn.Dropout(dropout)
        self.bn_out = nn.BatchNorm1d(hid, momentum=0.05)

    def forward(self, g):
        h = g.ndata["h"]
        for lyr in self.layers:
            h = lyr(g, h)
            h = self.dropout(h)
        g.ndata["h"] = h
        hn = dgl.mean_nodes(g, "h")
        hx = dgl.max_nodes(g, "h")
        hn = self.bn_out(hn)
        return torch.cat([hn, hx], dim=1)


class Head(nn.Module):
    def __init__(self, hid=128, d_dim=5, desc_hidden=32, gate_hidden=64, dropout1=0.28, dropout2=0.18):
        super().__init__()
        self.bn_d = nn.BatchNorm1d(d_dim)
        self.desc_proj = nn.Sequential(
            nn.Linear(d_dim, desc_hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * hid + desc_hidden, gate_hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(gate_hidden, d_dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * hid + d_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(128, 1),
        )

    def forward(self, hg, d):
        d_bn = self.bn_d(d)
        d_proj = self.desc_proj(d_bn)
        gate = 2.0 * torch.sigmoid(self.gate(torch.cat([hg, d_proj], dim=1)))
        d_fused = d_bn * gate
        x = torch.cat([hg, d_fused], dim=1)
        return self.mlp(x).squeeze(1)


# =========================================================
# helpers
# =========================================================
def load_feature_list(path):
    feat_data = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(feat_data, dict) and "features" in feat_data:
        feats = feat_data["features"]
    elif isinstance(feat_data, list):
        feats = feat_data
    else:
        raise SystemExit("features json 格式错误")
    return list(feats)


def prepare_master_df(csv_path, feats):
    df = pd.read_csv(csv_path)

    smi_col = pick_col(df, ["SMILES", "smiles"])
    y_col = pick_col(df, [["logBCF", "log_bcf", "logbcf"], ["log_BCF"], ["BCF_log"]])
    if smi_col is None or y_col is None:
        raise SystemExit("CSV必须包含 SMILES 与 logBCF 列")

    df = df.rename(columns={smi_col: "SMILES", y_col: "logBCF"})
    need_cols = ["SMILES", "logBCF"] + feats
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise SystemExit(f"CSV缺少列：{miss}")

    df = df[need_cols].dropna(subset=["SMILES", "logBCF"]).copy()
    df["SMILES"] = df["SMILES"].astype(str)
    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    df.insert(0, "row_id", np.arange(len(df)))
    return df


def read_outer_membership(path):
    mem = pd.read_csv(path)
    required = ["SMILES", "split_id", "seed", "set"]
    miss = [c for c in required if c not in mem.columns]
    if miss:
        raise SystemExit(f"outer_split_membership.csv 缺少列：{miss}")
    mem["SMILES"] = mem["SMILES"].astype(str)
    return mem


def get_split_pairs(mem, only_split_id=None, only_seed=None):
    tmp = mem[["split_id", "seed"]].drop_duplicates().copy()
    if only_split_id is not None:
        tmp = tmp[tmp["split_id"] == only_split_id]
    if only_seed is not None:
        tmp = tmp[tmp["seed"] == only_seed]
    pairs = list(tmp.sort_values(["split_id", "seed"]).itertuples(index=False, name=None))
    if len(pairs) == 0:
        raise SystemExit("没有匹配到可运行的 split_id / seed")
    return pairs


def extract_split_df(master_df, mem, split_id, seed, set_name):
    sub = mem[
        (mem["split_id"] == split_id)
        & (mem["seed"] == seed)
        & (mem["set"] == set_name)
    ][["SMILES"]].copy()
    sub = sub.reset_index(drop=True)
    sub["_order"] = np.arange(len(sub))

    merged = sub.merge(master_df, on="SMILES", how="left")
    if merged["logBCF"].isna().any():
        miss = merged.loc[merged["logBCF"].isna(), "SMILES"].tolist()[:10]
        raise SystemExit(f"{set_name} 集中有 SMILES 不在主表中，例如：{miss}")

    merged = merged.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return merged


def build_dataset_fit(df, feats):
    fit_ds = GraphDescDataset(df, feats, fit=True)
    return fit_ds, fit_ds.y_mean, fit_ds.y_std, fit_ds.d_mean, fit_ds.d_std


def build_dataset_eval(df, feats, y_mean, y_std, d_mean, d_std):
    return GraphDescDataset(df, feats, y_mean, y_std, d_mean, d_std, fit=False)


def resolve_encoder_meta(args):
    if not os.path.exists(args.encoder_meta):
        raise SystemExit(f"缺少 {args.encoder_meta}")
    meta = json.load(open(args.encoder_meta, "r", encoding="utf-8"))
    return {
        "hid": int(meta.get("hid", 128)),
        "in_dim": int(meta.get("in_dim", 11)),
        "layers": int(meta.get("layers", 4)),
        "edge_h": int(meta.get("edge_h", args.edge_h)),
    }


def init_models(feat_dim, args, device):
    meta = resolve_encoder_meta(args)
    enc = Encoder(
        in_dim=meta["in_dim"],
        hid=meta["hid"],
        layers=meta["layers"],
        dropout=args.dropout,
        edge_h=meta["edge_h"],
    ).to(device)
    head = Head(
        hid=meta["hid"],
        d_dim=feat_dim,
        desc_hidden=args.desc_hidden,
        gate_hidden=args.gate_hidden,
        dropout1=args.head_dropout1,
        dropout2=args.head_dropout2,
    ).to(device)

    if not args.no_tl:
        if not os.path.exists(args.encoder_pretrained):
            raise SystemExit(f"缺少预训练权重：{args.encoder_pretrained}")
        state = torch.load(args.encoder_pretrained, map_location="cpu")
        enc.load_state_dict(state, strict=True)

    return enc, head, meta


def build_optimizer(enc, head, args, frozen):
    if frozen:
        for p in enc.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
        params = list(head.parameters())
        cur_lr = args.lr
    else:
        for p in enc.parameters():
            p.requires_grad = True
        for p in head.parameters():
            p.requires_grad = True
        params = list(enc.parameters()) + list(head.parameters())
        cur_lr = args.lr * args.unfreeze_lr_factor
    opt = torch.optim.AdamW(params, lr=cur_lr, weight_decay=args.weight_decay)
    return opt, params, cur_lr


def build_plateau_scheduler(optimizer, args):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_reduce_factor,
        patience=args.lr_reduce_patience,
        min_lr=args.min_lr,
    )


@torch.no_grad()
def predict(enc, head, loader, inv, device, use_ema=False, ema_enc=None, ema_head=None):
    if use_ema:
        ema_enc.apply_to(enc)
        ema_head.apply_to(head)

    enc.eval()
    head.eval()
    smiles_all, row_ids_all, Ys, Ps = [], [], [], []

    for g, y, smiles, row_ids, D in loader:
        smiles_all += smiles
        row_ids_all += list(row_ids)
        g = g.to(device)
        D = D.to(device)
        p = head(enc(g), D).detach().cpu()
        Ys.append(y.detach().cpu())
        Ps.append(p)

    Y = torch.cat(Ys)
    P = torch.cat(Ps)
    y_true = inv(Y).numpy().reshape(-1)
    y_pred = inv(P).numpy().reshape(-1)

    if use_ema:
        ema_enc.restore(enc)
        ema_head.restore(head)

    return smiles_all, row_ids_all, y_true, y_pred


def evaluate_loader(enc, head, loader, inv, device, use_ema=False, ema_enc=None, ema_head=None):
    _, _, y_true, y_pred = predict(enc, head, loader, inv, device, use_ema, ema_enc, ema_head)
    return calc_metrics(y_true, y_pred), y_true, y_pred


def snapshot_state(enc, head, stats, meta, use_ema=False, ema_enc=None, ema_head=None):
    if use_ema:
        ema_enc.apply_to(enc)
        ema_head.apply_to(head)
    state = {
        "enc": {k: v.detach().cpu() for k, v in enc.state_dict().items()},
        "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "stats": {
            "y_mean": stats["y_mean"],
            "y_std": stats["y_std"],
            "d_mean": stats["d_mean"].tolist(),
            "d_std": stats["d_std"].tolist(),
        },
        "meta": meta,
    }
    if use_ema:
        ema_enc.restore(enc)
        ema_head.restore(head)
    return state


def load_state_for_eval(state_obj, feat_dim, args, device):
    meta = state_obj.get("meta", resolve_encoder_meta(args))
    enc = Encoder(
        in_dim=int(meta.get("in_dim", 11)),
        hid=int(meta.get("hid", 128)),
        layers=int(meta.get("layers", 4)),
        dropout=args.dropout,
        edge_h=int(meta.get("edge_h", args.edge_h)),
    ).to(device)
    head = Head(
        hid=int(meta.get("hid", 128)),
        d_dim=feat_dim,
        desc_hidden=args.desc_hidden,
        gate_hidden=args.gate_hidden,
        dropout1=args.head_dropout1,
        dropout2=args.head_dropout2,
    ).to(device)
    enc.load_state_dict(state_obj["enc"])
    head.load_state_dict(state_obj["head"])
    return enc, head


# =========================================================
# train / cv
# =========================================================
def train_one_epoch(enc, head, loader, opt, params, device, args, ema_enc=None, ema_head=None, epoch=1):
    enc.train()
    head.train()
    loss_list = []
    for g, y, _, _, D in loader:
        g = g.to(device)
        y = y.to(device)
        D = D.to(device)

        pred = head(enc(g), D)
        loss = F.smooth_l1_loss(pred, y, beta=args.huber_beta)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()

        if args.ema and epoch >= args.ema_start:
            if any(p.requires_grad for p in enc.parameters()):
                ema_enc.update(enc)
            ema_head.update(head)

        loss_list.append(float(loss.item()))
    return float(np.mean(loss_list)) if loss_list else np.nan


def train_cv_fold_curve(train_df, valid_df, feats, device, args, seed, verbose_prefix=""):
    set_seed(seed, deterministic=args.strict_deterministic)

    _, y_mean, y_std, d_mean, d_std = build_dataset_fit(train_df, feats)
    tr_ds = build_dataset_eval(train_df, feats, y_mean, y_std, d_mean, d_std)
    va_ds = build_dataset_eval(valid_df, feats, y_mean, y_std, d_mean, d_std)

    tr_loader = DataLoader(
        tr_ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
        num_workers=0, pin_memory=(device == "cuda")
    )
    va_loader = DataLoader(
        va_ds, batch_size=args.batch * 2, shuffle=False, collate_fn=collate,
        num_workers=0, pin_memory=(device == "cuda")
    )

    enc, head, meta = init_models(len(feats), args, device)
    freeze_epochs = 0 if args.no_tl else min(int(args.freeze_epochs), int(args.cv_epochs))
    opt, params, _ = build_optimizer(enc, head, args, frozen=(freeze_epochs > 0))
    scheduler = build_plateau_scheduler(opt, args)

    ema_enc = EMA(enc, decay=args.ema_decay) if args.ema else None
    ema_head = EMA(head, decay=args.ema_decay) if args.ema else None

    inv = lambda t: t * y_std + y_mean
    curve_rows = []
    best_r2 = -1e18
    best_epoch = 1

    for ep in range(1, args.cv_epochs + 1):
        if ep == freeze_epochs + 1 and freeze_epochs > 0:
            opt, params, _ = build_optimizer(enc, head, args, frozen=False)
            scheduler = build_plateau_scheduler(opt, args)

        train_loss = train_one_epoch(enc, head, tr_loader, opt, params, device, args, ema_enc, ema_head, ep)
        use_ema_eval = bool(args.ema and ep >= args.ema_start)
        m_va, _, _ = evaluate_loader(enc, head, va_loader, inv, device, use_ema_eval, ema_enc, ema_head)
        scheduler.step(m_va["R2"])

        curve_rows.append({
            "epoch": ep,
            "train_loss": train_loss,
            "R2_val": m_va["R2"],
            "RMSE_val": m_va["RMSE"],
            "MAE_val": m_va["MAE"],
            "lr": float(opt.param_groups[0]["lr"]),
        })

        if m_va["R2"] > best_r2:
            best_r2 = m_va["R2"]
            best_epoch = ep

        if args.verbose_epoch and ((ep % args.log_interval == 0) or ep == 1 or ep == args.cv_epochs):
            print(
                f"{verbose_prefix}[Epoch {ep:03d}] "
                f"valR2={m_va['R2']:.3f} valRMSE={m_va['RMSE']:.3f} "
                f"loss={train_loss:.4f} lr={opt.param_groups[0]['lr']:.2e}"
            )

    curve_df = pd.DataFrame(curve_rows)
    return {
        "curve_df": curve_df,
        "best_epoch": int(best_epoch),
        "best_r2": float(best_r2),
    }


def pick_epoch_from_cv_curves(curve_df, args):
    agg = curve_df.groupby("epoch", as_index=False).agg(
        R2_val_mean=("R2_val", "mean"),
        R2_val_std=("R2_val", "std"),
        RMSE_val_mean=("RMSE_val", "mean"),
        MAE_val_mean=("MAE_val", "mean"),
    )
    if args.epoch_smooth_window > 1:
        agg["R2_val_smooth"] = agg["R2_val_mean"].rolling(
            window=args.epoch_smooth_window, min_periods=1, center=True
        ).mean()
    else:
        agg["R2_val_smooth"] = agg["R2_val_mean"]

    cand = agg[agg["epoch"] >= args.min_final_epoch].copy()
    if cand.empty:
        cand = agg.copy()
    cand = cand.sort_values(["R2_val_smooth", "R2_val_mean", "RMSE_val_mean"], ascending=[False, False, True])
    chosen = cand.iloc[0].to_dict()
    return int(chosen["epoch"]), agg


def run_inner_cv(train_df, feats, device, args, split_seed, split_id, model_name, outdir):
    train_df = train_df.reset_index(drop=True)
    bins = make_strat_bins(train_df["logBCF"], n_bins=args.bins)
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=split_seed)

    fold_rows = []
    curve_rows = []
    assign_rows = []

    for fold_id, (idx_tr, idx_va) in enumerate(skf.split(train_df, bins), start=1):
        df_tr = train_df.iloc[idx_tr].copy().reset_index(drop=True)
        df_va = train_df.iloc[idx_va].copy().reset_index(drop=True)

        fold_out = train_cv_fold_curve(
            df_tr, df_va, feats, device, args,
            seed=split_seed + fold_id,
            verbose_prefix=f"[Split {split_id:02d} Fold {fold_id:02d}] "
        )

        cur = fold_out["curve_df"].copy()
        cur["split_id"] = split_id
        cur["seed"] = split_seed
        cur["Model"] = model_name
        cur["inner_fold"] = fold_id
        curve_rows.append(cur)

        fold_rows.append({
            "split_id": split_id,
            "seed": split_seed,
            "Model": model_name,
            "inner_fold": fold_id,
            "best_epoch_fold": int(fold_out["best_epoch"]),
            "best_R2_fold": float(fold_out["best_r2"]),
            "n_fold_train": len(df_tr),
            "n_fold_val": len(df_va),
        })

        tmp = df_va[["SMILES", "row_id", "logBCF"]].copy()
        tmp["split_id"] = split_id
        tmp["seed"] = split_seed
        tmp["Model"] = model_name
        tmp["inner_fold"] = fold_id
        assign_rows.append(tmp)

    curve_df = pd.concat(curve_rows, axis=0, ignore_index=True)
    final_epoch, cv_epoch_df = pick_epoch_from_cv_curves(curve_df, args)
    cv_row = cv_epoch_df.loc[cv_epoch_df["epoch"] == final_epoch].iloc[0]
    cv_r2 = float(cv_row["R2_val_mean"])

    fold_df = pd.DataFrame(fold_rows)
    fold_selected = curve_df[curve_df["epoch"] == final_epoch].copy()
    fold_selected = fold_selected[["split_id", "seed", "Model", "inner_fold", "epoch", "R2_val", "RMSE_val", "MAE_val"]]
    fold_selected = fold_selected.rename(columns={
        "epoch": "selected_epoch",
        "R2_val": "R2_fold_val",
        "RMSE_val": "RMSE_fold_val",
        "MAE_val": "MAE_fold_val",
    })
    fold_df = fold_df.merge(fold_selected, on=["split_id", "seed", "Model", "inner_fold"], how="left")

    assign_df = pd.concat(assign_rows, axis=0, ignore_index=True)
    fold_df.to_csv(
        os.path.join(outdir, "cv_details", f"split_{split_id:02d}_seed_{split_seed}_{model_name}_inner{args.cv_folds}fold_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    assign_df.to_csv(
        os.path.join(outdir, "cv_details", f"split_{split_id:02d}_seed_{split_seed}_{model_name}_inner_fold_assignment.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    curve_df.to_csv(
        os.path.join(outdir, "cv_details", f"split_{split_id:02d}_seed_{split_seed}_{model_name}_epoch_curves_long.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    cv_epoch_df.to_csv(
        os.path.join(outdir, "cv_details", f"split_{split_id:02d}_seed_{split_seed}_{model_name}_epoch_curve_mean.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    return cv_r2, final_epoch, fold_df, cv_epoch_df


def train_fixed_epochs(train_df, feats, device, args, seed, fixed_epochs):
    set_seed(seed, deterministic=args.strict_deterministic)

    _, y_mean, y_std, d_mean, d_std = build_dataset_fit(train_df, feats)
    tr_ds = build_dataset_eval(train_df, feats, y_mean, y_std, d_mean, d_std)
    tr_loader = DataLoader(
        tr_ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
        num_workers=0, pin_memory=(device == "cuda")
    )

    enc, head, meta = init_models(len(feats), args, device)
    freeze_epochs = 0 if args.no_tl else min(int(args.freeze_epochs), int(fixed_epochs))
    opt, params, _ = build_optimizer(enc, head, args, frozen=(freeze_epochs > 0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(fixed_epochs), 1), eta_min=args.min_lr)

    ema_enc = EMA(enc, decay=args.ema_decay) if args.ema else None
    ema_head = EMA(head, decay=args.ema_decay) if args.ema else None

    stats = {
        "y_mean": y_mean,
        "y_std": y_std,
        "d_mean": d_mean,
        "d_std": d_std,
    }

    for ep in range(1, fixed_epochs + 1):
        if ep == freeze_epochs + 1 and freeze_epochs > 0:
            opt, params, _ = build_optimizer(enc, head, args, frozen=False)
            remain = max(int(fixed_epochs - ep + 1), 1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remain, eta_min=args.min_lr)

        train_one_epoch(enc, head, tr_loader, opt, params, device, args, ema_enc, ema_head, ep)
        scheduler.step()

    use_ema_save = bool(args.ema and fixed_epochs >= args.ema_start)
    save_state = snapshot_state(enc, head, stats, meta, use_ema_save, ema_enc, ema_head)
    save_state["use_ema_saved"] = use_ema_save
    return save_state


def summarize_mean_std(metrics_df, group_col="Model"):
    metric_cols = [
        "R2_tra", "RMSE_tra", "MAE_tra",
        "R2_val", "RMSE_val", "MAE_val",
        "R2_cv", "Delta_R2"
    ]

    rows = []
    for model, g in metrics_df.groupby(group_col):
        row = {"Model": model, "n_splits": len(g)}
        for c in metric_cols:
            vals = g[c].astype(float).values
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            median_v = float(np.median(vals))
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))
            row[f"{c}_mean"] = mean_v
            row[f"{c}_std"] = std_v
            row[f"{c}_median"] = median_v
            row[f"{c}_min"] = min_v
            row[f"{c}_max"] = max_v
            row[f"{c}_mean±std"] = format_mean_std(mean_v, std_v, 3)
        rows.append(row)

    out = pd.DataFrame(rows)
    if "R2_val_mean" in out.columns:
        out = out.sort_values(["R2_val_mean", "RMSE_val_mean"], ascending=[False, True]).reset_index(drop=True)
    return out


# =========================================================
# main
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Pre-Gate-DAGNN with CV-curve epoch selection")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--outer-split-file", required=True)
    ap.add_argument("--outdir", default="pregate_dagnn_cvmean_results")

    ap.add_argument("--no-tl", action="store_true", help="Use Gate-DAGNN; default is Pre-Gate-DAGNN")
    ap.add_argument("--encoder-meta", type=str, default="encoder_meta_nn.json")
    ap.add_argument("--encoder-pretrained", type=str, default="encoder_kow_nnconv.pt")

    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-epochs", type=int, default=120)
    ap.add_argument("--min-final-epoch", type=int, default=20)
    ap.add_argument("--epoch-smooth-window", type=int, default=5)

    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--unfreeze-lr-factor", type=float, default=0.7)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--lr-reduce-factor", type=float, default=0.6)
    ap.add_argument("--lr-reduce-patience", type=int, default=8)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--huber-beta", type=float, default=0.25)
    ap.add_argument("--freeze-epochs", type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=5.0)

    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--ema-start", type=int, default=10)

    ap.add_argument("--dropout", type=float, default=0.12)
    ap.add_argument("--edge-h", type=int, default=64)
    ap.add_argument("--desc-hidden", type=int, default=32)
    ap.add_argument("--gate-hidden", type=int, default=64)
    ap.add_argument("--head-dropout1", type=float, default=0.28)
    ap.add_argument("--head-dropout2", type=float, default=0.18)

    ap.add_argument("--only-split-id", type=int, default=None)
    ap.add_argument("--only-seed", type=int, default=None)

    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--verbose-epoch", action="store_true")
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--strict-deterministic", action="store_true")

    args = ap.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "per_split"))
    ensure_dir(os.path.join(args.outdir, "predictions"))
    ensure_dir(os.path.join(args.outdir, "cv_details"))
    ensure_dir(os.path.join(args.outdir, "checkpoints"))

    feats = load_feature_list(args.features)
    master_df = prepare_master_df(args.csv, feats)
    membership = read_outer_membership(args.outer_split_file)
    split_pairs = get_split_pairs(membership, args.only_split_id, args.only_seed)

    model_name = "Gate-DAGNN-CVMean" if args.no_tl else "Pre-Gate-DAGNN-CVMean"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_info = {
        "csv": args.csv,
        "features_json": args.features,
        "outer_split_file": args.outer_split_file,
        "outdir": args.outdir,
        "model_name": model_name,
        "n_samples_master": int(len(master_df)),
        "n_features": len(feats),
        "features": feats,
        "bins": args.bins,
        "cv_folds": args.cv_folds,
        "cv_epochs": args.cv_epochs,
        "batch": args.batch,
        "device": device,
        "epoch_selection": "mean CV R2 curve with smoothing, not median(best_epoch)",
        "strict_deterministic": bool(args.strict_deterministic),
    }
    with open(os.path.join(args.outdir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    membership.to_csv(
        os.path.join(args.outdir, "outer_split_membership_used.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    all_split_metrics = []
    all_best_params = []
    all_predictions = []
    all_inner_fold_metrics = []

    print(f"[Model] {model_name}")
    print(f"[Device] {device}")
    print(f"[Splits] {len(split_pairs)}")

    for split_id, seed in split_pairs:
        print(f"\n========== Outer split {split_id:02d} | seed={seed} | {model_name} ==========")

        train_df = extract_split_df(master_df, membership, split_id, seed, "train")
        val_df = extract_split_df(master_df, membership, split_id, seed, "val")
        print(f"[Outer Split] Train={len(train_df)}  Val={len(val_df)}")

        cv_r2, final_epoch, fold_df, cv_epoch_df = run_inner_cv(
            train_df, feats, device, args,
            split_seed=seed, split_id=split_id, model_name=model_name, outdir=args.outdir
        )
        all_inner_fold_metrics.append(fold_df)

        print(f"[Split {split_id:02d}] R2_cv={cv_r2:.3f} | final_epoch(cv-mean)={final_epoch}")

        final_state = train_fixed_epochs(
            train_df, feats, device, args, seed=seed, fixed_epochs=final_epoch
        )

        stats = final_state["stats"]
        y_mean = stats["y_mean"]
        y_std = stats["y_std"]
        d_mean = np.array(stats["d_mean"], dtype=np.float32)
        d_std = np.array(stats["d_std"], dtype=np.float32)
        inv = lambda t: t * y_std + y_mean

        train_ds = build_dataset_eval(train_df, feats, y_mean, y_std, d_mean, d_std)
        val_ds = build_dataset_eval(val_df, feats, y_mean, y_std, d_mean, d_std)

        train_loader = DataLoader(train_ds, batch_size=args.batch * 2, shuffle=False, collate_fn=collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False, collate_fn=collate, num_workers=0)

        enc, head = load_state_for_eval(final_state, len(feats), args, device)
        ss_tr, row_tr, ytr, ptr = predict(enc, head, train_loader, inv, device)
        ss_va, row_va, yva, pva = predict(enc, head, val_loader, inv, device)

        m_tr = calc_metrics(ytr, ptr)
        m_va = calc_metrics(yva, pva)
        delta_r2 = m_tr["R2"] - m_va["R2"]

        print(
            f"[Split {split_id:02d}] "
            f"R2_tra={m_tr['R2']:.3f} RMSE_tra={m_tr['RMSE']:.3f} MAE_tra={m_tr['MAE']:.3f} | "
            f"R2_val={m_va['R2']:.3f} RMSE_val={m_va['RMSE']:.3f} MAE_val={m_va['MAE']:.3f} | "
            f"R2_cv={cv_r2:.3f} | Delta_R2={delta_r2:.3f}"
        )

        if args.save_checkpoints:
            ckpt_prefix = f"{model_name}_split_{split_id:02d}_seed_{seed}"
            torch.save(final_state["enc"], os.path.join(args.outdir, "checkpoints", f"{ckpt_prefix}_encoder.pt"))
            torch.save(final_state["head"], os.path.join(args.outdir, "checkpoints", f"{ckpt_prefix}_head.pt"))
            with open(os.path.join(args.outdir, "checkpoints", f"{ckpt_prefix}_scaler.json"), "w", encoding="utf-8") as f:
                json.dump(final_state["stats"], f, indent=2, ensure_ascii=False)

        metric_row = {
            "split_id": split_id,
            "seed": seed,
            "Model": model_name,
            "best_epoch": final_epoch,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "R2_tra": m_tr["R2"],
            "RMSE_tra": m_tr["RMSE"],
            "MAE_tra": m_tr["MAE"],
            "R2_val": m_va["R2"],
            "RMSE_val": m_va["RMSE"],
            "MAE_val": m_va["MAE"],
            "R2_cv": cv_r2,
            "Delta_R2": delta_r2,
        }
        all_split_metrics.append(metric_row)

        pd.DataFrame([metric_row]).to_csv(
            os.path.join(args.outdir, "per_split", f"{model_name}_split_{split_id:02d}_seed_{seed}_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        pred_train_df = pd.DataFrame({
            "SMILES": ss_tr,
            "row_id": row_tr,
            "split_id": split_id,
            "seed": seed,
            "Model": model_name,
            "set": "train",
            "y_true": ytr,
            "y_pred": ptr,
            "residual": ytr - ptr,
        })
        pred_val_df = pd.DataFrame({
            "SMILES": ss_va,
            "row_id": row_va,
            "split_id": split_id,
            "seed": seed,
            "Model": model_name,
            "set": "val",
            "y_true": yva,
            "y_pred": pva,
            "residual": yva - pva,
        })
        pred_one = pd.concat([pred_train_df, pred_val_df], axis=0, ignore_index=True)
        pred_one.to_csv(
            os.path.join(args.outdir, "predictions", f"{model_name}_split_{split_id:02d}_seed_{seed}_predictions.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        all_predictions.append(pred_one)

        all_best_params.append({
            "split_id": split_id,
            "seed": seed,
            "Model": model_name,
            "best_epoch": final_epoch,
            "config_json": json.dumps({
                "no_tl": args.no_tl,
                "cv_epochs": args.cv_epochs,
                "cv_folds": args.cv_folds,
                "epoch_smooth_window": args.epoch_smooth_window,
                "min_final_epoch": args.min_final_epoch,
                "batch": args.batch,
                "lr": args.lr,
                "unfreeze_lr_factor": args.unfreeze_lr_factor,
                "weight_decay": args.weight_decay,
                "huber_beta": args.huber_beta,
                "freeze_epochs": 0 if args.no_tl else args.freeze_epochs,
                "dropout": args.dropout,
                "edge_h": args.edge_h,
                "desc_hidden": args.desc_hidden,
                "gate_hidden": args.gate_hidden,
                "head_dropout1": args.head_dropout1,
                "head_dropout2": args.head_dropout2,
                "ema": args.ema,
                "ema_decay": args.ema_decay,
                "ema_start": args.ema_start,
            }, ensure_ascii=False),
        })

    metrics_df = pd.DataFrame(all_split_metrics)
    metrics_df.to_csv(
        os.path.join(args.outdir, "all_models_all_splits_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    best_params_df = pd.DataFrame(all_best_params)
    best_params_df.to_csv(
        os.path.join(args.outdir, "all_models_best_params.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    all_predictions_df = pd.concat(all_predictions, axis=0, ignore_index=True)
    all_predictions_df.to_csv(
        os.path.join(args.outdir, "all_models_all_predictions_long.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    all_inner_fold_df = pd.concat(all_inner_fold_metrics, axis=0, ignore_index=True)
    all_inner_fold_df.to_csv(
        os.path.join(args.outdir, "all_models_all_innerfold_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = summarize_mean_std(metrics_df, group_col="Model")
    summary_df.to_csv(
        os.path.join(args.outdir, "summary_mean_std_by_model.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    table_s8 = summary_df[
        [
            "Model",
            "R2_tra_mean±std",
            "RMSE_tra_mean±std",
            "MAE_tra_mean±std",
            "R2_val_mean±std",
            "RMSE_val_mean±std",
            "MAE_val_mean±std",
            "R2_cv_mean±std",
            "Delta_R2_mean±std",
        ]
    ].copy()
    table_s8.columns = [
        "Model",
        "R2_tra",
        "RMSE_tra",
        "MAE_tra",
        "R2_val",
        "RMSE_val",
        "MAE_val",
        "R2_cv",
        "Delta_R2",
    ]
    table_s8.to_csv(
        os.path.join(args.outdir, "Table_S8_repeated_split_mean_std.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDone.")
    print(f"Results saved to: {args.outdir}")
    print("Key files:")
    print("  - all_models_all_splits_metrics.csv")
    print("  - summary_mean_std_by_model.csv")
    print("  - Table_S8_repeated_split_mean_std.csv")
    print("  - all_models_all_predictions_long.csv")
    print("  - all_models_all_innerfold_metrics.csv")


if __name__ == "__main__":
    main()
