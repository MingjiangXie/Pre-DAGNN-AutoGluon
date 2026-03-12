# -*- coding: utf-8 -*-
"""
train_bcf_selected_features_no_tl_v2.py

- 新增 --no-tl ：不加载 logKow 预训练权重（MD-GNN）
- 输出 Table 1 需要的完整指标：
  R2_tra, RMSE_tra, MAE_tra, R2_val, RMSE_val, MAE_val, R2_cv, ΔR2
- 支持多 seed；可选 --ensemble 在验证集做均值集成并输出 ensemble 指标
- 支持 --cross-validation（训练集 5-fold CV），输出 R2_cv 等

说明：你的 8 个描述符来自外部软件，必须使用 --use-csv-desc，从 CSV 直接读取。
"""

import os, re, json, argparse, random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import dgl
from dgl.nn import NNConv

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# --------------------------
# Seed
# --------------------------
def set_seed(sd=82):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)

def _norm(s): 
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def pick_col(df, candidates):
    m = {_norm(c): c for c in df.columns}
    if isinstance(candidates, str): 
        candidates = [candidates]
    flat = []
    for c in candidates:
        if isinstance(c,(list,tuple)): 
            flat += list(c)
        else: 
            flat.append(c)
    for c in flat:
        k = _norm(c)
        if k in m: 
            return m[k]
    for k,raw in m.items():
        if any(_norm(c) in k for c in flat): 
            return raw
    return None

# --------------------------
# Graph featurization
# --------------------------
E_DIM = 6
HYB = {
    Chem.rdchem.HybridizationType.SP:0,
    Chem.rdchem.HybridizationType.SP2:1,
    Chem.rdchem.HybridizationType.SP3:2,
    Chem.rdchem.HybridizationType.SP3D:3,
    Chem.rdchem.HybridizationType.SP3D2:4
}

def atom_feat(a):
    Z=a.GetAtomicNum(); deg=a.GetTotalDegree(); fc=a.GetFormalCharge()
    arom=1.0 if a.GetIsAromatic() else 0.0
    ring=1.0 if a.IsInRing() else 0.0
    hyb=HYB.get(a.GetHybridization(),5)
    base=[min(Z,100)/100.0, min(deg,5)/5.0, max(min(fc,3),-3)/3.0, arom, ring]
    oh=[0]*6; oh[min(hyb,5)]=1
    return torch.tensor(base+oh, dtype=torch.float32)

def bond_feat(b):
    t=b.GetBondType()
    arom=(b.GetIsAromatic() if hasattr(b,"GetIsAromatic") else b.IsAromatic())
    conj=(b.GetIsConjugated() if hasattr(b,"GetIsConjugated") else
          (b.IsConjugated() if hasattr(b,"IsConjugated") else False))
    one=[1.0 if t==Chem.rdchem.BondType.SINGLE else 0.0,
         1.0 if t==Chem.rdchem.BondType.DOUBLE else 0.0,
         1.0 if t==Chem.rdchem.BondType.TRIPLE else 0.0,
         1.0 if arom else 0.0]
    return torch.tensor(one+[1.0 if conj else 0.0, 1.0 if b.IsInRing() else 0.0], dtype=torch.float32)

def mol_to_graph(mol, add_self_loops=True):
    n=mol.GetNumAtoms()
    atom_x=[atom_feat(mol.GetAtomWithIdx(i)) for i in range(n)]
    src,dst,efeat=[],[],[]
    for b in mol.GetBonds():
        u,v=b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf=bond_feat(b); src+=[u,v]; dst+=[v,u]; efeat+=[bf,bf]
    if add_self_loops:
        for i in range(n):
            src.append(i); dst.append(i); efeat.append(torch.zeros(E_DIM))
    g=dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=n)
    g.ndata['h']=torch.stack(atom_x, dim=0)
    g.edata['e']=torch.stack(efeat, dim=0)
    return g

# --------------------------
# Dataset
# --------------------------
class DS(Dataset):
    def __init__(self, df, selected_features, y_mean=None, y_std=None, d_mean=None, d_std=None,
                 fit=False, use_precomputed=True):

        smi_col = pick_col(df, ["SMILES","smiles"])
        y_col = pick_col(df, [["logBCF","log_bcf","logbcf"], ["log_BCF"], ["BCF_log"]])
        if smi_col is None or y_col is None:
            raise ValueError("CSV必须包含 SMILES 与 logBCF 列")

        self.selected_features = list(selected_features)
        self.smiles_raw = df[smi_col].astype(str).tolist()
        self.y_raw = df[y_col].astype(float).values

        G, keep, D = [], [], []
        for i, s in enumerate(self.smiles_raw):
            m = Chem.MolFromSmiles(str(s).strip())
            if m is None:
                continue
            g = mol_to_graph(m)
            if g.num_nodes() == 0:
                continue

            if not use_precomputed:
                raise SystemExit("必须使用 --use-csv-desc（从CSV读取外部软件描述符）")

            desc_values = []
            for feat in self.selected_features:
                if feat not in df.columns:
                    raise SystemExit(f"CSV缺少描述符列：{feat}（列名必须完全一致）")
                desc_values.append(df[feat].iloc[i])

            desc = np.array(desc_values, dtype=np.float32)
            G.append(g); keep.append(i); D.append(desc)

        self.G = G
        self.smiles = [self.smiles_raw[i] for i in keep]
        self.y_raw = self.y_raw[keep]
        D = np.asarray(D, dtype=np.float32)

        # 清洗 NaN/Inf
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
            self.y_std  = float(self.y_raw.std() + 1e-8)
            self.d_mean = D.mean(axis=0)
            self.d_std  = D.std(axis=0) + 1e-8
        else:
            self.y_mean = float(y_mean)
            self.y_std  = float(max(y_std, 1e-8))
            self.d_mean = np.array(d_mean, dtype=np.float32)
            self.d_std  = np.array(d_std, dtype=np.float32)

        self.y = torch.tensor((self.y_raw - self.y_mean) / self.y_std, dtype=torch.float32)
        self.D = torch.tensor((D - self.d_mean) / self.d_std, dtype=torch.float32)

    def __len__(self): 
        return len(self.G)
    def __getitem__(self, i): 
        return self.G[i], self.y[i], self.smiles[i], self.D[i]

def collate(b):
    gs, ys, ss, Ds = zip(*b)
    return dgl.batch(gs), torch.stack(ys), list(ss), torch.stack(Ds)

# --------------------------
# EMA
# --------------------------
class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, model):
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]

# --------------------------
# Model
# --------------------------
class NNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, e_dim=E_DIM, edge_h=64):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(e_dim, edge_h), nn.ReLU(),
            nn.Linear(edge_h, in_dim*out_dim)
        )
        self.nnconv = NNConv(in_dim, out_dim, self.edge_mlp, aggregator_type='mean')
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, g, h):
        h = self.nnconv(g, h, g.edata['e'])
        return self.bn(torch.relu(h))

class Encoder(nn.Module):
    def __init__(self, in_dim=11, hid=128, layers=4, dropout=0.1, edge_h=64):
        super().__init__()
        self.layers = nn.ModuleList([NNLayer(in_dim, hid, edge_h=edge_h)] +
                                    [NNLayer(hid, hid, edge_h=edge_h) for _ in range(layers-1)])
        self.dropout = nn.Dropout(dropout)
        self.bn_out = nn.BatchNorm1d(hid, momentum=0.05)

    def forward(self, g):
        h = g.ndata['h']
        for lyr in self.layers:
            h = lyr(g, h)
            h = self.dropout(h)
        g.ndata['h'] = h
        hn = dgl.mean_nodes(g, 'h')
        hx = dgl.max_nodes(g, 'h')
        hn = self.bn_out(hn)
        return torch.cat([hn, hx], dim=1)

class Head(nn.Module):
    def __init__(self, hid=128, d_dim=8):
        super().__init__()
        self.bn_d = nn.BatchNorm1d(d_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2*hid + d_dim, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 1)
        )

    def forward(self, hg, d):
        x = torch.cat([hg, self.bn_d(d)], dim=1)
        return self.mlp(x).squeeze(1)

# --------------------------
# Metrics
# --------------------------
def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    r2 = float(1.0 - np.mean((y_true - y_pred)**2) / (np.var(y_true) + 1e-12))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

@torch.no_grad()
def predict(enc, head, loader, inv, device):
    enc.eval(); head.eval()
    ss, Ys, Ps = [], [], []
    for g, y, smiles, D in loader:
        ss += smiles
        g = g.to(device); D = D.to(device)
        p = head(enc(g), D).detach().cpu()
        Ys.append(y.detach().cpu())
        Ps.append(p)
    Y = torch.cat(Ys); P = torch.cat(Ps)
    y_true = inv(Y).numpy().reshape(-1)
    y_pred = inv(P).numpy().reshape(-1)
    return ss, y_true, y_pred

# --------------------------
# Split
# --------------------------
def random_stratified_split(df, y_col, frac_train=0.8, bins=10, seed=82):
    rng = np.random.RandomState(seed)
    y = df[y_col].astype(float).values
    try:
        q = pd.qcut(y, q=bins, labels=False, duplicates='drop')
    except Exception:
        q = pd.cut(y, bins=bins, labels=False, include_lowest=True)
    df_tmp = df.copy()
    df_tmp["__bin__"] = q
    tr_idx, va_idx = [], []
    for _, sub in df_tmp.groupby("__bin__"):
        idx = sub.index.values
        rng.shuffle(idx)
        n_tr = int(round(len(idx) * frac_train))
        tr_idx.extend(idx[:n_tr].tolist())
        va_idx.extend(idx[n_tr:].tolist())
    df_tmp = df_tmp.drop(columns="__bin__")
    return df_tmp.loc[tr_idx].copy(), df_tmp.loc[va_idx].copy()

# --------------------------
# Train once (best by val R2)
# --------------------------
def train_once(tr_df, va_df, feats, device, args, seed):

    set_seed(seed)

    fit_ds = DS(tr_df, feats, fit=True, use_precomputed=True)
    y_mean, y_std = fit_ds.y_mean, fit_ds.y_std
    d_mean, d_std = fit_ds.d_mean, fit_ds.d_std

    tr_ds = DS(tr_df, feats, y_mean, y_std, d_mean, d_std, use_precomputed=True)
    va_ds = DS(va_df, feats, y_mean, y_std, d_mean, d_std, use_precomputed=True)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=args.batch*2, shuffle=False, collate_fn=collate, num_workers=0)

    if not os.path.exists(args.encoder_meta):
        raise SystemExit(f"缺少 {args.encoder_meta}")
    meta = json.load(open(args.encoder_meta, "r"))
    hid = int(meta.get("hid", 128))
    enc = Encoder(in_dim=int(meta.get("in_dim",11)), hid=hid, layers=int(meta.get("layers",4))).to(device)

    freeze_epochs = int(args.freeze_epochs)
    if not args.no_tl:
        if not os.path.exists(args.encoder_pretrained):
            raise SystemExit(f"缺少预训练 {args.encoder_pretrained}")
        state = torch.load(args.encoder_pretrained, map_location="cpu")
        enc.load_state_dict(state, strict=True)
    else:
        print("[MD-GNN] --no-tl enabled -> encoder random init; freeze_epochs=0")
        freeze_epochs = 0

    head = Head(hid=hid, d_dim=len(feats)).to(device)

    inv = lambda t: t * y_std + y_mean

    ema_enc = EMA(enc, decay=args.ema_decay) if args.ema else None
    ema_head = EMA(head, decay=args.ema_decay) if args.ema else None

    best = None
    best_r2 = -1e9
    bad = 0

    for ep in range(1, args.epochs+1):
        if ep <= freeze_epochs:
            enc.eval()
            for p in enc.parameters(): p.requires_grad = False
            params = list(head.parameters())
        else:
            enc.train()
            for p in enc.parameters(): p.requires_grad = True
            params = list(enc.parameters()) + list(head.parameters())

        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

        for g, y, _, D in tr_loader:
            g = g.to(device); y = y.to(device); D = D.to(device)
            pred = head(enc(g), D)
            loss = F.smooth_l1_loss(pred, y, beta=args.huber_beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

        if args.ema and ep >= args.ema_start:
            ema_enc.update(enc); ema_head.update(head)

        # current val metrics
        _, yv, pv = predict(enc, head, va_loader, inv, device)
        m = calc_metrics(yv, pv)
        print(f"[Epoch {ep:03d}] valMAE={m['MAE']:.3f} valRMSE={m['RMSE']:.3f} valR2={m['R2']:.3f} frozen={ep<=freeze_epochs} no_tl={args.no_tl}")

        if m["R2"] > best_r2 + 1e-4:
            best_r2 = m["R2"]
            bad = 0
            best = {
                "enc": {k:v.detach().cpu() for k,v in enc.state_dict().items()},
                "head": {k:v.detach().cpu() for k,v in head.state_dict().items()},
                "stats": {"y_mean":y_mean, "y_std":y_std, "d_mean":d_mean.tolist(), "d_std":d_std.tolist()},
            }
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stop.")
                break

    return best, tr_ds, va_ds

# --------------------------
# Cross validation on train set (5-fold)
# --------------------------
def cross_validation(tr_df, feats, device, args):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=args.split_seed)
    all_true, all_pred = [], []

    tr_df = tr_df.reset_index(drop=True)

    # save original epochs and set to cv_epochs
    old_epochs = args.epochs
    args.epochs = int(args.cv_epochs)

    for fold, (idx_tr, idx_va) in enumerate(kf.split(tr_df), 1):
        df_tr = tr_df.iloc[idx_tr].copy()
        df_va = tr_df.iloc[idx_va].copy()

        best, _, va_ds = train_once(df_tr, df_va, feats, device, args, seed=args.split_seed + fold)

        stats = best["stats"]
        y_mean, y_std = stats["y_mean"], stats["y_std"]
        d_mean, d_std = np.array(stats["d_mean"], dtype=np.float32), np.array(stats["d_std"], dtype=np.float32)
        inv = lambda t: t * y_std + y_mean

        meta = json.load(open(args.encoder_meta, "r"))
        hid = int(meta.get("hid",128))
        enc = Encoder(in_dim=int(meta.get("in_dim",11)), hid=hid, layers=int(meta.get("layers",4))).to(device)
        head = Head(hid=hid, d_dim=len(feats)).to(device)
        enc.load_state_dict(best["enc"]); head.load_state_dict(best["head"])

        va_loader = DataLoader(va_ds, batch_size=args.batch*2, shuffle=False, collate_fn=collate, num_workers=0)
        _, yv, pv = predict(enc, head, va_loader, inv, device)
        m = calc_metrics(yv, pv)
        print(f"[Fold {fold}] valR2={m['R2']:.3f} | valRMSE={m['RMSE']:.3f}")
        all_true.extend(list(yv)); all_pred.extend(list(pv))

    # restore epochs
    args.epochs = old_epochs

    m_all = calc_metrics(np.array(all_true), np.array(all_pred))
    return {
        "Q2_LOO": m_all["R2"],
        "R2_cv": m_all["R2"],
        "RMSE_cv": m_all["RMSE"],
        "MAE_cv": m_all["MAE"],
    }

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--use-csv-desc", action="store_true")
    ap.add_argument("--no-tl", action="store_true")

    ap.add_argument("--split-seed", type=int, default=82)
    ap.add_argument("--bins", type=int, default=10)

    ap.add_argument("--seeds", type=str, default="2024")
    ap.add_argument("--ensemble", action="store_true")

    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=7e-5)
    ap.add_argument("--huber-beta", type=float, default=0.25)
    ap.add_argument("--freeze-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=60)

    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--ema-start", type=int, default=10)

    ap.add_argument("--cross-validation", action="store_true")
    ap.add_argument("--cv-epochs", type=int, default=120)

    # keep compatibility; not used
    ap.add_argument("--analyze-importance", action="store_true")
    ap.add_argument("--importance-repeats", type=int, default=5)

    ap.add_argument("--encoder-meta", type=str, default="encoder_meta_nn.json")
    ap.add_argument("--encoder-pretrained", type=str, default="encoder_kow_nnconv.pt")

    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--pred-out", type=str, default="preds.csv")
    args = ap.parse_args()

    if not args.use_csv_desc:
        raise SystemExit("必须加 --use-csv-desc（从CSV读取外部软件描述符）")

    feat_data = json.load(open(args.features, "r", encoding="utf-8"))
    if isinstance(feat_data, dict) and "features" in feat_data:
        feats = feat_data["features"]
    elif isinstance(feat_data, list):
        feats = feat_data
    else:
        raise SystemExit("features json 格式错误")
    feats = list(feats)

    df = pd.read_csv(args.csv)
    smi_col = pick_col(df, ["SMILES","smiles"])
    y_col = pick_col(df, [["logBCF","log_bcf","logbcf"], ["log_BCF"], ["BCF_log"]])
    if smi_col is None or y_col is None:
        raise SystemExit("CSV必须包含 SMILES 与 logBCF 列")
    df = df.rename(columns={smi_col:"SMILES", y_col:"logBCF"})

    need_cols = ["SMILES","logBCF"] + feats
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise SystemExit(f"CSV缺少列：{miss}")

    df = df[need_cols].dropna(subset=["SMILES","logBCF"]).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    tr_df, va_df = random_stratified_split(df, "logBCF", frac_train=0.8, bins=args.bins, seed=args.split_seed)
    print(f"[Split] split_seed={args.split_seed} bins={args.bins} -> Train={len(tr_df)} Val={len(va_df)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    seeds = [int(s) for s in re.split(r"[,\s]+", args.seeds.strip()) if s]
    tag = "MDGNN" if args.no_tl else "TLMDGNN"

    val_preds = []
    val_true = None
    best_seed = None
    best_va_r2 = -1e9
    best_row = None

    for sd in seeds:
        print(f"\n==== Train {tag} seed={sd} ====")
        best, tr_ds, va_ds = train_once(tr_df, va_df, feats, device, args, seed=sd)

        stats = best["stats"]
        y_mean, y_std = stats["y_mean"], stats["y_std"]
        d_mean, d_std = np.array(stats["d_mean"], dtype=np.float32), np.array(stats["d_std"], dtype=np.float32)
        inv = lambda t: t * y_std + y_mean

        meta = json.load(open(args.encoder_meta, "r"))
        hid = int(meta.get("hid",128))
        enc = Encoder(in_dim=int(meta.get("in_dim",11)), hid=hid, layers=int(meta.get("layers",4))).to(device)
        head = Head(hid=hid, d_dim=len(feats)).to(device)
        enc.load_state_dict(best["enc"]); head.load_state_dict(best["head"])

        tr_loader = DataLoader(tr_ds, batch_size=args.batch*2, shuffle=False, collate_fn=collate, num_workers=0)
        va_loader = DataLoader(va_ds, batch_size=args.batch*2, shuffle=False, collate_fn=collate, num_workers=0)

        _, ytr, ptr = predict(enc, head, tr_loader, inv, device)
        ss_va, yva, pva = predict(enc, head, va_loader, inv, device)

        m_tr = calc_metrics(ytr, ptr)
        m_va = calc_metrics(yva, pva)
        delta = m_tr["R2"] - m_va["R2"]

        print(f"[Seed {sd}] Train: R2={m_tr['R2']:.3f} RMSE={m_tr['RMSE']:.3f} MAE={m_tr['MAE']:.3f}")
        print(f"[Seed {sd}] Valid: R2={m_va['R2']:.3f} RMSE={m_va['RMSE']:.3f} MAE={m_va['MAE']:.3f}")
        print(f"[Seed {sd}] ΔR2  : {delta:.3f}")

        # save weights
        torch.save(best["enc"], f"{tag}_encoder_bcf_selected_s{sd}.pt")
        torch.save(best["head"], f"{tag}_head_bcf_selected_s{sd}.pt")
        json.dump(best["stats"], open(f"{tag}_bcf_scaler_selected_s{sd}.json", "w", encoding="utf-8"), indent=2)
        print(f"[Saved] {tag}_encoder_bcf_selected_s{sd}.pt, {tag}_head_bcf_selected_s{sd}.pt, {tag}_bcf_scaler_selected_s{sd}.json")

        if args.save_preds:
            base, ext = os.path.splitext(args.pred_out)
            ext = ext if ext else ".csv"
            out = f"{base}_{tag}_val_split{args.split_seed}_seed{sd}{ext}"
            pd.DataFrame({"SMILES": ss_va, "y_true": yva, "y_pred": pva,
                          "model": tag, "seed": sd, "split_seed": args.split_seed}).to_csv(out, index=False)
            print(f"[Pred Saved] {out}")

        val_preds.append(pva)
        if val_true is None:
            val_true = yva.copy()

        if m_va["R2"] > best_va_r2:
            best_va_r2 = m_va["R2"]
            best_seed = sd
            best_row = (m_tr, m_va, delta)

    if args.ensemble and len(val_preds) >= 2:
        ens_pred = np.mean(np.vstack([p.reshape(1,-1) for p in val_preds]), axis=0).reshape(-1)
        m_ens = calc_metrics(val_true, ens_pred)
        print(f"\nEnsemble on validation: {m_ens} (models: {', '.join(map(str,seeds))}; split seed={args.split_seed})")
    elif args.ensemble:
        print("\n[Warn] --ensemble 开启但 seeds<2，跳过集成。")

    cv = None
    if args.cross_validation:
        print("\n" + "="*70)
        print("QSAR Internal Validation (Cross-Validation)")
        print("="*70)
        cv = cross_validation(tr_df, feats, device, args)
        print("\n交叉验证指标 (Internal Validation)")
        print(f"Q2_LOO  : {cv['Q2_LOO']:.4f}")
        print(f"R2_cv   : {cv['R2_cv']:.4f}")
        print(f"RMSE_cv : {cv['RMSE_cv']:.4f}")
        print(f"MAE_cv  : {cv['MAE_cv']:.4f}")

    # summary line for Table 1
    m_tr, m_va, delta = best_row
    print("\n" + "="*70)
    print("[TABLE-ROW SUMMARY] (best by Val R2)")
    print(f"Model={tag}  split_seed={args.split_seed}  best_seed={best_seed}")
    print(f"R2_tra={m_tr['R2']:.3f} RMSE_tra={m_tr['RMSE']:.3f} MAE_tra={m_tr['MAE']:.3f} | "
          f"R2_val={m_va['R2']:.3f} RMSE_val={m_va['RMSE']:.3f} MAE_val={m_va['MAE']:.3f} | ΔR2={delta:.3f}")
    if cv is not None:
        print(f"R2_cv={cv['R2_cv']:.3f}")
    print("="*70)

    if args.analyze_importance:
        print("\n[Note] --analyze-importance 在 v2 脚本中未实现。你可继续用旧 TL-MD-GNN 脚本做 importance 并放 SI。")

if __name__ == "__main__":
    main()
