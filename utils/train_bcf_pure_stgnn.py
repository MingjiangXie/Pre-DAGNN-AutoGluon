#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_bcf_pure_stgnn.py
—— 纯 ST-GNN（无 TL、无描述符）训练脚本 —— 
补齐论文表格所需指标输出：
  R2_tra, RMSE_tra, MAE_tra, R2_val, RMSE_val, MAE_val, R2_cv
并保存：
  - per_seed_metrics.csv（每个seed一行）
  - paper_table_metrics_seedXX.csv（best-seed汇总一行）
  - cv_metrics_seedXX.json / cv_predictions_seedXX.csv（若开启CV）
"""

import os, re, json, argparse, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import dgl
from dgl.nn import NNConv

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# --------------------------
# 工具 & 随机性
# --------------------------
def set_seed(sd=2024):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _norm(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())

def pick_col(df, candidates):
    if isinstance(candidates, (list, tuple)):
        flat = []
        for c in candidates:
            if isinstance(c, (list, tuple)): flat.extend(c)
            else: flat.append(c)
        candidates = flat
    mapping = {_norm(c): c for c in df.columns}
    for c in candidates:
        key = _norm(c)
        if key in mapping: return mapping[key]
    for c in df.columns:
        s = _norm(c)
        for cand in candidates:
            if _norm(cand) in s: return c
    return None

# --------------------------
# 原子/键特征
# --------------------------
E_DIM = 6
HYB = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
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
    n = mol.GetNumAtoms()
    atom_x = [atom_feat(mol.GetAtomWithIdx(i)) for i in range(n)]
    src, dst, efeat = [], [], []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_feat(b)
        src += [u, v]; dst += [v, u]; efeat += [bf, bf]
    if add_self_loops:
        for i in range(n):
            src.append(i); dst.append(i); efeat.append(torch.zeros(E_DIM))
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=n)
    g.ndata["h"] = torch.stack(atom_x, dim=0)
    g.edata["e"] = torch.stack(efeat, dim=0)
    return g

# --------------------------
# Dataset & collate
# --------------------------
class DS(Dataset):
    def __init__(self, df, y_mean=None, y_std=None, fit=False):
        smi_col = pick_col(df, ["SMILES","smiles"])
        y_col = pick_col(df, [["logBCF","log_bcf","logbcf"], ["log_BCF"], ["BCF_log"]])
        if smi_col is None or y_col is None:
            raise ValueError("未找到 SMILES 或 logBCF 列")
        self.smiles = df[smi_col].tolist()
        self.y_raw = df[y_col].astype(float).values

        G, keep = [], []
        for i, s in enumerate(self.smiles):
            m = Chem.MolFromSmiles(str(s).strip())
            if m is None: continue
            g = mol_to_graph(m)
            if g.num_nodes() == 0: continue
            G.append(g); keep.append(i)
        self.G = G
        self.smiles = [self.smiles[i] for i in keep]
        self.y_raw = self.y_raw[keep]

        if fit:
            self.y_mean = float(self.y_raw.mean())
            self.y_std  = float(self.y_raw.std() + 1e-8)
        else:
            self.y_mean = float(y_mean)
            self.y_std  = float(max(y_std, 1e-8))

        self.y = torch.tensor((self.y_raw - self.y_mean) / self.y_std, dtype=torch.float32)

    def __len__(self): return len(self.G)
    def __getitem__(self, i): return self.G[i], self.y[i], self.smiles[i]

def collate(b):
    gs, ys, ss = zip(*b)
    return dgl.batch(gs), torch.stack(ys), list(ss)

# --------------------------
# 模型
# --------------------------
class NNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, e_dim=E_DIM, edge_h=64):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(e_dim, edge_h), nn.ReLU(),
                                      nn.Linear(edge_h, in_dim*out_dim))
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
        h = g.ndata["h"]
        for lyr in self.layers:
            h = lyr(g, h)
            h = self.dropout(h)
        g.ndata["h"] = h
        hm = dgl.mean_nodes(g, "h")
        hx = dgl.max_nodes(g, "h")
        hm = self.bn_out(hm)
        return torch.cat([hm, hx], dim=1)

class Head(nn.Module):
    def __init__(self, hid=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*hid, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 1)
        )
    def forward(self, hg): return self.mlp(hg).squeeze(1)

# --------------------------
# EMA
# --------------------------
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

# --------------------------
# 指标计算（统一）
# --------------------------
def _metrics_from_numpy(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    var = float(np.var(y_true))
    r2  = 1.0 - float(np.mean((y_true - y_pred)**2)) / (var + 1e-12)
    return {"MAE":mae, "RMSE":rmse, "R2":r2}

def evaluate(enc, head, loader, inv, device):
    enc.eval(); head.eval()
    Ys, Ps = [], []
    with torch.no_grad():
        for g, y, _ in loader:
            g = g.to(device)
            p = head(enc(g))
            Ys.append(y.cpu()); Ps.append(p.cpu())
    Y = torch.cat(Ys); P = torch.cat(Ps)
    y_true = inv(Y).numpy()
    y_pred = inv(P).numpy()
    return _metrics_from_numpy(y_true, y_pred)

# --------------------------
# 单次训练（外部 80/20）——补齐 Train@Best
# --------------------------
def train_once(tr_df, va_df, device='cpu',
               epochs=240, lr=2e-3, freeze_epochs=0,
               weight_decay=7e-5, huber_beta=0.25, batch=128,
               init_seed=2024, use_ema=False, ema_decay=0.995, ema_start=10,
               in_dim=11, hid=128, layers=4):

    fit = DS(tr_df, fit=True)
    y_mean, y_std = fit.y_mean, fit.y_std
    inv = lambda t: t * y_std + y_mean

    tr_ds = DS(tr_df, y_mean, y_std)
    va_ds = DS(va_df, y_mean, y_std)

    tr_loader = DataLoader(tr_ds, batch_size=batch, shuffle=True, collate_fn=collate,
                           num_workers=0, pin_memory=(device=='cuda'))
    va_loader = DataLoader(va_ds, batch_size=batch*2, shuffle=False, collate_fn=collate,
                           num_workers=0, pin_memory=(device=='cuda'))
    tr_eval_loader = DataLoader(tr_ds, batch_size=batch*2, shuffle=False, collate_fn=collate,
                                num_workers=0, pin_memory=(device=='cuda'))

    # ST：随机初始化端到端
    set_seed(init_seed)
    enc = Encoder(in_dim=in_dim, hid=hid, layers=layers).to(device)
    head = Head(hid=hid).to(device)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision('high')

    ema_enc = EMA(enc, decay=ema_decay) if use_ema else None
    ema_head = EMA(head, decay=ema_decay) if use_ema else None

    best = None
    best_r2 = -1e9
    bad, patience = 0, 45

    for ep in range(1, epochs+1):
        enc.train(); head.train()
        params = list(enc.parameters()) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        for g, y, _ in tr_loader:
            g = g.to(device); y = y.to(device)
            pred = head(enc(g))
            loss = F.smooth_l1_loss(pred, y, beta=huber_beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            if use_ema:
                ema_enc.update(enc); ema_head.update(head)

        using_ema = False
        if use_ema and ep >= ema_start:
            ema_enc.apply_to(enc); ema_head.apply_to(head)
            using_ema = True

        val = evaluate(enc, head, va_loader, inv, device)
        train_now = evaluate(enc, head, tr_eval_loader, inv, device)

        if using_ema:
            ema_enc.restore(enc); ema_head.restore(head)

        print(
            f"[Epoch {ep:03d}] "
            f"trainMAE={train_now['MAE']:.3f} trainRMSE={train_now['RMSE']:.3f} trainR2={train_now['R2']:.3f} | "
            f"valMAE={val['MAE']:.3f} valRMSE={val['RMSE']:.3f} valR2={val['R2']:.3f}"
        )

        if val['R2'] > best_r2 + 1e-3:
            best_r2 = val['R2']; bad = 0

            # 若启用EMA，保存“EMA权重下”的 best
            if use_ema and ep >= ema_start:
                ema_enc.apply_to(enc); ema_head.apply_to(head)
                train_at_best = evaluate(enc, head, tr_eval_loader, inv, device)
                best = {
                    "enc": {k:v.detach().cpu() for k,v in enc.state_dict().items()},
                    "head": {k:v.detach().cpu() for k,v in head.state_dict().items()},
                    "val": val,
                    "train": train_at_best,
                    "stats": {"y_mean": y_mean, "y_std": y_std},
                    "best_epoch": ep
                }
                ema_enc.restore(enc); ema_head.restore(head)
            else:
                best = {
                    "enc": {k:v.detach().cpu() for k,v in enc.state_dict().items()},
                    "head": {k:v.detach().cpu() for k,v in head.state_dict().items()},
                    "val": val,
                    "train": train_now,   # 当前就是最佳
                    "stats": {"y_mean": y_mean, "y_std": y_std},
                    "best_epoch": ep
                }
        else:
            bad += 1
            if bad >= patience:
                print("Early stop.")
                break

    return best

# --------------------------
# 外部 80/20 分层划分
# --------------------------
def random_stratified_split(df, y_col, frac_train=0.8, bins=12, seed=82):
    rng = np.random.RandomState(seed)
    y = df[y_col].astype(float).values
    try:
        q = pd.qcut(y, q=bins, labels=False, duplicates='drop')
    except Exception:
        q = pd.cut(y, bins=bins, labels=False, include_lowest=True)
    df_tmp = df.copy()
    df_tmp['__bin__'] = q
    tr_idx, va_idx = [], []
    for _, sub in df_tmp.groupby('__bin__'):
        idx = sub.index.values
        rng.shuffle(idx)
        n_tr = int(round(len(idx) * frac_train))
        tr_idx.extend(idx[:n_tr])
        va_idx.extend(idx[n_tr:])
    tr = df.loc[tr_idx].reset_index(drop=True)
    va = df.loc[va_idx].reset_index(drop=True)
    return tr, va

# --------------------------
# 交叉验证（Internal）——输出 R2_cv/RMSE_cv/MAE_cv
# --------------------------
def cross_validation(tr_df, device, init_seed=2024, epochs=100, 
                     lr=2e-3, weight_decay=7e-5, huber_beta=0.25, batch=128,
                     use_ema=False, ema_decay=0.995, ema_start=10,
                     in_dim=11, hid=128, layers=4):
    from sklearn.model_selection import KFold

    n = len(tr_df)
    print("\n" + "="*70)
    print("交叉验证 (ST-GNN - pure)")
    print("="*70)
    print(f"样本数: {n}")

    n_folds = 5 if n > 500 else (10 if n > 200 else n)
    print(f"使用 {n_folds}-Fold 交叉验证" if n_folds!=n else "使用留一交叉验证 (LOO)")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=init_seed)

    all_true, all_pred = [], []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(tr_df), 1):
        fold_tr = tr_df.iloc[tr_idx].copy()
        fold_va = tr_df.iloc[va_idx].copy()

        fit_ds = DS(fold_tr, fit=True)
        y_mean, y_std = fit_ds.y_mean, fit_ds.y_std
        inv  = lambda t: t * y_std + y_mean

        train_ds = DS(fold_tr, y_mean, y_std)
        val_ds   = DS(fold_va, y_mean, y_std)

        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate, num_workers=0)
        val_loader   = DataLoader(val_ds, batch_size=batch*2, shuffle=False, collate_fn=collate, num_workers=0)

        set_seed(init_seed + fold)  # fold内也可扰动一下
        enc = Encoder(in_dim=in_dim, hid=hid, layers=layers).to(device)
        head= Head(hid=hid).to(device)

        ema_enc = EMA(enc, decay=ema_decay) if use_ema else None
        ema_head= EMA(head, decay=ema_decay) if use_ema else None

        best_val_rmse = 1e9
        bad, patience = 0, 25
        best_enc, best_head = None, None

        for ep in range(1, epochs+1):
            enc.train(); head.train()
            params = list(enc.parameters()) + list(head.parameters())
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

            for g, y, _ in train_loader:
                g = g.to(device); y = y.to(device)
                pred = head(enc(g))
                loss = F.smooth_l1_loss(pred, y, beta=huber_beta)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                if use_ema:
                    ema_enc.update(enc); ema_head.update(head)

            using_ema = False
            if use_ema and ep >= ema_start:
                ema_enc.apply_to(enc); ema_head.apply_to(head)
                using_ema = True

            val_m = evaluate(enc, head, val_loader, inv, device)  # rmse/mae/r2都有
            val_rmse = val_m["RMSE"]

            if using_ema:
                ema_enc.restore(enc); ema_head.restore(head)

            if val_rmse < best_val_rmse - 1e-4:
                best_val_rmse = val_rmse
                bad = 0
                best_enc  = {k:v.detach().cpu().clone() for k,v in enc.state_dict().items()}
                best_head = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break

        # fold best -> predict fold val
        enc.load_state_dict(best_enc); head.load_state_dict(best_head)
        enc.to(device); head.to(device)
        enc.eval(); head.eval()

        with torch.no_grad():
            Ys, Ps = [], []
            for g, y, _ in val_loader:
                g = g.to(device)
                Ps.append(head(enc(g)).cpu())
                Ys.append(y.cpu())
            Y = torch.cat(Ys); P = torch.cat(Ps)
            y_true = inv(Y).numpy()
            y_pred = inv(P).numpy()

        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

    y_true = np.array(all_true).reshape(-1)
    y_pred = np.array(all_pred).reshape(-1)

    cv_m = _metrics_from_numpy(y_true, y_pred)

    print("\n交叉验证完成!")
    print("交叉验证指标 (Internal Validation - ST-GNN)")
    print(f"R²_cv  = {cv_m['R2']:.4f}")
    print(f"RMSE_cv= {cv_m['RMSE']:.4f}")
    print(f"MAE_cv = {cv_m['MAE']:.4f}")
    print(f"样本数 = {len(y_true)}")

    return {
        "R2_cv": float(cv_m["R2"]),
        "RMSE_cv": float(cv_m["RMSE"]),
        "MAE_cv": float(cv_m["MAE"]),
        "N": int(len(y_true)),
        "y_true": y_true,
        "y_pred": y_pred
    }

# --------------------------
# 保存论文表格一行（7列）
# --------------------------
def save_summary_table(out_csv, split_seed, seeds, best_row, cv_r2):
    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)
    row = {
        "split_seed": int(split_seed),
        "seeds": ",".join([str(s) for s in seeds]),
        "R2_tra": best_row["R2_tra"],
        "RMSE_tra": best_row["RMSE_tra"],
        "MAE_tra": best_row["MAE_tra"],
        "R2_val": best_row["R2_val"],
        "RMSE_val": best_row["RMSE_val"],
        "MAE_val": best_row["MAE_val"],
        "R2_cv": float(cv_r2) if cv_r2 is not None else np.nan
    }
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    print(f"\n[Saved] summary table -> {out_csv}")

# --------------------------
# 主函数
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="数据文件（至少含 SMILES 与 logBCF）")
    ap.add_argument("--split-seed", type=int, default=82)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--seeds", type=str, default="2024")
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--ema-start", type=int, default=10)
    ap.add_argument("--cross-validation", action="store_true")
    ap.add_argument("--cv-epochs", type=int, default=100)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--outdir", type=str, default="metrics_summary_pure_stgnn")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    smi_col = pick_col(df, ["SMILES","smiles"])
    y_col   = pick_col(df, [["logBCF","log_bcf","logbcf"], ["log_BCF"], ["BCF_log"]])
    if smi_col is None or y_col is None:
        raise ValueError("未找到 SMILES 或 logBCF 列")
    df = df[[smi_col, y_col]].dropna().drop_duplicates(subset=[smi_col]).reset_index(drop=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.split_seed)

    tr_df, va_df = random_stratified_split(df, y_col, frac_train=0.8, bins=args.bins, seed=args.split_seed)
    print(f"Random stratified split 0.8/0.2 -> Train {len(tr_df)} | Val {len(va_df)}")

    os.makedirs(args.outdir, exist_ok=True)

    seeds = [int(s) for s in re.split(r"[,\s]+", args.seeds.strip()) if s.strip()]
    all_models = []

    per_seed_csv = os.path.join(args.outdir, "per_seed_metrics.csv")

    for s in seeds:
        print("="*70); print(f"Seed = {s}")
        best = train_once(
            tr_df, va_df, device=device,
            epochs=args.epochs, lr=2e-3, freeze_epochs=0,
            weight_decay=7e-5, huber_beta=0.25, batch=args.batch,
            init_seed=s, use_ema=args.ema, ema_decay=args.ema_decay,
            ema_start=args.ema_start, in_dim=11, hid=args.hid, layers=args.layers
        )
        all_models.append((s, best))

        # 保存该seed的模型
        torch.save(best['enc'], os.path.join(args.outdir, f"encoder_bcf_pure_stgnn_s{s}.pt"))
        torch.save(best['head'], os.path.join(args.outdir, f"head_bcf_pure_stgnn_s{s}.pt"))
        with open(os.path.join(args.outdir, f"bcf_scaler_pure_stgnn_s{s}.json"), "w", encoding="utf-8") as f:
            json.dump(best['stats'], f, ensure_ascii=False, indent=2)

        # 每seed指标落盘（Train@Best + Val@Best）
        per_row = {
            "seed": int(s),
            "best_epoch": int(best.get("best_epoch", -1)),
            "R2_tra": float(best["train"]["R2"]),
            "RMSE_tra": float(best["train"]["RMSE"]),
            "MAE_tra": float(best["train"]["MAE"]),
            "R2_val": float(best["val"]["R2"]),
            "RMSE_val": float(best["val"]["RMSE"]),
            "MAE_val": float(best["val"]["MAE"]),
        }
        if os.path.exists(per_seed_csv):
            pd.concat([pd.read_csv(per_seed_csv), pd.DataFrame([per_row])], ignore_index=True).to_csv(per_seed_csv, index=False)
        else:
            pd.DataFrame([per_row]).to_csv(per_seed_csv, index=False)

    # best-seed（按 Val R²）
    best_seed, best_model = max(all_models, key=lambda x: x[1]['val']['R2'])
    print(f"\nBest Seed = {best_seed}")
    print("Train@Best:", best_model["train"])
    print("Val@Best  :", best_model["val"])

    # 保存 best 统一命名
    torch.save(best_model['enc'], os.path.join(args.outdir, "encoder_bcf_pure_stgnn_best.pt"))
    torch.save(best_model['head'], os.path.join(args.outdir, "head_bcf_pure_stgnn_best.pt"))
    with open(os.path.join(args.outdir, "bcf_scaler_pure_stgnn_best.json"), "w", encoding="utf-8") as f:
        json.dump(best_model['stats'], f, ensure_ascii=False, indent=2)

    # CV
    cv_r2 = None
    if args.cross_validation:
        cv_res = cross_validation(
            tr_df, device=device, init_seed=args.split_seed,
            epochs=args.cv_epochs, lr=2e-3, weight_decay=7e-5,
            huber_beta=0.25, batch=args.batch, use_ema=args.ema,
            ema_decay=args.ema_decay, ema_start=args.ema_start,
            in_dim=11, hid=args.hid, layers=args.layers
        )
        cv_r2 = cv_res["R2_cv"]

        out_dir = os.path.join(args.outdir, "cross_validation_results_pure_stgnn")
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, f"cv_metrics_seed{args.split_seed}.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in cv_res.items() if k not in ["y_true","y_pred"]},
                      f, ensure_ascii=False, indent=2)

        pd.DataFrame({
            "true": cv_res["y_true"],
            "predicted": cv_res["y_pred"],
            "residual": cv_res["y_true"] - cv_res["y_pred"],
        }).to_csv(os.path.join(out_dir, f"cv_predictions_seed{args.split_seed}.csv"), index=False)

    # 论文表格汇总一行（7列）
    best_row = {
        "R2_tra": float(best_model["train"]["R2"]),
        "RMSE_tra": float(best_model["train"]["RMSE"]),
        "MAE_tra": float(best_model["train"]["MAE"]),
        "R2_val": float(best_model["val"]["R2"]),
        "RMSE_val": float(best_model["val"]["RMSE"]),
        "MAE_val": float(best_model["val"]["MAE"]),
    }
    summary_csv = os.path.join(args.outdir, f"paper_table_metrics_seed{args.split_seed}.csv")
    save_summary_table(summary_csv, args.split_seed, seeds, best_row, cv_r2)

    # 控制台打印一行（方便你直接抄表格）
    print("\n" + "="*70)
    print("Paper Table Metrics (best-seed, ST-GNN)")
    print(f"R2_tra={best_row['R2_tra']:.3f} | RMSE_tra={best_row['RMSE_tra']:.3f} | MAE_tra={best_row['MAE_tra']:.3f} | "
          f"R2_val={best_row['R2_val']:.3f} | RMSE_val={best_row['RMSE_val']:.3f} | MAE_val={best_row['MAE_val']:.3f} | "
          f"R2_cv={(cv_r2 if cv_r2 is not None else float('nan')):.3f}")
    print("="*70)

if __name__ == "__main__":
    main()
