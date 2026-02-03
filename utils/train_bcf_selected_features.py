# -*- coding: utf-8 -*-
"""
使用筛选后特征的BCF迁移学习训练脚本
- 读取 selected_features.json 中的特征列表
- 只使用筛选后的分子描述符进行训练
"""

import os, re, json, math, argparse, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import dgl
from dgl.nn import NNConv

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen
from rdkit.Chem import rdMolDescriptors as rdm
from rdkit.ML.Descriptors import MoleculeDescriptors

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RDLogger.DisableLog('rdApp.*')

# ========================
# 工具函数
# ========================
def set_seed(sd=82):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(sd)

def _norm(s): return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def pick_col(df, candidates):
    m = {_norm(c): c for c in df.columns}
    if isinstance(candidates, str): candidates = [candidates]
    flat = []
    for c in candidates:
        if isinstance(c,(list,tuple)): flat += list(c)
        else: flat.append(c)
    for c in flat:
        k = _norm(c)
        if k in m: return m[k]
    for k,raw in m.items():
        if any(_norm(c) in k for c in flat): return raw
    return None

# ========================
# 图构建
# ========================
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
    g.ndata['h']=torch.stack(atom_x, dim=0); g.edata['e']=torch.stack(efeat, dim=0)
    return g

# ========================
# 描述符计算（使用筛选后的特征）
# ========================
def compute_selected_descriptors(mol, selected_features):
    """
    只计算筛选后的描述符
    selected_features: list of feature names
    """
    # 所有RDKit描述符
    all_desc_names = [name for name, _ in Descriptors._descList]
    
    # PFAS特异性描述符
    pfas_names = ['F_count', 'CF2_count', 'CF3_count', 'has_SO3', 'has_COO', 'F_heavy_ratio']
    
    # 需要计算的RDKit描述符
    rdkit_needed = [f for f in selected_features if f not in pfas_names]
    
    result = {}
    
    # 计算RDKit描述符
    if len(rdkit_needed) > 0:
        try:
            # 只计算需要的描述符
            calculator = MoleculeDescriptors.MolecularDescriptorCalculator(all_desc_names)
            all_values = calculator.CalcDescriptors(mol)
            desc_dict = dict(zip(all_desc_names, all_values))
            
            for feat in rdkit_needed:
                if feat in desc_dict:
                    result[feat] = float(desc_dict[feat])
                else:
                    result[feat] = np.nan
        except:
            for feat in rdkit_needed:
                result[feat] = np.nan
    
    # 计算PFAS特异性描述符
    if any(f in selected_features for f in pfas_names):
        try:
            F_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum()==9)
            heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum()>1)
            F_heavy_ratio = F_count / max(1, heavy)
            
            CF2_count = CF3_count = 0
            for a in mol.GetAtoms():
                if a.GetAtomicNum() != 6: continue
                F_nb = sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum()==9)
                if F_nb == 2: CF2_count += 1
                elif F_nb == 3: CF3_count += 1
            
            patt_SO3 = Chem.MolFromSmarts("S(=O)(=O)[O-]")
            patt_SO3a = Chem.MolFromSmarts("S(=O)(=O)O")
            patt_COO = Chem.MolFromSmarts("C(=O)[O-]")
            patt_COOa = Chem.MolFromSmarts("C(=O)O")
            
            has_SO3 = 1.0 if (mol.HasSubstructMatch(patt_SO3) or mol.HasSubstructMatch(patt_SO3a)) else 0.0
            has_COO = 1.0 if (mol.HasSubstructMatch(patt_COO) or mol.HasSubstructMatch(patt_COOa)) else 0.0
            
            pfas_dict = {
                'F_count': float(F_count),
                'CF2_count': float(CF2_count),
                'CF3_count': float(CF3_count),
                'has_SO3': has_SO3,
                'has_COO': has_COO,
                'F_heavy_ratio': float(F_heavy_ratio)
            }
            
            for feat in pfas_names:
                if feat in selected_features:
                    result[feat] = pfas_dict[feat]
        except:
            for feat in pfas_names:
                if feat in selected_features:
                    result[feat] = np.nan
    
    # 按照 selected_features 的顺序返回
    return np.array([result.get(f, np.nan) for f in selected_features], dtype=np.float32)

# ========================
# Dataset
# ========================
class DS(Dataset):
    def __init__(self, df, selected_features, y_mean=None, y_std=None, d_mean=None, d_std=None, fit=False, use_precomputed=False):
        """
        Args:
            df: DataFrame，必须包含SMILES列和logBCF列
            selected_features: 筛选后的特征列表
            use_precomputed: 如果为True，从df中直接读取预计算的描述符列；否则从SMILES重新计算
        """
        smi_col = pick_col(df, ["SMILES","smiles"])
        y_col = pick_col(df, [["logBCF","log_bcf","logbcf"], ["log_BCF"], ["BCF_log"]])
        if smi_col is None or y_col is None:
            raise ValueError("未找到 SMILES 或 logBCF 列")
        
        self.smiles = df[smi_col].tolist()
        self.y_raw = df[y_col].astype(float).values
        self.selected_features = selected_features
        self.use_precomputed = use_precomputed
        
        G, keep, D = [], [], []
        
        if use_precomputed:
            # 从CSV直接读取预计算的描述符
            for i, s in enumerate(self.smiles):
                m = Chem.MolFromSmiles(str(s).strip())
                if m is None: continue
                g = mol_to_graph(m)
                if g.num_nodes() == 0: continue
                
                # 从DataFrame中提取对应的描述符值
                desc_values = []
                for feat in selected_features:
                    if feat in df.columns:
                        desc_values.append(df[feat].iloc[i])
                    else:
                        desc_values.append(np.nan)
                desc = np.array(desc_values, dtype=np.float32)
                G.append(g); keep.append(i); D.append(desc)
        else:
            # 从SMILES重新计算描述符
            for i, s in enumerate(self.smiles):
                m = Chem.MolFromSmiles(str(s).strip())
                if m is None: continue
                g = mol_to_graph(m)
                if g.num_nodes() == 0: continue
                
                desc = compute_selected_descriptors(m, selected_features)
                G.append(g); keep.append(i); D.append(desc)
        
        self.G = G
        self.smiles = [self.smiles[i] for i in keep]
        self.y_raw = self.y_raw[keep]
        D = np.asarray(D, dtype=np.float32)
        
        # 缺失值填充
        med = np.nanmedian(D, axis=0)
        idx = np.where(np.isnan(D))
        if idx[0].size > 0:
            D[idx] = np.take(med, idx[1])
        
        if fit:
            self.y_mean = float(self.y_raw.mean())
            self.y_std = float(self.y_raw.std() + 1e-8)
            self.d_mean = D.mean(axis=0)
            self.d_std = D.std(axis=0) + 1e-8
        else:
            self.y_mean = float(y_mean)
            self.y_std = float(max(y_std, 1e-8))
            self.d_mean = d_mean
            self.d_std = d_std
        
        self.y = torch.tensor((self.y_raw - self.y_mean) / self.y_std, dtype=torch.float32)
        self.D = torch.tensor((D - self.d_mean) / self.d_std, dtype=torch.float32)
    
    def __len__(self): return len(self.G)
    def __getitem__(self, i): return self.G[i], self.y[i], self.smiles[i], self.D[i]

def collate(b):
    gs, ys, ss, Ds = zip(*b)
    return dgl.batch(gs), torch.stack(ys), list(ss), torch.stack(Ds)

# ========================
# 模型
# ========================
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
    def __init__(self, hid=128, d_dim=20):
        super().__init__()
        self.bn_d = nn.BatchNorm1d(d_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2*hid + d_dim, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 1)
        )
    
    def forward(self, hg, d):
        return self.mlp(torch.cat([hg, self.bn_d(d)], dim=1)).squeeze(1)

# ========================
# EMA
# ========================
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

# ========================
# 评估
# ========================
def evaluate(enc, head, loader, inv, device):
    enc.eval(); head.eval()
    Ys, Ps = [], []
    with torch.no_grad():
        for g, y, _, D in loader:
            g = g.to(device); D = D.to(device)
            Ps.append(head(enc(g), D).cpu())
            Ys.append(y.cpu())
    Ys = torch.cat(Ys); Ps = torch.cat(Ps)
    y_true, y_pred = inv(Ys), inv(Ps)
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    mse = torch.mean((y_true - y_pred)**2).item()
    rmse = math.sqrt(mse)
    var = torch.var(y_true, unbiased=False).item()
    r2 = 1.0 - (mse / (var + 1e-12))
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def compute_qsar_metrics(y_true, y_pred, y_train_mean=None):
    """
    计算完整的QSAR验证指标
    
    Args:
        y_true: 真实值 (numpy array or tensor)
        y_pred: 预测值 (numpy array or tensor)
        y_train_mean: 训练集均值（用于计算Q²）
    
    Returns:
        dict: 包含所有QSAR指标
    """
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    n = len(y_true)
    
    # 基础指标
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    
    # R²
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-12))
    
    # Q² (需要训练集均值)
    if y_train_mean is not None:
        press = np.sum((y_true - y_pred)**2)
        tss = np.sum((y_true - y_train_mean)**2)
        q2 = 1 - (press / (tss + 1e-12))
    else:
        q2 = None
    
    # CCC (Concordance Correlation Coefficient)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covar = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * covar) / (var_true + var_pred + (mean_true - mean_pred)**2 + 1e-12)
    
    # Pearson相关系数
    corr = np.corrcoef(y_true, y_pred)[0, 1] if n > 1 else 0.0
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'CCC': ccc,
        'Correlation': corr,
        'N': n
    }
    
    if q2 is not None:
        metrics['Q2'] = q2
    
    return metrics

def leave_one_out_cv(tr_df, selected_features, device='cpu', 
                     epochs=100, lr=2e-3, freeze_epochs=20,
                     weight_decay=7e-5, huber_beta=0.25, batch=128,
                     init_seed=2024, use_ema=False, ema_decay=0.995, 
                     ema_start=10, use_precomputed=False):
    """
    留一交叉验证 (Leave-One-Out Cross-Validation)
    

    """
    from sklearn.model_selection import KFold
    
    n_samples = len(tr_df)
    print(f"\n{'='*70}")
    print(f"交叉验证 (Leave-One-Out CV Approximation)")
    print(f"{'='*70}")
    print(f"样本数: {n_samples}")
    
    # 根据样本量选择fold数
    if n_samples <= 200:
        n_folds = n_samples  # 真正的LOO
        print(f"使用留一交叉验证 (LOO)")
    elif n_samples <= 500:
        n_folds = 10
        print(f"使用 10-Fold 交叉验证")
    else:
        n_folds = 5
        print(f"使用 5-Fold 交叉验证")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=init_seed)
    
    all_true = []
    all_pred = []
    
    # 计算整体训练集统计
    fit_ds = DS(tr_df, selected_features, fit=True, use_precomputed=use_precomputed)
    y_mean_global = fit_ds.y_mean
    y_std_global = fit_ds.y_std
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(tr_df), 1):
        if fold % max(1, n_folds // 10) == 0 or fold == n_folds:
            print(f"  Fold {fold}/{n_folds}...", end='\r')
        
        # 划分数据
        fold_train = tr_df.iloc[train_idx].copy()
        fold_val = tr_df.iloc[val_idx].copy()
        
        # 准备数据集
        fit = DS(fold_train, selected_features, fit=True, use_precomputed=use_precomputed)
        y_mean, y_std, d_mean, d_std = fit.y_mean, fit.y_std, fit.d_mean, fit.d_std
        
        train_ds = DS(fold_train, selected_features, y_mean, y_std, d_mean, d_std, 
                     use_precomputed=use_precomputed)
        val_ds = DS(fold_val, selected_features, y_mean, y_std, d_mean, d_std,
                   use_precomputed=use_precomputed)
        
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, 
                                 collate_fn=collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch*2, shuffle=False,
                               collate_fn=collate, num_workers=0)
        
        # 加载预训练编码器
        meta = json.load(open("encoder_meta_nn.json", "r"))
        enc = Encoder(in_dim=meta.get('in_dim',11), hid=meta.get('hid',128), 
                     layers=meta.get('layers',4)).to(device)
        state = torch.load("encoder_kow_nnconv.pt", map_location="cpu")
        enc.load_state_dict(state, strict=True)
        
        d_dim = len(selected_features)
        head = Head(hid=meta.get('hid',128), d_dim=d_dim).to(device)
        
        inv = lambda t: t * y_std + y_mean
        
        ema_enc = EMA(enc, decay=ema_decay) if use_ema else None
        ema_head = EMA(head, decay=ema_decay) if use_ema else None
        
        best_val_loss = 1e9
        patience_counter = 0
        max_patience = 25  
        
        # 训练
        for ep in range(1, epochs+1):
            if ep <= freeze_epochs:
                enc.eval()
                for p in enc.parameters(): p.requires_grad = False
                params = head.parameters()
                cur_lr = lr
            else:
                enc.train()
                for p in enc.parameters(): p.requires_grad = True
                params = list(enc.parameters()) + list(head.parameters())
                cur_lr = lr * 0.7
            
            opt = torch.optim.AdamW(params, lr=cur_lr, weight_decay=weight_decay)
            
            # 训练一个epoch
            for g, y, _, D in train_loader:
                g = g.to(device); y = y.to(device); D = D.to(device)
                pred = head(enc(g), D)
                loss = F.smooth_l1_loss(pred, y, beta=huber_beta)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                
                if use_ema:
                    if isinstance(params, list):
                        ema_enc.update(enc)
                    ema_head.update(head)
            
            # 验证
            if use_ema and ep >= ema_start:
                ema_enc.apply_to(enc)
                ema_head.apply_to(head)
            
            val_metrics = evaluate(enc, head, val_loader, inv, device)
            val_loss = val_metrics['RMSE']
            
            if use_ema and ep >= ema_start:
                ema_enc.restore(enc)
                ema_head.restore(head)
            
            # Early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型状态
                if use_ema and ep >= ema_start:
                    ema_enc.apply_to(enc)
                    ema_head.apply_to(head)
                best_enc_state = {k: v.cpu().clone() for k, v in enc.state_dict().items()}
                best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                if use_ema and ep >= ema_start:
                    ema_enc.restore(enc)
                    ema_head.restore(head)
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break
        
        # 加载最佳模型
        enc.load_state_dict(best_enc_state)
        head.load_state_dict(best_head_state)
        enc.to(device)
        head.to(device)
        
        # 预测验证集
        enc.eval()
        head.eval()
        with torch.no_grad():
            Ys, Ps = [], []
            for g, y, _, D in val_loader:
                g = g.to(device); D = D.to(device)
                Ps.append(head(enc(g), D).cpu())
                Ys.append(y.cpu())
            Y = torch.cat(Ys)
            P = torch.cat(Ps)
            y_true = inv(Y).numpy()
            y_pred = inv(P).numpy()
        
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())
    
    print(f"\n交叉验证完成！")
    
    # 计算整体CV指标
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    
    # 使用原始训练集均值计算Q²
    y_train_mean = tr_df['logBCF'].mean()
    
    cv_metrics = compute_qsar_metrics(all_true, all_pred, y_train_mean)
    
    return cv_metrics, all_true, all_pred

# ========================
# 特征重要性分析
# ========================
def compute_permutation_importance(enc, head, loader, inv, device, selected_features, n_repeats=5):
    """
    计算排列重要性：随机打乱每个特征的值，看性能下降多少
    """
    print("\n" + "="*70)
    print("Computing Permutation Importance...")
    print("="*70)
    
    enc.eval(); head.eval()
    
    # 1. 计算基准性能
    baseline_metrics = evaluate(enc, head, loader, inv, device)
    baseline_r2 = baseline_metrics['R2']
    baseline_mae = baseline_metrics['MAE']
    
    print(f"Baseline - R²: {baseline_r2:.4f}, MAE: {baseline_mae:.4f}")
    
    # 2. 收集所有数据
    all_graphs, all_y, all_D = [], [], []
    with torch.no_grad():
        for g, y, _, D in loader:
            all_graphs.append(g)
            all_y.append(y)
            all_D.append(D)
    
    batched_graph = dgl.batch(all_graphs)
    all_y = torch.cat(all_y)
    all_D = torch.cat(all_D)
    
    # 3. 对每个特征进行排列测试
    importance_r2 = []
    importance_mae = []
    
    for feat_idx, feat_name in enumerate(selected_features):
        r2_drops = []
        mae_increases = []
        
        for repeat in range(n_repeats):
            # 复制描述符并打乱当前特征
            D_permuted = all_D.clone()
            perm_indices = torch.randperm(D_permuted.shape[0])
            D_permuted[:, feat_idx] = D_permuted[perm_indices, feat_idx]
            
            # 预测
            with torch.no_grad():
                batched_graph_device = batched_graph.to(device)
                D_permuted_device = D_permuted.to(device)
                pred = head(enc(batched_graph_device), D_permuted_device).cpu()
            
            y_true = inv(all_y)
            y_pred = inv(pred)
            
            mae = torch.mean(torch.abs(y_true - y_pred)).item()
            mse = torch.mean((y_true - y_pred)**2).item()
            var = torch.var(y_true, unbiased=False).item()
            r2 = 1.0 - (mse / (var + 1e-12))
            
            r2_drops.append(baseline_r2 - r2)
            mae_increases.append(mae - baseline_mae)
        
        importance_r2.append(np.mean(r2_drops))
        importance_mae.append(np.mean(mae_increases))
        
        print(f"  {feat_name:30s} - R² drop: {np.mean(r2_drops):7.4f} ± {np.std(r2_drops):.4f}, "
              f"MAE increase: {np.mean(mae_increases):7.4f} ± {np.std(mae_increases):.4f}")
    
    return {
        'features': selected_features,
        'importance_r2': importance_r2,
        'importance_mae': importance_mae,
        'baseline_r2': baseline_r2,
        'baseline_mae': baseline_mae
    }

def analyze_feature_weights(head, selected_features, hid=128):
    """
    分析head网络中第一层的权重，查看每个特征的影响
    
    Head的forward: mlp(torch.cat([hg, bn_d(d)], dim=1))
    其中 hg 是图特征 (维度: 2*hid), d 是描述符特征 (维度: d_dim)
    """
    print("\n" + "="*70)
    print("Analyzing Feature Weights in Neural Network...")
    print("="*70)
    
    # 获取MLP第一层权重
    first_layer = head.mlp[0]  # nn.Linear(2*hid + d_dim, 512)
    weights = first_layer.weight.detach().cpu().numpy()  # shape: [out_features, in_features]
    
    out_features, in_features = weights.shape
    n_desc = len(selected_features)
    graph_feature_dim = 2 * hid
    
    print(f"\nNetwork structure:")
    print(f"  First layer weight shape: {weights.shape}")
    print(f"  Input features: {in_features} = {graph_feature_dim} (graph) + {n_desc} (descriptors)")
    print(f"  Output features: {out_features}")
    
    # 验证维度匹配
    if in_features != graph_feature_dim + n_desc:
        print(f"  WARNING: Expected input dimension {graph_feature_dim + n_desc}, got {in_features}")
    
    # 前面是图特征(2*hid维)，后面是描述符特征(d_dim维)
    # 这是由 Head.forward 中的 torch.cat([hg, self.bn_d(d)], dim=1) 决定的
    graph_weights = weights[:, :graph_feature_dim]
    desc_weights = weights[:, graph_feature_dim:]  # 最后 n_desc 列
    
    if desc_weights.shape[1] != n_desc:
        print(f"  WARNING: Descriptor weight dimension mismatch: {desc_weights.shape[1]} vs {n_desc}")
    
    # 计算每个特征的权重重要性（L2范数）
    weight_importance = np.sqrt(np.sum(desc_weights**2, axis=0))
    
    # 也计算图特征的整体权重（作为对比）
    graph_weight_importance = np.sqrt(np.sum(graph_weights**2, axis=0))
    graph_weight_mean = np.mean(graph_weight_importance)
    
    print(f"\nGraph features weight L2 norm (mean): {graph_weight_mean:.4f}")
    print(f"\nDescriptor Feature Weight Importance (L2 norm):")
    for feat, imp in zip(selected_features, weight_importance):
        print(f"  {feat:30s} : {imp:8.4f}")
    
    return {
        'features': selected_features,
        'weight_importance': weight_importance.tolist(),
        'graph_weight_mean': float(graph_weight_mean),
        'network_structure': {
            'in_features': int(in_features),
            'out_features': int(out_features),
            'graph_dim': int(graph_feature_dim),
            'descriptor_dim': int(n_desc)
        }
    }

def plot_feature_importance(importance_dict, weight_dict, output_dir='feature_importance_results'):
    """
    可视化特征重要性
    """
    print("\n" + "="*70)
    print("Plotting Feature Importance...")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    features = importance_dict['features']
    
    # 图1: 排列重要性 (R² drop)
    plt.figure(figsize=(12, max(6, len(features) * 0.3)))
    sorted_idx = np.argsort(importance_dict['importance_r2'])[::-1]
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = [importance_dict['importance_r2'][i] for i in sorted_idx]
    
    plt.barh(range(len(features)), sorted_importance[::-1])
    plt.yticks(range(len(features)), sorted_features[::-1])
    plt.xlabel('R² Drop (Higher = More Important)')
    plt.title('Permutation Importance - R² Drop')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, 'permutation_importance_r2.png')
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot1_path}")
    
    # 图2: 排列重要性 (MAE increase)
    plt.figure(figsize=(12, max(6, len(features) * 0.3)))
    sorted_idx = np.argsort(importance_dict['importance_mae'])[::-1]
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = [importance_dict['importance_mae'][i] for i in sorted_idx]
    
    plt.barh(range(len(features)), sorted_importance[::-1])
    plt.yticks(range(len(features)), sorted_features[::-1])
    plt.xlabel('MAE Increase (Higher = More Important)')
    plt.title('Permutation Importance - MAE Increase')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, 'permutation_importance_mae.png')
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot2_path}")
    
    # 图3: 权重重要性
    if weight_dict is not None:
        plt.figure(figsize=(12, max(6, len(features) * 0.3)))
        sorted_idx = np.argsort(weight_dict['weight_importance'])[::-1]
        sorted_features = [features[i] for i in sorted_idx]
        sorted_weights = [weight_dict['weight_importance'][i] for i in sorted_idx]
        
        plt.barh(range(len(features)), sorted_weights[::-1])
        plt.yticks(range(len(features)), sorted_features[::-1])
        plt.xlabel('Weight Magnitude (L2 Norm)')
        plt.title('Neural Network Weight Importance')
        plt.tight_layout()
        plot3_path = os.path.join(output_dir, 'weight_importance.png')
        plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot3_path}")
    
    # 图4: 对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(features) * 0.3)))
    
    # R² drop
    sorted_idx = np.argsort(importance_dict['importance_r2'])[::-1]
    axes[0].barh(range(len(features)), [importance_dict['importance_r2'][i] for i in sorted_idx][::-1])
    axes[0].set_yticks(range(len(features)))
    axes[0].set_yticklabels([features[i] for i in sorted_idx][::-1])
    axes[0].set_xlabel('R² Drop')
    axes[0].set_title('Permutation (R²)')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.3)
    
    # MAE increase
    sorted_idx = np.argsort(importance_dict['importance_mae'])[::-1]
    axes[1].barh(range(len(features)), [importance_dict['importance_mae'][i] for i in sorted_idx][::-1])
    axes[1].set_yticks(range(len(features)))
    axes[1].set_yticklabels([features[i] for i in sorted_idx][::-1])
    axes[1].set_xlabel('MAE Increase')
    axes[1].set_title('Permutation (MAE)')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.3)
    
    # Weight importance
    if weight_dict is not None:
        sorted_idx = np.argsort(weight_dict['weight_importance'])[::-1]
        axes[2].barh(range(len(features)), [weight_dict['weight_importance'][i] for i in sorted_idx][::-1])
        axes[2].set_yticks(range(len(features)))
        axes[2].set_yticklabels([features[i] for i in sorted_idx][::-1])
        axes[2].set_xlabel('Weight L2 Norm')
        axes[2].set_title('Network Weights')
    
    plt.tight_layout()
    plot4_path = os.path.join(output_dir, 'feature_importance_comparison.png')
    plt.savefig(plot4_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot4_path}")
    
    print(f"\nAll plots saved to: {output_dir}/")
    
    return output_dir

# ========================
# 随机分层划分
# ========================
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
        tr_idx.extend(idx[:n_tr].tolist())
        va_idx.extend(idx[n_tr:].tolist())
    df_tmp.drop(columns=['__bin__'], inplace=True)
    return df_tmp.loc[tr_idx], df_tmp.loc[va_idx]



# === Save current split (train / val) for ADSAL ===
def _std_pick_col(df, candidates):
    # 简单就地匹配，无需依赖项目里的 pick_col
    cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # 模糊包含匹配
    for c in df.columns:
        lc = str(c).strip().lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c
    return None

def save_split_csv(tr_df, va_df, split_seed, outdir="dataset_splits"):
    """
    仅保存本次 80/20 划分得到的训练/验证集（含 SMILES 与 logBCF），
    供 ADSAL/查询样本适用域评估使用；不影响训练流程。
    """
    import os, json
    os.makedirs(outdir, exist_ok=True)

    smi_col = _std_pick_col(tr_df, ["SMILES", "smiles"])
    y_col   = _std_pick_col(tr_df, ["logBCF", "log_bcf", "logbcf", "bcf_log", "log_bcf"])
    if smi_col is None or y_col is None:
        # 如果你的脚本前面已经统一改名为 SMILES / logBCF，这里一定能命中；否则直接跳过保存
        print("[Warn] save_split_csv: 未找到 SMILES/logBCF 列，跳过保存。")
        return

    keep_cols = [smi_col, y_col]
    train_path = os.path.join(outdir, f"train_bcf_seed{split_seed}.csv")
    valid_path = os.path.join(outdir, f"valid_bcf_seed{split_seed}.csv")

    tr_df[keep_cols].to_csv(train_path, index=False)
    va_df[keep_cols].to_csv(valid_path, index=False)

    summary = {
        "split_seed": int(split_seed),
        "train_size": int(len(tr_df)),
        "valid_size": int(len(va_df)),
        "columns": keep_cols,
        "note": "These are the exact 80/20 split subsets used by TL+Desc-GNN for this run."
    }
    with open(os.path.join(outdir, f"split_summary_seed{split_seed}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[Split Saved] Train -> {train_path}")
    print(f"[Split Saved] Valid -> {valid_path}")





# ========================
# 训练
# ========================
def train_once(tr_df, va_df, selected_features, device='cpu',
               epochs=240, lr=1e-3, freeze_epochs=10,
               weight_decay=1e-4, huber_beta=0.25, batch=128,
               init_seed=2024, use_ema=False, ema_decay=0.995, ema_start=10, use_precomputed=False):
    
    fit = DS(tr_df, selected_features, fit=True, use_precomputed=use_precomputed)
    y_mean, y_std, d_mean, d_std = fit.y_mean, fit.y_std, fit.d_mean, fit.d_std
    tr_ds = DS(tr_df, selected_features, y_mean, y_std, d_mean, d_std, use_precomputed=use_precomputed)
    va_ds = DS(va_df, selected_features, y_mean, y_std, d_mean, d_std, use_precomputed=use_precomputed)
    
    tr_loader = DataLoader(tr_ds, batch_size=batch, shuffle=True, collate_fn=collate, 
                          num_workers=0, pin_memory=(device=='cuda'))
    va_loader = DataLoader(va_ds, batch_size=batch*2, shuffle=False, collate_fn=collate,
                          num_workers=0, pin_memory=(device=='cuda'))
    
    # 加载预训练编码器
    if not (os.path.exists("encoder_kow_nnconv.pt") and os.path.exists("encoder_meta_nn.json")):
        raise SystemExit("缺少预训练文件")
    
    meta = json.load(open("encoder_meta_nn.json", "r"))
    enc = Encoder(in_dim=meta.get('in_dim',11), hid=meta.get('hid',128), layers=meta.get('layers',4)).to(device)
    state = torch.load("encoder_kow_nnconv.pt", map_location="cpu")
    enc.load_state_dict(state, strict=True)
    
    d_dim = len(selected_features)
    head = Head(hid=meta.get('hid',128), d_dim=d_dim).to(device)
    
    set_seed(init_seed)
    inv = lambda t: t * y_std + y_mean
    best, best_r2, bad, patience = None, -1e9, 0, 60
    
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision('high')
    
    ema_enc = EMA(enc, decay=ema_decay) if use_ema else None
    ema_head = EMA(head, decay=ema_decay) if use_ema else None
    
    for ep in range(1, epochs+1):
        # 冻结/解冻阶段
        if ep <= freeze_epochs:
            enc.eval()
            for p in enc.parameters(): p.requires_grad = False
            params = head.parameters()
            cur_lr = lr
        else:
            enc.train()
            for p in enc.parameters(): p.requires_grad = True
            params = list(enc.parameters()) + list(head.parameters())
            cur_lr = lr * 0.7
        
        opt = torch.optim.AdamW(params, lr=cur_lr, weight_decay=weight_decay)
        
        # 训练一个 epoch
        for g, y, _, D in tr_loader:
            g = g.to(device, non_blocking=True); y = y.to(device); D = D.to(device)
            pred = head(enc(g), D)
            loss = F.smooth_l1_loss(pred, y, beta=huber_beta)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            # EMA 更新
            if use_ema:
                if isinstance(params, list):  # 解冻阶段才更新 encoder 的 EMA
                    ema_enc.update(enc)
                ema_head.update(head)
        
        # 验证（优先用 EMA 权重）
        using_ema = False
        if use_ema and ep >= ema_start:
            ema_enc.apply_to(enc); ema_head.apply_to(head)
            using_ema = True
        val = evaluate(enc, head, va_loader, inv, device)
        if using_ema:
            ema_enc.restore(enc); ema_head.restore(head)
        
        print(f"[Epoch {ep:03d}] valMAE={val['MAE']:.3f} valRMSE={val['RMSE']:.3f} valR2={val['R2']:.3f}  frozen={ep<=freeze_epochs}")
        
        # 按 R² 选择最佳；若用 EMA 评估，则保存 EMA 权重
        if val['R2'] > best_r2 + 1e-3:
            best_r2 = val['R2']; bad = 0
            if use_ema and ep >= ema_start:
                ema_enc.apply_to(enc); ema_head.apply_to(head)
                best = {
                    'enc': {k:v.detach().cpu() for k,v in enc.state_dict().items()},
                    'head': {k:v.detach().cpu() for k,v in head.state_dict().items()},
                    'val': val,
                    'stats': {'y_mean':y_mean, 'y_std':y_std, 'd_mean':d_mean.tolist(), 'd_std':d_std.tolist()}
                }
                ema_enc.restore(enc); ema_head.restore(head)
            else:
                best = {
                    'enc': {k:v.detach().cpu() for k,v in enc.state_dict().items()},
                    'head': {k:v.detach().cpu() for k,v in head.state_dict().items()},
                    'val': val,
                    'stats': {'y_mean':y_mean, 'y_std':y_std, 'd_mean':d_mean.tolist(), 'd_std':d_std.tolist()}
                }
        else:
            bad += 1
            if bad >= patience:
                print("Early stop."); break
    
    return best

# ========================
# 主函数
# ========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="数据文件")
    ap.add_argument("--features", required=True, help="selected_features.json路径")
    ap.add_argument("--use-csv-desc", action="store_true", help="从CSV直接读取预计算的描述符（而不是从SMILES重新计算）")
    ap.add_argument("--split-seed", type=int, default=82)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--seeds", type=str, default="2024")
    ap.add_argument("--ensemble", action="store_true", help="训练完在同一验证集上做均值集成")
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--ema-start", type=int, default=10)
    ap.add_argument("--analyze-importance", action="store_true", help="训练后分析特征重要性")
    ap.add_argument("--importance-repeats", type=int, default=5, help="排列重要性的重复次数")
    ap.add_argument("--importance-output", type=str, default="feature_importance_results", help="特征重要性结果输出目录")
    ap.add_argument("--cross-validation", action="store_true", help="进行交叉验证计算Q²_LOO等指标")
    ap.add_argument("--cv-epochs", type=int, default=100, help="交叉验证时每个fold的训练epoch数")
    args = ap.parse_args()
    
    # 加载筛选的特征
    with open(args.features, 'r', encoding='utf-8') as f:
        feat_data = json.load(f)
    if isinstance(feat_data, dict) and 'features' in feat_data:
        selected_features = feat_data['features']
    elif isinstance(feat_data, list):
        selected_features = feat_data
    else:
        raise SystemExit("selected_features.json格式错误")
    
    print(f"加载筛选特征: {len(selected_features)}个")
    print(f"  {', '.join(selected_features[:min(5, len(selected_features))])}...")
    
    # 加载数据
    if not os.path.exists(args.csv):
        raise SystemExit(f"未找到数据文件：{args.csv}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv(args.csv)
    
    smi_col = pick_col(df, ["SMILES","smiles"])
    y_log = pick_col(df, [["logBCF","log_bcf","logbcf","log_BCF","BCF_log"]])
    if smi_col is None or y_log is None:
        raise SystemExit("CSV需包含SMILES和logBCF列")
    
    # 检查是否使用CSV中的预计算描述符
    if args.use_csv_desc:
        # 检查所有特征列是否在CSV中
        missing_feats = [f for f in selected_features if f not in df.columns]
        if missing_feats:
            print(f"警告: 以下特征在CSV中缺失: {missing_feats}")
            print("将从SMILES重新计算这些特征")
        
        # 只保留SMILES, logBCF和选中的特征列
        cols_to_keep = [smi_col, y_log] + [f for f in selected_features if f in df.columns]
        df = df[cols_to_keep]
        print(f"从CSV读取预计算描述符: {len([f for f in selected_features if f in df.columns])}个")
    else:
        # 只保留SMILES和logBCF列
        df = df[[smi_col, y_log]]
    
    df = df.rename(columns={smi_col:"SMILES", y_log:"logBCF"})
    df = df.dropna(subset=["SMILES", "logBCF"]).drop_duplicates(subset=["SMILES"])
    
    tr_df, va_df = random_stratified_split(df, "logBCF", frac_train=0.8, bins=args.bins, seed=args.split_seed)
    print(f"Random stratified split 80/20 (seed={args.split_seed}) -> Train {len(tr_df)} | Val {len(va_df)}")
    
    # 额外保存当次划分（仅 SMILES + logBCF），用于 ADSAL；不影响训练流程
    save_split_csv(tr_df, va_df, args.split_seed, outdir="dataset_splits")

    
    seeds = [int(s) for s in re.split(r'[,\s]+', args.seeds.strip()) if s]
    
    names = []; preds = []; truths = None
    
    for s in seeds:
        print(f"\n==== Train with seed {s} ====")
        best = train_once(tr_df, va_df, selected_features, device=device,
                         epochs=args.epochs, batch=args.batch, init_seed=s,
                         use_ema=args.ema, ema_decay=args.ema_decay, ema_start=args.ema_start,
                         use_precomputed=args.use_csv_desc)
        
        if best is None:
            raise SystemExit("训练失败")
        
        enc_name = f"encoder_bcf_selected_s{s}.pt"
        head_name = f"head_bcf_selected_s{s}.pt"
        sca_name = f"bcf_scaler_selected_s{s}.json"
        
        torch.save(best['enc'], enc_name)
        torch.save(best['head'], head_name)
        with open(sca_name, "w", encoding="utf-8") as f:
            json.dump(best['stats'], f, indent=2)
        
        print(f"Best Val (seed={s}): {best['val']}")
        print(f"Saved: {enc_name}, {head_name}, {sca_name}")
        names.append(s)
        
        # 为集成记录验证集预测
        if args.ensemble:
            y_mean = best['stats']['y_mean']; y_std = best['stats']['y_std']
            d_mean = np.array(best['stats']['d_mean']); d_std = np.array(best['stats']['d_std'])
            va_ds = DS(va_df, selected_features, y_mean, y_std, d_mean, d_std, use_precomputed=args.use_csv_desc)
            va_loader = DataLoader(va_ds, batch_size=256, shuffle=False, collate_fn=collate, num_workers=0)
            
            meta = json.load(open("encoder_meta_nn.json", "r"))
            enc = Encoder(in_dim=meta.get('in_dim',11), hid=meta.get('hid',128), layers=meta.get('layers',4)).to(device)
            head = Head(hid=meta.get('hid',128), d_dim=len(selected_features)).to(device)
            enc.load_state_dict(torch.load(enc_name, map_location='cpu')); enc.eval()
            head.load_state_dict(torch.load(head_name, map_location='cpu')); head.eval()
            
            Ys, Ps = [], []
            with torch.no_grad():
                for g, y, _, D in va_loader:
                    g = g.to(device); D = D.to(device)
                    Ps.append(head(enc(g), D).cpu()); Ys.append(y.cpu())
            P = torch.cat(Ps); Y = torch.cat(Ys)
            inv = lambda t: t * y_std + y_mean
            if truths is None: truths = inv(Y)
            preds.append(inv(P))
    
    # 集成评估
    if args.ensemble and len(preds) >= 2:
        P = torch.stack(preds, dim=0).mean(0)
        y_true, y_pred = truths, P
        mae = torch.mean(torch.abs(y_true - y_pred)).item()
        mse = torch.mean((y_true - y_pred)**2).item()
        rmse = math.sqrt(mse); var = torch.var(y_true, unbiased=False).item()
        r2 = 1.0 - (mse / (var + 1e-12))
        print("\nEnsemble on validation:",
              {"MAE": round(mae,3), "RMSE": round(rmse,3), "R2": round(r2,2)})
        print(f"(models: {', '.join(map(str,names))}; split seed={args.split_seed})")
    
    # 特征重要性分析
    if args.analyze_importance:
        print("\n" + "="*70)
        print("Feature Importance Analysis")
        print("="*70)
        
        # 使用最后一个训练的模型（或第一个）进行分析
        if len(seeds) > 0:
            s = seeds[0]  # 使用第一个种子的模型
            enc_name = f"encoder_bcf_selected_s{s}.pt"
            head_name = f"head_bcf_selected_s{s}.pt"
            sca_name = f"bcf_scaler_selected_s{s}.json"
            
            print(f"\nLoading model from seed {s} for importance analysis...")
            
            # 加载模型
            with open(sca_name, "r", encoding="utf-8") as f:
                stats = json.load(f)
            
            y_mean = stats['y_mean']
            y_std = stats['y_std']
            d_mean = np.array(stats['d_mean'])
            d_std = np.array(stats['d_std'])
            
            # 创建验证集
            va_ds = DS(va_df, selected_features, y_mean, y_std, d_mean, d_std, use_precomputed=args.use_csv_desc)
            va_loader = DataLoader(va_ds, batch_size=256, shuffle=False, collate_fn=collate, num_workers=0)
            
            # 加载模型
            meta = json.load(open("encoder_meta_nn.json", "r"))
            enc = Encoder(in_dim=meta.get('in_dim',11), hid=meta.get('hid',128), layers=meta.get('layers',4)).to(device)
            head = Head(hid=meta.get('hid',128), d_dim=len(selected_features)).to(device)
            
            enc.load_state_dict(torch.load(enc_name, map_location='cpu'))
            head.load_state_dict(torch.load(head_name, map_location='cpu'))
            enc.eval()
            head.eval()
            
            inv = lambda t: t * y_std + y_mean
            
            # 1. 排列重要性
            importance_dict = compute_permutation_importance(
                enc, head, va_loader, inv, device, selected_features, 
                n_repeats=args.importance_repeats
            )
            
            # 2. 权重分析
            weight_dict = analyze_feature_weights(head, selected_features, hid=meta.get('hid', 128))
            
            # 3. 保存结果
            os.makedirs(args.importance_output, exist_ok=True)
            
            # 保存数值结果
            results_dict = {
                'permutation_importance': importance_dict,
                'weight_importance': weight_dict,
                'model_seed': s,
                'split_seed': args.split_seed
            }
            
            results_path = os.path.join(args.importance_output, 'feature_importance_results.json')
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {results_path}")
            
            # 保存CSV格式
            importance_df = pd.DataFrame({
                'feature': selected_features,
                'permutation_r2_drop': importance_dict['importance_r2'],
                'permutation_mae_increase': importance_dict['importance_mae'],
                'weight_l2_norm': weight_dict['weight_importance']
            })
            importance_df = importance_df.sort_values('permutation_r2_drop', ascending=False)
            csv_path = os.path.join(args.importance_output, 'feature_importance_summary.csv')
            importance_df.to_csv(csv_path, index=False)
            print(f"Summary saved to: {csv_path}")
            
            # 4. 生成可视化
            plot_feature_importance(importance_dict, weight_dict, args.importance_output)
            
            # 5. 打印总结报告
            print("\n" + "="*70)
            print("Feature Importance Summary")
            print("="*70)
            print("\nTop 10 Most Important Features (by R² drop):")
            for i, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']:30s} - R² drop: {row['permutation_r2_drop']:7.4f}, "
                      f"MAE inc: {row['permutation_mae_increase']:7.4f}, "
                      f"Weight: {row['weight_l2_norm']:7.4f}")
            
            print(f"\nAll results saved to: {args.importance_output}/")
            print("="*70)
    
    # 交叉验证
    if args.cross_validation:
        print("\n" + "="*70)
        print("QSAR Internal Validation (Cross-Validation)")
        print("="*70)
        
        cv_metrics, cv_true, cv_pred = leave_one_out_cv(
            tr_df, selected_features, device=device,
            epochs=args.cv_epochs, batch=args.batch,
            init_seed=args.split_seed,
            use_ema=args.ema, ema_decay=args.ema_decay, ema_start=args.ema_start,
            use_precomputed=args.use_csv_desc
        )
        
        print("\n" + "="*70)
        print("交叉验证指标 (Internal Validation)")
        print("="*70)
        print(f"  Q²_LOO  : {cv_metrics['Q2']:.4f}")
        print(f"  R²_cv   : {cv_metrics['R2']:.4f}")
        print(f"  RMSE_cv : {cv_metrics['RMSE']:.4f}")
        print(f"  MAE_cv  : {cv_metrics['MAE']:.4f}")
        print(f"  CCC_cv  : {cv_metrics['CCC']:.4f}")
        print(f"  样本数  : {cv_metrics['N']}")
        
        # 保存交叉验证结果
        cv_output_dir = "cross_validation_results"
        os.makedirs(cv_output_dir, exist_ok=True)
        
        # 保存指标
        cv_results_path = os.path.join(cv_output_dir, f'cv_metrics_seed{args.split_seed}.json')
        with open(cv_results_path, 'w', encoding='utf-8') as f:
            json.dump(cv_metrics, f, indent=2, ensure_ascii=False)
        print(f"\nCV指标已保存: {cv_results_path}")
        
        # 保存预测结果
        cv_predictions = pd.DataFrame({
            'true': cv_true,
            'predicted': cv_pred,
            'residual': cv_true - cv_pred
        })
        cv_pred_path = os.path.join(cv_output_dir, f'cv_predictions_seed{args.split_seed}.csv')
        cv_predictions.to_csv(cv_pred_path, index=False)
        print(f"CV预测已保存: {cv_pred_path}")
        
        print("="*70)

if __name__ == "__main__":
    main()

