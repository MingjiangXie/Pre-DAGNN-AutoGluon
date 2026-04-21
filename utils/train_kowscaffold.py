# -*- coding: utf-8 -*-
# Pretrain on logKow.csv -> save encoder_kow_nnconv.pt + encoder_meta_nn.json
import os, re, math, random, json
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dgl
from dgl.nn import NNConv
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Crippen, Descriptors
from rdkit.Chem import rdMolDescriptors as rdm
RDLogger.DisableLog('rdApp.*')

def set_seed(sd=62):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(sd)

def _norm(s): return re.sub(r'[^a-z0-9]', '', s.lower())
def pick_col(df, cand):
    m = {_norm(c): c for c in df.columns}
    seq = cand if isinstance(cand,(set,list,tuple)) else [cand]
    for k in seq:
        if _norm(k) in m: return m[_norm(k)]
    for k, raw in m.items():
        if any(t in k for t in seq): return raw
    return None

# ---- featurization ----
HYB={Chem.rdchem.HybridizationType.SP:0, Chem.rdchem.HybridizationType.SP2:1,
     Chem.rdchem.HybridizationType.SP3:2, Chem.rdchem.HybridizationType.SP3D:3,
     Chem.rdchem.HybridizationType.SP3D2:4}
def atom_feat(a):
    Z=a.GetAtomicNum(); deg=a.GetTotalDegree(); fc=a.GetFormalCharge()
    arom=1.0 if a.GetIsAromatic() else 0.0; ring=1.0 if a.IsInRing() else 0.0
    hyb=HYB.get(a.GetHybridization(),5)
    base=[min(Z,100)/100.0, min(deg,5)/5.0, max(min(fc,3),-3)/3.0, arom, ring]
    oh=[0]*6; oh[min(hyb,5)]=1
    return torch.tensor(base+oh, dtype=torch.float32)  # 11

def bond_feat(b):
    t=b.GetBondType()
    arom = (b.GetIsAromatic() if hasattr(b,"GetIsAromatic") else b.IsAromatic())
    conj = (b.GetIsConjugated() if hasattr(b,"GetIsConjugated") else
            (b.IsConjugated() if hasattr(b,"IsConjugated") else False))
    one=[1.0 if t==Chem.rdchem.BondType.SINGLE else 0.0,
         1.0 if t==Chem.rdchem.BondType.DOUBLE else 0.0,
         1.0 if t==Chem.rdchem.BondType.TRIPLE else 0.0,
         1.0 if arom else 0.0]
    return torch.tensor(one+[1.0 if conj else 0.0, 1.0 if b.IsInRing() else 0.0], dtype=torch.float32)

E_DIM=6; ADD_SELF_LOOPS=True
def mol_to_graph(mol):
    n=mol.GetNumAtoms()
    atom_x=[atom_feat(mol.GetAtomWithIdx(i)) for i in range(n)]
    src,dst,efeat=[],[],[]
    for b in mol.GetBonds():
        u,v=b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf=bond_feat(b); src+=[u,v]; dst+=[v,u]; efeat+=[bf,bf]
    if ADD_SELF_LOOPS:
        for i in range(n): src.append(i); dst.append(i); efeat.append(torch.zeros(E_DIM))
    g=dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=n)
    g.ndata['h']=torch.stack(atom_x, dim=0); g.edata['e']=torch.stack(efeat, dim=0)
    return g

# ---- dataset ----
class DS(Dataset):
    def __init__(self, df, y_mean=None, y_std=None, fit=False):
        self.smiles=df['SMILES'].tolist(); self.y_raw=df['logKow'].astype(float).values
        G,keep=[],[]
        for i,s in enumerate(self.smiles):
            m=Chem.MolFromSmiles(str(s).strip()); 
            if m is None: continue
            g=mol_to_graph(m); 
            if g.num_nodes()==0: continue
            G.append(g); keep.append(i)
        self.G=G; self.smiles=[self.smiles[i] for i in keep]; self.y_raw=self.y_raw[keep]
        
        if fit:
            self.y_mean=float(self.y_raw.mean()); self.y_std=float(self.y_raw.std()+1e-8)
        else:
            self.y_mean=float(y_mean); self.y_std=float(max(y_std,1e-8))
            
        self.y=torch.tensor((self.y_raw-self.y_mean)/self.y_std, dtype=torch.float32)
        
    def __len__(self): return len(self.G)
    def __getitem__(self,i): return self.G[i], self.y[i], self.smiles[i]

def collate(b):
    gs, ys, ss = zip(*b)
    return dgl.batch(gs), torch.stack(ys), list(ss)

# ---- model ----
class NNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, e_dim=6, edge_h=64):
        super().__init__()
        self.edge_mlp=nn.Sequential(nn.Linear(e_dim, edge_h), nn.ReLU(),
                                    nn.Linear(edge_h, in_dim*out_dim))
        self.nnconv=NNConv(in_dim, out_dim, self.edge_mlp, aggregator_type='mean')
        self.bn=nn.BatchNorm1d(out_dim)
    def forward(self, g, h):
        h=self.nnconv(g, h, g.edata['e'])
        return self.bn(torch.relu(h))

class Encoder(nn.Module):
    def __init__(self, in_dim=11, hid=128, layers=4, dropout=0.1, edge_h=64):
        super().__init__()
        self.layers=nn.ModuleList([NNLayer(in_dim, hid, edge_h=edge_h)]+
                                  [NNLayer(hid, hid, edge_h=edge_h) for _ in range(layers-1)])
        self.dropout=nn.Dropout(dropout)
        self.bn_out=nn.BatchNorm1d(hid, momentum=0.05)
    def forward(self, g):
        h=g.ndata['h']
        for lyr in self.layers:
            h=lyr(g,h); h=self.dropout(h)
        g.ndata['h']=h
        hn=dgl.mean_nodes(g,'h'); hx=dgl.max_nodes(g,'h')
        hn=self.bn_out(hn)
        return torch.cat([hn,hx], dim=1)

class Head(nn.Module):
    def __init__(self, hid=128):
        super().__init__()
        # 移除描述符拼接，全连接层输入维度直接映射自双向聚合的隐向量 (2*hid)
        self.mlp=nn.Sequential(
            nn.Linear(2*hid, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256,1)
        )
    def forward(self, hg):
        return self.mlp(hg).squeeze(1)

def evaluate(enc, head, loader, inv, device):
    enc.eval(); head.eval()
    Ys, Ps = [], []
    with torch.no_grad():
        for g,y,_ in loader:
            g=g.to(device); y=y.to(device)
            Ps.append(head(enc(g)).cpu()); Ys.append(y.cpu())
    Ys=torch.cat(Ys); Ps=torch.cat(Ps)
    y_true, y_pred = inv(Ys), inv(Ps)
    mae=torch.mean(torch.abs(y_true-y_pred)).item()
    mse=torch.mean((y_true-y_pred)**2).item()
    rmse=math.sqrt(mse); var=torch.var(y_true, unbiased=False).item()
    r2=1.0 - (mse/(var+1e-12))
    return {'MAE':mae,'RMSE':rmse,'R2':r2}

# ---- 修正后的稳健骨架划分函数 ----
def scaffold_split(df, frac_train=0.8, seed=62):
    scaf = df['SMILES'].apply(lambda s: MurckoScaffold.MurckoScaffoldSmiles(
        mol=Chem.MolFromSmiles(s), includeChirality=False) if Chem.MolFromSmiles(s) else s)
    df = df.copy()
    df['SCAF'] = scaf
    
    # 将分子按骨架分组并提取索引
    scaffold_groups = df.groupby('SCAF').apply(lambda x: x.index.tolist()).tolist()
    
    # 引入随机种子打乱，随后按骨架组内分子数量降序排列
    random.Random(seed).shuffle(scaffold_groups)
    scaffold_groups.sort(key=len, reverse=True)
    
    tr, va = [], []
    tgt_tr = int(len(df) * frac_train)
    
    for idx_list in scaffold_groups:
        # 核心物理映射：中心偏移评估。防止超大簇导致划分比例严重失衡。
        if len(tr) + len(idx_list) / 2.0 <= tgt_tr:
            tr.extend(idx_list)
        else:
            va.extend(idx_list)
            
    return df.loc[tr].drop(columns=['SCAF']), df.loc[va].drop(columns=['SCAF'])

def train_once(tr_df, va_df, device='cpu', epochs=160, lr=2.5e-3, huber_beta=0.25, weight_decay=8e-5, batch=128):
    fit = DS(tr_df[['SMILES','logKow']], fit=True)
    y_mean,y_std = fit.y_mean,fit.y_std
    tr_ds=DS(tr_df[['SMILES','logKow']], y_mean, y_std)
    va_ds=DS(va_df[['SMILES','logKow']], y_mean, y_std)
    
    tr=DataLoader(tr_ds, batch_size=batch, shuffle=True, collate_fn=collate, num_workers=0, pin_memory=(device=='cuda'))
    va=DataLoader(va_ds, batch_size=batch*2, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=(device=='cuda'))

    enc=Encoder(in_dim=11, hid=128, layers=4, dropout=0.1, edge_h=64).to(device)
    head=Head(hid=128).to(device)

    opt=torch.optim.AdamW(list(enc.parameters())+list(head.parameters()), lr=lr, weight_decay=weight_decay)
    steps_per_epoch=max(1,len(tr))
    sched=torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch)

    scaler=torch.cuda.amp.GradScaler(enabled=(device=='cuda'))
    inv=lambda t: t*y_std + y_mean
    best, best_rmse, bad, patience = None, 1e9, 0, 24
    if hasattr(torch, "set_float32_matmul_precision"): torch.set_float32_matmul_precision('high')

    for ep in range(1, epochs+1):
        enc.train(); head.train()
        for g,y,_ in tr:
            g=g.to(device, non_blocking=True); y=y.to(device)
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                pred=head(enc(g))
                loss=F.smooth_l1_loss(pred,y,beta=huber_beta)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(head.parameters()),5.0)
            scaler.step(opt); scaler.update(); sched.step()
            
        val=evaluate(enc, head, va, inv, device)
        print(f"[Epoch {ep:03d}] valMAE={val['MAE']:.3f} valRMSE={val['RMSE']:.3f} valR2={val['R2']:.3f}")
        
        if val['RMSE'] < best_rmse-1e-3:
            best_rmse=val['RMSE']; bad=0
            best={'enc':enc.state_dict(), 'head':head.state_dict(), 'val':val}
        else:
            bad+=1
            if bad>=patience:
                print("Early stop."); break
    return best

if __name__=="__main__":
    base=os.path.dirname(__file__)
    csv=os.path.join(base,"logKow.csv")
    if not os.path.exists(csv):
        raise SystemExit(f"未找到 {csv} ，请先运行 prep.py 生成。")
    df=pd.read_csv(csv)
    smi=pick_col(df,{"SMILES","final-smiles","finalsmiles"})
    y  =pick_col(df,{"logKow","logkow","kow"})
    df=df[[smi,y]].rename(columns={smi:"SMILES", y:"logKow"}).dropna().drop_duplicates(subset=["SMILES"])

    device='cuda' if torch.cuda.is_available() else 'cpu'
    seeds=[62]
    overall_best, best_seed = None, None
    for sd in seeds:
        print("\n"+"="*80); print(f"Seed = {sd}")
        set_seed(sd)
        
        # -------------- 基于分子骨架严格且均衡划分 --------------
        tr, va = scaffold_split(df, frac_train=0.8, seed=sd)
        print(f"Scaffold split -> Train {len(tr)} | Val {len(va)}")
        
        best = train_once(tr, va, device=device, epochs=160, lr=2.5e-3, batch=128)
        if best is None: continue
        if (overall_best is None) or (best['val']['RMSE']<overall_best['val']['RMSE']):
            overall_best, best_seed = best, sd

    if overall_best is None: raise SystemExit("预训练失败（所有种子未收敛）")

    torch.save(overall_best['enc'], os.path.join(base,"encoder_kow_nnconv.pt"))
    meta={'hid':128,'layers':4,'in_dim':11}
    with open(os.path.join(base,"encoder_meta_nn.json"),"w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("\n"+"="*80); print(f"Best Seed = {best_seed}")
    print("Best Val on logKow:", overall_best['val'])
    print("Saved:", "encoder_kow_nnconv.pt, encoder_meta_nn.json")