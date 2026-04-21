#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 导入基础库函数 (确保 pre_gate_dagnn_cvmean_fixed_v2.py 与此脚本在同一目录下)
from pre_gate_dagnn_cvmean_fixed_v2 import (
    prepare_master_df, load_feature_list, read_outer_membership, 
    get_split_pairs, extract_split_df, build_dataset_eval, collate, Encoder
)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    raise SystemExit("请先安装 AutoGluon: pip install autogluon")

# =========================================================
# 基础工具与指标函数
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def regression_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred))
    }

def format_mean_std(mean_val, std_val, ndigits=3):
    return f"{mean_val:.{ndigits}f} ± {std_val:.{ndigits}f}"

def summarize_mean_std(metrics_df, group_col="Model"):
    metric_cols = [
        "R2_tra", "RMSE_tra", "MAE_tra",
        "R2_val", "RMSE_val", "MAE_val",
        "R2_cv", "RMSE_cv", "MAE_cv",
        "Delta_R2"
    ]

    rows = []
    for model, g in metrics_df.groupby(group_col):
        row = {"Model": model, "n_splits": len(g)}
        for c in metric_cols:
            vals = g[c].astype(float).values
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{c}_mean"] = mean_v
            row[f"{c}_std"] = std_v
            row[f"{c}_mean±std"] = format_mean_std(mean_v, std_v, 3)
        rows.append(row)

    out = pd.DataFrame(rows)
    return out

# =========================================================
# 主程序
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="GNN + AutoGluon Fusion with Repeated 10x5 Nested CV")
    parser.add_argument("--csv", default="logBCF_5descriptors.csv", help="Input CSV")
    parser.add_argument("--features", default="selected_features_5.json", help="Features JSON")
    parser.add_argument("--outer_split_file", default="outer_split_membership.csv", help="Split definitions")
    parser.add_argument("--ckpt_base_dir", default="pregate_cvmean_10seeds_tuned/checkpoints", help="Dir containing GNN checkpoints")
    parser.add_argument("--outdir", default="gnn_autogluon_repeated_results", help="Output directory")
    parser.add_argument("--time_limit", type=int, default=600, help="AutoGluon time limit per split")
    parser.add_argument("--num_bag_folds", type=int, default=5, help="AutoGluon internal CV folds")
    args = parser.parse_args()

    # 基础配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    id_col = "SMILES"
    target_col = "logBCF"
    model_name = "GNN_AutoGluon_Fusion"

    # 结果输出目录配置
    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "per_split"))
    ensure_dir(os.path.join(args.outdir, "predictions"))
    ensure_dir(os.path.join(args.outdir, "ag_models"))
    ensure_dir(os.path.join(args.outdir, "feature_importance"))

    # 1. 加载数据与划分配置
    feats = load_feature_list(args.features)
    master_df = prepare_master_df(args.csv, feats)
    membership = read_outer_membership(args.outer_split_file)
    split_pairs = get_split_pairs(membership) 
    
    print(f"[Device] {device}")
    print(f"[Total Splits Scheduled] {len(split_pairs)}")

    all_split_metrics = []
    all_predictions = []

    # =========================================================
    # 外层循环：遍历所有的 Split 确保统计学效应
    # =========================================================
    for split_id, seed in split_pairs:
        print(f"\n" + "="*70)
        print(f"========== Processing Outer Split {split_id:02d} | Seed = {seed} ==========")
        print("="*70)
        
        # 定位当前 Split 对应的 GNN 权重与缩放器
        prefix = f"Pre-Gate-DAGNN-CVMean_split_{split_id:02d}_seed_{seed}"
        encoder_pt = os.path.join(args.ckpt_base_dir, f"{prefix}_encoder.pt")
        scaler_json = os.path.join(args.ckpt_base_dir, f"{prefix}_scaler.json")
        
        if not os.path.exists(encoder_pt) or not os.path.exists(scaler_json):
            print(f"[Warning] 找不到当前 Split ({split_id:02d}) 的 GNN 权重或缩放器，跳过此循环。")
            continue

        train_df = extract_split_df(master_df, membership, split_id, seed, "train")
        val_df = extract_split_df(master_df, membership, split_id, seed, "val")

        # 2. 加载缩放器与构建 GNN 数据集
        with open(scaler_json, "r", encoding="utf-8") as f:
            stats = json.load(f)
        y_mean, y_std = stats["y_mean"], stats["y_std"]
        d_mean = np.array(stats["d_mean"], dtype=np.float32)
        d_std = np.array(stats["d_std"], dtype=np.float32)

        train_ds = build_dataset_eval(train_df, feats, y_mean, y_std, d_mean, d_std)
        val_ds = build_dataset_eval(val_df, feats, y_mean, y_std, d_mean, d_std)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate)

        # 3. 提取 GNN 物理拓扑特征
        enc = Encoder(in_dim=11, hid=128, layers=4, dropout=0.0, edge_h=64).to(device)
        enc.load_state_dict(torch.load(encoder_pt, map_location=device))
        enc.eval()

        @torch.no_grad()
        def extract_features(loader):
            hgs, ds, ys, smis, row_ids = [], [], [], [], []
            for g, y, smi, row_id, D in loader:
                hgs.append(enc(g.to(device)).cpu().numpy())
                ds.append(D.numpy())
                ys.append((y.numpy() * y_std) + y_mean)
                smis.extend(smi)
                row_ids.extend(row_id)
            return np.vstack(hgs), np.vstack(ds), np.concatenate(ys), smis, row_ids

        print("[Step 1] Extracting GNN Topology Features...")
        hg_tr, d_tr, y_tr, smi_tr, rid_tr = extract_features(train_loader)
        hg_va, d_va, y_va, smi_va, rid_va = extract_features(val_loader)

        # 4. 构建供 AutoGluon 消费的多模态融合 DataFrame
        def build_ag_df(hg, d, y, smi, rid):
            df = pd.DataFrame(hg, columns=[f"gnn_feat_{i}" for i in range(hg.shape[1])])
            for i, f_name in enumerate(feats):
                df[f_name] = d[:, i]
            df[target_col] = y
            df[id_col] = smi
            df["row_id"] = rid
            return df

        ag_train = build_ag_df(hg_tr, d_tr, y_tr, smi_tr, rid_tr)
        ag_val = build_ag_df(hg_va, d_va, y_va, smi_va, rid_va)

        train_data = ag_train.drop(columns=[id_col, "row_id"])
        val_data = ag_val.drop(columns=[id_col, "row_id"])

        ag_model_dir = os.path.join(args.outdir, "ag_models", f"ag_split_{split_id:02d}")
        
        # 5. 启动 AutoGluon (执行内部 5 折 CV)
        if os.path.exists(os.path.join(ag_model_dir, "predictor.pkl")):
            print(f"[Step 2] Found existing predictor, skipping fit: {ag_model_dir}")
            predictor = TabularPredictor.load(ag_model_dir)
        else:
            print(f"[Step 2] Starting AutoGluon Training with {args.num_bag_folds}-Fold CV...")
            predictor = TabularPredictor(label=target_col, eval_metric="r2", path=ag_model_dir).fit(
                train_data,
                presets="best_quality",
                time_limit=args.time_limit,
                num_bag_folds=args.num_bag_folds,       
                num_bag_sets=1,        
                dynamic_stacking=True
            )

        print('\n========== 🏆 物理特征融合排行榜 (Leaderboard) 🏆 ==========')
        lb = predictor.leaderboard(val_data, silent=True)
        display_cols = ['model', 'score_val', 'score_test', 'fit_time', 'time_train']
        exist_cols = [c for c in display_cols if c in lb.columns]
        print(lb[exist_cols].head(5).to_string())

        # ================= 显式打印选择的最佳模型 =================
        best_model_name = predictor.model_best
        best_score_val = lb.loc[lb['model'] == best_model_name, 'score_val'].values[0]
        print(f"\n[Info] 严格依据内部评测准则，AutoGluon 已选定最高 score_val 的模型出战: ")
        print(f"       >>> 终极预测器: {best_model_name} (内部 R2_cv: {best_score_val:.4f}) <<<")
        # =============================================================

        # 6. 严谨指标计算
        pred_train = predictor.predict(train_data).values
        pred_val = predictor.predict(val_data).values
        oof_preds = predictor.predict_oof().values

        tr_m = regression_metrics(ag_train[target_col].values, pred_train)
        va_m = regression_metrics(ag_val[target_col].values, pred_val)
        cv_m = regression_metrics(ag_train[target_col].values, oof_preds)
        delta_r2 = tr_m["R2"] - va_m["R2"]

        # 7. 导出当前 Split 的预测值 DataFrame
        pred_train_df = ag_train[[id_col, "row_id", target_col]].copy()
        pred_train_df["split_id"] = split_id
        pred_train_df["seed"] = seed
        pred_train_df["Model"] = model_name
        pred_train_df["set"] = "train"
        pred_train_df["y_true"] = ag_train[target_col].values
        pred_train_df["y_pred"] = pred_train
        pred_train_df["residual"] = pred_train_df["y_true"] - pred_train_df["y_pred"]

        pred_val_df = ag_val[[id_col, "row_id", target_col]].copy()
        pred_val_df["split_id"] = split_id
        pred_val_df["seed"] = seed
        pred_val_df["Model"] = model_name
        pred_val_df["set"] = "val"
        pred_val_df["y_true"] = ag_val[target_col].values
        pred_val_df["y_pred"] = pred_val
        pred_val_df["residual"] = pred_val_df["y_true"] - pred_val_df["y_pred"]

        pred_one = pd.concat([pred_train_df, pred_val_df], axis=0, ignore_index=True)
        pred_one.to_csv(
            os.path.join(args.outdir, "predictions", f"{model_name}_split_{split_id:02d}_seed_{seed}_predictions.csv"),
            index=False, encoding="utf-8-sig"
        )
        all_predictions.append(pred_one)

        # 8. 记录当前 Split 的 Metrics
        metric_row = {
            "split_id": split_id,
            "seed": seed,
            "Model": model_name,
            "Selected_AG_Model": best_model_name,
            "n_train": len(train_data),
            "n_val": len(val_data),
            "R2_tra": tr_m["R2"], "RMSE_tra": tr_m["RMSE"], "MAE_tra": tr_m["MAE"],
            "R2_val": va_m["R2"], "RMSE_val": va_m["RMSE"], "MAE_val": va_m["MAE"],
            "R2_cv": cv_m["R2"], "RMSE_cv": cv_m["RMSE"], "MAE_cv": cv_m["MAE"],
            "Delta_R2": delta_r2
        }
        all_split_metrics.append(metric_row)

        per_split_df = pd.DataFrame([metric_row])
        per_split_df.to_csv(
            os.path.join(args.outdir, "per_split", f"{model_name}_split_{split_id:02d}_seed_{seed}_metrics.csv"),
            index=False, encoding="utf-8-sig"
        )

        # 9. 补齐特征重要性分析
        print(f"[Step 3] 正在计算异构特征重要性 (Permutation FI) ...")
        fi = predictor.feature_importance(val_data, silent=True)
        fi.to_csv(
            os.path.join(args.outdir, "feature_importance", f"{model_name}_split_{split_id:02d}_seed_{seed}_FI.csv"),
            encoding="utf-8-sig"
        )

        print(f"[Results for Split {split_id:02d}] R2_cv: {cv_m['R2']:.4f} | R2_val: {va_m['R2']:.4f}")

    # =========================================================
    # 汇总：计算具有统计学效应的均值和标准差
    # =========================================================
    print("\n" + "="*70)
    print("Aggregating Final Statistics Across All Splits...")
    
    if not all_split_metrics:
        raise SystemExit("No splits were processed successfully. Check your checkpoint paths.")

    metrics_df = pd.DataFrame(all_split_metrics)
    metrics_df.to_csv(
        os.path.join(args.outdir, "all_models_all_splits_metrics.csv"),
        index=False, encoding="utf-8-sig"
    )

    # ===== NEW: 终极预测器选择频次汇总表 =====
    model_counts = metrics_df['Selected_AG_Model'].value_counts().reset_index()
    model_counts.columns = ['Selected_AG_Model', 'Selection_Count']
    model_counts.to_csv(
        os.path.join(args.outdir, "selected_models_frequency_summary.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("\n[Info] 终极预测器选择频次汇总：")
    print(model_counts.to_string(index=False))
    # =========================================

    all_predictions_df = pd.concat(all_predictions, axis=0, ignore_index=True)
    all_predictions_df.to_csv(
        os.path.join(args.outdir, "all_models_all_predictions_long.csv"),
        index=False, encoding="utf-8-sig"
    )

    summary_df = summarize_mean_std(metrics_df, group_col="Model")
    summary_df.to_csv(
        os.path.join(args.outdir, "summary_mean_std_by_model.csv"),
        index=False, encoding="utf-8-sig"
    )

    # 生成与基线严格对齐的 Table S8
    table_s8 = summary_df[[
        "Model", "R2_tra_mean±std", "RMSE_tra_mean±std", "MAE_tra_mean±std",
        "R2_val_mean±std", "RMSE_val_mean±std", "MAE_val_mean±std",
        "R2_cv_mean±std", "RMSE_cv_mean±std", "MAE_cv_mean±std", "Delta_R2_mean±std"
    ]].copy()

    table_s8.columns = [
        "Model", "R2_tra", "RMSE_tra", "MAE_tra", 
        "R2_val", "RMSE_val", "MAE_val", 
        "R2_cv", "RMSE_cv", "MAE_cv", "Delta_R2"
    ]
    table_s8.to_csv(
        os.path.join(args.outdir, "Table_S8_repeated_split_mean_std.csv"),
        index=False, encoding="utf-8-sig"
    )

    print(f"\n[Done] Pipeline executed successfully across {len(all_split_metrics)} splits.")
    print("Key output files with statistical significance:")
    print(f"  - {os.path.join(args.outdir, 'Table_S8_repeated_split_mean_std.csv')} (Mean±Std formatting)")
    print(f"  - {os.path.join(args.outdir, 'summary_mean_std_by_model.csv')}")
    print(f"  - {os.path.join(args.outdir, 'selected_models_frequency_summary.csv')} (Model Selection Frequencies)")

if __name__ == "__main__":
    main()