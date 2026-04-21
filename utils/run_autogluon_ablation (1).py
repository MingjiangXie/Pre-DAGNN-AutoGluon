#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import math
import numpy as np
import pandas as pd

# 导入基础库函数以确保与 GNN 模型的数据划分绝对一致
from pre_gate_dagnn_cvmean_fixed_v2 import (
    prepare_master_df, load_feature_list, read_outer_membership, 
    get_split_pairs, extract_split_df
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
# 主程序 - 纯 AutoGluon 消融实验
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Standalone AutoGluon Ablation Study with 10x5 Nested CV")
    parser.add_argument("--csv", default="logBCF_5descriptors.csv", help="Input CSV")
    parser.add_argument("--features", default="selected_features_5.json", help="Features JSON")
    parser.add_argument("--outer_split_file", default="outer_split_membership.csv", help="Split definitions")
    parser.add_argument("--outdir", default="autogluon_ablation_repeated_results", help="Output directory")
    parser.add_argument("--time_limit", type=int, default=600, help="AutoGluon time limit per split")
    parser.add_argument("--num_bag_folds", type=int, default=5, help="AutoGluon internal CV folds")
    args = parser.parse_args()

    # 基础配置
    id_col = "SMILES"
    target_col = "logBCF"
    model_name = "Standalone_AutoGluon"  # 明确标识为消融基线模型

    # 结果输出目录配置
    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "per_split"))
    ensure_dir(os.path.join(args.outdir, "predictions"))
    ensure_dir(os.path.join(args.outdir, "ag_models"))
    ensure_dir(os.path.join(args.outdir, "feature_importance"))

    # 1. 加载数据与划分配置 (复用 GNN 切分文件以保证严格对齐)
    feats = load_feature_list(args.features)
    master_df = prepare_master_df(args.csv, feats)
    membership = read_outer_membership(args.outer_split_file)
    split_pairs = get_split_pairs(membership) 
    
    print(f"[Ablation Study] Processing Pure Tabular Features: {feats}")
    print(f"[Total Splits Scheduled] {len(split_pairs)}")

    all_split_metrics = []
    all_predictions = []

    # =========================================================
    # 外层循环：穿越 10 次物理数据空间分布
    # =========================================================
    for split_id, seed in split_pairs:
        print(f"\n" + "="*70)
        print(f"========== Processing Outer Split {split_id:02d} | Seed = {seed} ==========")
        print("="*70)
        
        # 提取当前分布的真实划分集
        train_df = extract_split_df(master_df, membership, split_id, seed, "train")
        val_df = extract_split_df(master_df, membership, split_id, seed, "val")

        # 2. 构建纯宏观描述符数据集 (剥离微观图向量)
        ag_train = train_df[[id_col, "row_id", target_col] + feats].copy()
        ag_val = val_df[[id_col, "row_id", target_col] + feats].copy()

        train_data = ag_train.drop(columns=[id_col, "row_id"])
        val_data = ag_val.drop(columns=[id_col, "row_id"])

        ag_model_dir = os.path.join(args.outdir, "ag_models", f"ag_split_{split_id:02d}")
        
        # 3. 启动内部 5 折交叉验证机制 (基学习器推演)
        if os.path.exists(os.path.join(ag_model_dir, "predictor.pkl")):
            print(f"[Info] Found existing predictor, skipping fit: {ag_model_dir}")
            predictor = TabularPredictor.load(ag_model_dir)
        else:
            print(f"[Info] Starting AutoGluon Training with {args.num_bag_folds}-Fold CV...")
            predictor = TabularPredictor(label=target_col, eval_metric="r2", path=ag_model_dir).fit(
                train_data,
                presets="best_quality",
                time_limit=args.time_limit,
                num_bag_folds=args.num_bag_folds,       
                num_bag_sets=1,        
                dynamic_stacking=True
            )

        print('\n========== 🏆 消融基线排行榜 (Leaderboard) 🏆 ==========')
        lb = predictor.leaderboard(val_data, silent=True)
        display_cols = ['model', 'score_val', 'score_test', 'fit_time', 'time_train']
        exist_cols = [c for c in display_cols if c in lb.columns]
        print(lb[exist_cols].head(5).to_string())

        # ================= 记录当折最佳宏观基线模型 =================
        best_model_name = predictor.model_best
        best_score_val = lb.loc[lb['model'] == best_model_name, 'score_val'].values[0]
        print(f"\n[Info] 严格依据内部评测准则，已选定最高 score_val 的集成策略: ")
        print(f"       >>> 基线预测器: {best_model_name} (内部 R2_cv: {best_score_val:.4f}) <<<")
        # =============================================================

        # 4. 严谨数学指标推导
        pred_train = predictor.predict(train_data).values
        pred_val = predictor.predict(val_data).values
        oof_preds = predictor.predict_oof().values

        tr_m = regression_metrics(ag_train[target_col].values, pred_train)
        va_m = regression_metrics(ag_val[target_col].values, pred_val)
        cv_m = regression_metrics(ag_train[target_col].values, oof_preds)
        delta_r2 = tr_m["R2"] - va_m["R2"]

        # 5. 导出消融对照预测值
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

        # 6. 对齐记录单次划分全量指标
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

        # 7. 生成纯宏观物理描述符的特征重要性矩阵
        print(f"[FI Analysis] 正在进行纯特征置换排列检验 (Permutation FI) ...")
        fi = predictor.feature_importance(val_data, silent=True)
        fi.to_csv(
            os.path.join(args.outdir, "feature_importance", f"{model_name}_split_{split_id:02d}_seed_{seed}_FI.csv"),
            encoding="utf-8-sig"
        )

        print(f"[Results for Split {split_id:02d}] R2_cv: {cv_m['R2']:.4f} | R2_val: {va_m['R2']:.4f}")

    # =========================================================
    # 全局指标汇总分析 (构建对照表 S8)
    # =========================================================
    print("\n" + "="*70)
    print("Aggregating Ablation Statistics Across All Splits...")
    
    if not all_split_metrics:
        raise SystemExit("Execution Failed: Check dataset parsing or configuration.")

    metrics_df = pd.DataFrame(all_split_metrics)
    metrics_df.to_csv(
        os.path.join(args.outdir, "all_models_all_splits_metrics.csv"),
        index=False, encoding="utf-8-sig"
    )

    # 导出消融基线的最优堆叠策略频次
    model_counts = metrics_df['Selected_AG_Model'].value_counts().reset_index()
    model_counts.columns = ['Selected_AG_Model', 'Selection_Count']
    model_counts.to_csv(
        os.path.join(args.outdir, "selected_models_frequency_summary.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("\n[Info] 基线预测器选择频次分布：")
    print(model_counts.to_string(index=False))

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

    # 生成同尺度 Table S8 用于消融性能差值对比
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

    print(f"\n[Ablation Done] Successfully compiled baseline across {len(all_split_metrics)} rigorous splits.")
    print(f"Results located in: {args.outdir}")

if __name__ == "__main__":
    main()