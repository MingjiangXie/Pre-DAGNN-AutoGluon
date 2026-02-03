#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ml_9models_baseline.py

在同一 80/20 划分和 5 折 CV 下比较 9 种 ML 模型：
- LR      : LinearRegression
- Ridge   : Ridge
- Lasso   : Lasso
- ENet    : ElasticNet
- KNN     : KNeighborsRegressor
- SVR     : SVR (RBF)
- RF      : RandomForestRegressor
- GBR     : GradientBoostingRegressor
- MLP     : MLPRegressor

输出：
- 训练 / CV / 外部验证集 R², RMSE, MAE
- 额外增加 ΔR² = |R²_train − R²_val|
"""

import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone


# ---------- 小工具 ----------

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def eval_cv_fixed_folds(name, base_model, X_train, y_train, fold_ids_train):
    """在给定的折号 fold_ids_train (1..k) 上做 k 折 CV"""
    folds = sorted([f for f in np.unique(fold_ids_train) if f > 0])
    n_train = X_train.shape[0]
    oof_pred = np.zeros(n_train, dtype=float)

    print(f"\n==== {name}: {len(folds)}-fold CV on TRAIN (fixed folds) ====")

    for f in folds:
        idx_tr = np.where(fold_ids_train != f)[0]
        idx_va = np.where(fold_ids_train == f)[0]

        X_tr, X_va = X_train[idx_tr], X_train[idx_va]
        y_tr, y_va = y_train[idx_tr], y_train[idx_va]

        model = clone(base_model)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)

        oof_pred[idx_va] = pred

        r2 = r2_score(y_va, pred)
        r = rmse(y_va, pred)
        mae = mean_absolute_error(y_va, pred)
        print(f"[Fold {f}] R2={r2:.3f}, RMSE={r:.3f}, MAE={mae:.3f}")

    # 整体 CV 指标（Q²）
    r2_cv_all = r2_score(y_train, oof_pred)
    rmse_cv_all = rmse(y_train, oof_pred)
    mae_cv_all = mean_absolute_error(y_train, oof_pred)

    print(f"[{name}] TRAIN CV (Q²): R2_cv={r2_cv_all:.3f}, "
          f"RMSE_cv={rmse_cv_all:.3f}, MAE_cv={mae_cv_all:.3f}")

    return oof_pred, (r2_cv_all, rmse_cv_all, mae_cv_all)


def train_and_eval_full(name, base_model, X_train, y_train, X_ext, y_ext):
    """用全部训练集拟合，并在训练集/外部验证集上评估"""
    model = clone(base_model)
    model.fit(X_train, y_train)

    # TRAIN
    y_tr_pred = model.predict(X_train)
    r2_tr = r2_score(y_train, y_tr_pred)
    rmse_tr = rmse(y_train, y_tr_pred)
    mae_tr = mean_absolute_error(y_train, y_tr_pred)

    # EXTERNAL
    y_ext_pred = model.predict(X_ext)
    r2_val = r2_score(y_ext, y_ext_pred)
    rmse_val = rmse(y_ext, y_ext_pred)
    mae_val = mean_absolute_error(y_ext, y_ext_pred)

    delta_r2 = r2_tr - r2_val
    delta_r2_abs = abs(delta_r2)

    print(f"\n[{name}] TRAIN: R2={r2_tr:.3f}, RMSE={rmse_tr:.3f}, MAE={mae_tr:.3f}")
    print(f"[{name}]  VAL : R2_val={r2_val:.3f}, RMSE_val={rmse_val:.3f}, MAE_val={mae_val:.3f}")
    print(f"[{name}] ΔR2(train−val) = {delta_r2:.3f} (abs={delta_r2_abs:.3f})")

    return r2_tr, rmse_tr, mae_tr, r2_val, rmse_val, mae_val, y_ext_pred, delta_r2_abs


# ---------- 主流程 ----------

def main():
    split_csv = "MLcontrast_with_split_rf.csv"
    if not os.path.exists(split_csv):
        raise FileNotFoundError(
            "找不到 MLcontrast_with_split_rf.csv，请先运行 RF 脚本生成该文件。"
        )

    df = pd.read_csv(split_csv)
    print(f"读取 {split_csv}, 形状: {df.shape}")

    feature_cols = [
        "P_VSA_v_3",
        "P_VSA_p_2",
        "MLOGP2",
        "LOGP99",
        "ESOL",
        "BLTA96",
        "SM1_Dz(p)",
        "SpMaxA_B(s)",
    ]
    target_col = "logBCF"

    X_all = df[feature_cols].values
    y_all = df[target_col].values

    if "ext_val_flag" not in df.columns or "cv5_fold" not in df.columns:
        raise ValueError("文件中缺少 ext_val_flag 或 cv5_fold 列，请确认是 RF 脚本生成的版本。")

    train_mask = df["ext_val_flag"] == 0
    ext_mask = df["ext_val_flag"] == 1

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_ext, y_ext = X_all[ext_mask], y_all[ext_mask]

    print(f"训练集: {X_train.shape[0]} 条, 外部验证集: {X_ext.shape[0]} 条")

    fold_ids_train = df.loc[train_mask, "cv5_fold"].values.astype(int)
    train_idx_all = np.where(train_mask)[0]
    ext_idx_all = np.where(ext_mask)[0]

    # 9 个模型定义（已去掉 ExtraTrees，只保留 RF）
    model_defs = {
        "LR": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1e-3, max_iter=10000),
        "ENet": ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=10000),

        # KNN：保守一点，避免严重过拟合（如果你想继续用旧参数，可以改回去）
        "KNN": KNeighborsRegressor(
            n_neighbors=25,
            weights="distance",
            p=2,
        ),

        "SVR": SVR(
            kernel="rbf",
            C=5.0,
            gamma="scale",
            epsilon=0.1,
        ),

        "RF": RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),

        "GBR": GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=42,
        ),

        "MLP": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-3,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        ),
    }

    # 统一加 StandardScaler
    pipelines = {
        name: Pipeline([("scaler", StandardScaler()), ("model", mdl)])
        for name, mdl in model_defs.items()
    }

    # 为每个模型准备预测列
    for name in model_defs.keys():
        low = name.lower()
        df[f"{low}_cv_oof_pred"] = np.nan
        df[f"{low}_ext_pred"] = np.nan

    summary_records = []

    for name, pipe in pipelines.items():
        print("\n================ {} ================".format(name))
        oof_pred, (r2_cv, rmse_cv, mae_cv) = eval_cv_fixed_folds(
            name, pipe, X_train, y_train, fold_ids_train
        )

        r2_tr, rmse_tr, mae_tr, r2_val, rmse_val, mae_val, ext_pred, delta_r2_abs = \
            train_and_eval_full(name, pipe, X_train, y_train, X_ext, y_ext)

        low = name.lower()
        df.loc[train_idx_all, f"{low}_cv_oof_pred"] = oof_pred
        df.loc[ext_idx_all, f"{low}_ext_pred"] = ext_pred

        summary_records.append({
            "Model": name,
            "R2_train": r2_tr, "RMSE_train": rmse_tr, "MAE_train": mae_tr,
            "R2_cv": r2_cv, "RMSE_cv": rmse_cv, "MAE_cv": mae_cv,
            "R2_val": r2_val, "RMSE_val": rmse_val, "MAE_val": mae_val,
            "Delta_R2_abs": delta_r2_abs,   # |R2_train − R2_val|
        })

    # 保存详细数据
    out_csv = "MLcontrast_with_split_9models.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n已保存带 9 个模型预测的文件: {out_csv}")

    # 摘要表
    summary_df = pd.DataFrame(summary_records)
    print("\n===== Summary of 9 ML models =====")
    print(summary_df)

    summary_df.to_csv("ML_9models_summary.csv", index=False)
    print("\n已保存摘要表: ML_9models_summary.csv")


if __name__ == "__main__":
    main()
