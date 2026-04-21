#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


# =========================================================
# 基础函数
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_split_dir_name(name):
    """ split_01_seed_42 -> (1, 42) """
    m = re.match(r"split_(\d+)_seed_(\d+)", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def make_strat_bins(y, n_bins=10):
    """ 回归任务分层：按 logBCF 分位数分箱 """
    bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    return bins


def to_numeric_descriptors(df, id_col, target_col):
    """ 保留 id/target，并将描述符列统一转为数值 """
    if id_col not in df.columns:
        raise ValueError(f"缺少列: {id_col}")
    if target_col not in df.columns:
        raise ValueError(f"缺少列: {target_col}")
    desc_cols = [c for c in df.columns if c not in [id_col, target_col]]
    desc = df[desc_cols].apply(pd.to_numeric, errors="coerce")
    return df[[id_col]].copy(), df[target_col].copy(), desc


# =========================================================
# 第0步：全数据无监督基础过滤
# =========================================================

def filter_missing_columns(desc_df):
    keep_cols = desc_df.columns[desc_df.notna().all()].tolist()
    return keep_cols


def filter_low_variance(desc_df, threshold=1e-4):
    stds = desc_df.std(axis=0, skipna=True)
    keep_cols = stds[stds >= threshold].index.tolist()
    return keep_cols, stds


def filter_high_correlation(desc_df, threshold=0.95):
    """ 高相关过滤：保留 std 更大的列 """
    if desc_df.shape[1] <= 1:
        return desc_df.columns.tolist()
    stds = desc_df.std(axis=0)
    sorted_cols = stds.sort_values(ascending=False).index.tolist()
    corr_matrix = desc_df[sorted_cols].corr().abs()
    keep_mask = np.ones(len(sorted_cols), dtype=bool)
    for i in range(len(sorted_cols)):
        if keep_mask[i]:
            for j in range(i + 1, len(sorted_cols)):
                if keep_mask[j] and corr_matrix.iloc[i, j] >= threshold:
                    keep_mask[j] = False
    keep_cols = [sorted_cols[i] for i in range(len(sorted_cols)) if keep_mask[i]]
    return keep_cols


def global_unsupervised_filter(df, id_col, target_col, std_thresh=1e-4, corr_thresh=0.95):
    id_df, y, desc = to_numeric_descriptors(df, id_col, target_col)
    n_initial = desc.shape[1]

    # 1. 去缺失列
    keep_no_missing = filter_missing_columns(desc)
    desc = desc[keep_no_missing]

    # 2. 去低方差列
    keep_std, _ = filter_low_variance(desc, threshold=std_thresh)
    desc = desc[keep_std]

    # 3. 去高相关列
    keep_uncorr = filter_high_correlation(desc, threshold=corr_thresh)
    desc = desc[keep_uncorr]

    filtered_df = pd.concat([id_df, y.rename(target_col), desc], axis=1)

    stats = {
        "n_rows": int(len(filtered_df)),
        "n_desc_initial": int(n_initial),
        "n_after_no_missing": int(len(keep_no_missing)),
        "n_after_low_variance": int(len(keep_std)),
        "n_after_high_corr": int(len(keep_uncorr))
    }
    return filtered_df, stats


# =========================================================
# 第1步：每个训练集内部的监督筛选
# =========================================================

def pearson_filter(train_x, train_y, threshold=0.3):
    tmp = train_x.copy()
    tmp["__target__"] = train_y.values
    corr = tmp.corr(numeric_only=True)["__target__"].drop("__target__").abs()
    keep_cols = corr[corr >= threshold].index.tolist()
    return keep_cols, corr


def compute_rf_importance(X, y, random_state=42):
    model = RandomForestRegressor(
        n_estimators=500, random_state=random_state, n_jobs=-1
    )
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=X.columns)


def compute_mi_scores(X, y, random_state=42):
    mi = mutual_info_regression(X, y, random_state=random_state)
    return pd.Series(mi, index=X.columns)


def compute_pearson_scores(X, y):
    tmp = X.copy()
    tmp["__target__"] = y.values
    corr = tmp.corr(numeric_only=True)["__target__"].drop("__target__").abs()
    return corr


def compute_rfe_ranking(X, y, n_features_to_select=None):
    if n_features_to_select is None:
        n_features_to_select = max(1, min(20, X.shape[1] // 2 if X.shape[1] > 2 else X.shape[1]))
    estimator = LinearRegression()
    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X, y)
    return pd.Series(selector.ranking_, index=X.columns)


def combine_rankings(rf_imp, mi_scores, pearson_scores, rfe_ranking):
    rf_rank = rf_imp.rank(ascending=False, method="average")
    mi_rank = mi_scores.rank(ascending=False, method="average")
    pearson_rank = pearson_scores.rank(ascending=False, method="average")
    rfe_rank = rfe_ranking.rank(ascending=True, method="average")  # ranking越小越好

    combined = pd.DataFrame({
        "RF_Importance": rf_imp,
        "RF_Rank": rf_rank,
        "MI_Score": mi_scores,
        "MI_Rank": mi_rank,
        "Pearson_Corr": pearson_scores,
        "Pearson_Rank": pearson_rank,
        "RFE_Ranking": rfe_ranking,
        "RFE_Rank": rfe_rank
    })

    combined["Average_Rank"] = combined[
        ["RF_Rank", "MI_Rank", "Pearson_Rank", "RFE_Rank"]
    ].mean(axis=1)
    combined = combined.sort_values(["Average_Rank", "RF_Rank"], ascending=[True, True])
    return combined


# =========================================================
# CV评估与选择规则
# =========================================================

def evaluate_topn_cv(train_df, ordered_features, subset_sizes, target_col="logBCF", cv_folds=10, random_state=42):
    """ 对 top-n 特征子集进行 10 折 CV
        输出： 1) 每折结果 2) 每个 n 的汇总结果
    """
    y = train_df[target_col].values
    fold_rows = []
    summary_rows = []
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for n in subset_sizes:
        feats = ordered_features[:n]
        X = train_df[feats].copy()
        r2_list = []
        rmse_list = []
        for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
            X_tr = X.iloc[tr_idx]
            X_va = X.iloc[va_idx]
            y_tr = y[tr_idx]
            y_va = y[va_idx]

            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("rf", RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1))
            ])
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)

            r2 = r2_score(y_va, y_pred)
            rmse = np.sqrt(mean_squared_error(y_va, y_pred))

            r2_list.append(r2)
            rmse_list.append(rmse)

            fold_rows.append({
                "n_features": n,
                "fold_id": fold_id,
                "r2": r2,
                "rmse": rmse
            })

        summary_rows.append({
            "n_features": n,
            "mean_r2": float(np.mean(r2_list)),
            "std_r2": float(np.std(r2_list, ddof=1)),
            "mean_rmse": float(np.mean(rmse_list)),
            "std_rmse": float(np.std(rmse_list, ddof=1))
        })

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("n_features").reset_index(drop=True)
    return fold_df, summary_df


def choose_compact_n(summary_df, delta_r2=0.03):
    """ 规则：
        1) 找到最高 mean_r2
        2) 在 mean_r2 >= best_r2 - delta_r2 的候选里
        3) 选特征数最少的那个
    """
    best_r2 = summary_df["mean_r2"].max()
    candidates = summary_df[summary_df["mean_r2"] >= best_r2 - delta_r2].copy()
    candidates = candidates.sort_values("n_features").reset_index(drop=True)
    recommended = candidates.iloc[0].to_dict()
    recommended["best_r2"] = float(best_r2)
    recommended["delta_r2"] = float(delta_r2)
    return recommended


# =========================================================
# 单个 split 的完整流程
# =========================================================

def run_one_split(train_df, test_df, id_col, target_col, pearson_thresh, subset_sizes, cv_folds, random_state, delta_r2):
    """ 一个 split 内部：
        1) Pearson预筛
        2) RF + MI + Pearson + RFE 综合排序
        3) 比较 top5~top12
        4) 用 delta rule 选更紧凑的 n
    """
    train_x = train_df.drop(columns=[id_col, target_col]).copy()
    train_y = train_df[target_col].copy()

    # Step A: Pearson预筛
    keep_pearson, pearson_corr = pearson_filter(train_x, train_y, threshold=pearson_thresh)
    train_x = train_x[keep_pearson]

    if train_x.shape[1] == 0:
        raise ValueError("Pearson过滤后没有特征剩余。")

    # Step B: 四种方法联合评分
    rf_imp = compute_rf_importance(train_x, train_y, random_state=random_state)
    mi_scores = compute_mi_scores(train_x, train_y, random_state=random_state)
    pearson_scores = compute_pearson_scores(train_x, train_y)
    n_rfe = max(1, min(20, train_x.shape[1] // 2 if train_x.shape[1] > 2 else train_x.shape[1]))
    rfe_ranking = compute_rfe_ranking(train_x, train_y, n_features_to_select=n_rfe)

    combined = combine_rankings(rf_imp, mi_scores, pearson_scores, rfe_ranking)
    ordered_features = combined.index.tolist()

    # Step C: 只比较 top5~top12
    valid_subset_sizes = [n for n in subset_sizes if n <= len(ordered_features)]
    if len(valid_subset_sizes) == 0:
        raise ValueError("可比较的 subset_sizes 为空。")

    fold_df, summary_df = evaluate_topn_cv(
        train_df=pd.concat([train_df[[id_col, target_col]], train_x], axis=1),
        ordered_features=ordered_features,
        subset_sizes=valid_subset_sizes,
        target_col=target_col,
        cv_folds=cv_folds,
        random_state=random_state
    )

    recommended = choose_compact_n(summary_df, delta_r2=delta_r2)
    selected_n = int(recommended["n_features"])
    final_features = ordered_features[:selected_n]

    # 最终应用到 train/test
    train_final = pd.concat([
        train_df[[id_col, target_col]],
        train_df[final_features]
    ], axis=1)
    test_final = pd.concat([
        test_df[[id_col, target_col]],
        test_df[final_features]
    ], axis=1)

    return {
        "pearson_kept": keep_pearson,
        "combined_ranking": combined,
        "cv_fold_results": fold_df,
        "cv_summary": summary_df,
        "recommended": recommended,
        "final_features": final_features,
        "train_final": train_final,
        "test_final": test_final
    }


# =========================================================
# 稳定特征池精炼
# =========================================================

def collect_rank_summary(workdir, stable_features):
    rank_rows = []
    for name in sorted(os.listdir(workdir)):
        split_dir = os.path.join(workdir, name)
        if not os.path.isdir(split_dir):
            continue
        split_id, seed = parse_split_dir_name(name)
        if split_id is None:
            continue
        rank_file = os.path.join(split_dir, "combined_feature_ranking.csv")
        if not os.path.exists(rank_file):
            continue

        rank_df = pd.read_csv(rank_file, index_col=0).reset_index().rename(columns={"index": "feature"})
        rank_df = rank_df[rank_df["feature"].isin(stable_features)].copy()
        rank_df["split_id"] = split_id
        rank_df["seed"] = seed
        rank_rows.append(rank_df[["split_id", "seed", "feature", "Average_Rank"]])

    if len(rank_rows) == 0:
        raise ValueError("没有找到可用的 combined_feature_ranking.csv。")

    all_rank_df = pd.concat(rank_rows, axis=0, ignore_index=True)

    rank_summary = (
        all_rank_df.groupby("feature", as_index=False)
        .agg(
            mean_average_rank=("Average_Rank", "mean"),
            std_average_rank=("Average_Rank", "std"),
            n_splits_found=("Average_Rank", "count")
        )
        .sort_values(["mean_average_rank", "feature"], ascending=[True, True])
        .reset_index(drop=True)
    )
    return all_rank_df, rank_summary


def evaluate_stable_pool_across_splits(global_filtered_df, membership_df, ordered_stable_features, subset_sizes,
                                       target_col="logBCF", cv_folds=10):
    fold_rows = []
    summary_rows = []
    split_cols = [c for c in membership_df.columns if c.startswith("split_")]

    for col in split_cols:
        m = re.match(r"split_(\d+)_seed_(\d+)", col)
        if not m:
            continue
        split_id = int(m.group(1))
        seed = int(m.group(2))

        train_smiles = membership_df.loc[membership_df[col] == "train", "SMILES"].tolist()
        train_df = global_filtered_df[global_filtered_df["SMILES"].isin(train_smiles)].copy().reset_index(drop=True)

        valid_subset_sizes = [n for n in subset_sizes if n <= len(ordered_stable_features)]
        if len(valid_subset_sizes) == 0:
            continue

        fold_df, summary_df = evaluate_topn_cv(
            train_df=pd.concat([train_df[["SMILES", target_col]], train_df[ordered_stable_features]], axis=1),
            ordered_features=ordered_stable_features,
            subset_sizes=valid_subset_sizes,
            target_col=target_col,
            cv_folds=cv_folds,
            random_state=seed
        )

        fold_df["split_id"] = split_id
        fold_df["seed"] = seed
        summary_df["split_id"] = split_id
        summary_df["seed"] = seed

        fold_rows.append(fold_df)
        summary_rows.append(summary_df)

    all_fold_df = pd.concat(fold_rows, axis=0, ignore_index=True)
    all_summary_df = pd.concat(summary_rows, axis=0, ignore_index=True)

    summary_across_splits = (
        all_summary_df.groupby("n_features", as_index=False)
        .agg(
            mean_of_split_mean_r2=("mean_r2", "mean"),
            std_of_split_mean_r2=("mean_r2", "std"),
            mean_of_split_mean_rmse=("mean_rmse", "mean"),
            std_of_split_mean_rmse=("mean_rmse", "std")
        )
        .sort_values("n_features")
        .reset_index(drop=True)
    )

    summary_across_all_folds = (
        all_fold_df.groupby("n_features", as_index=False)
        .agg(
            mean_r2_all_folds=("r2", "mean"),
            std_r2_all_folds=("r2", "std"),
            mean_rmse_all_folds=("rmse", "mean"),
            std_rmse_all_folds=("rmse", "std")
        )
        .sort_values("n_features")
        .reset_index(drop=True)
    )

    summary_final = summary_across_splits.merge(
        summary_across_all_folds, on="n_features", how="left"
    )

    return all_fold_df, all_summary_df, summary_final


# =========================================================
# 主程序
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--id_col", type=str, default="SMILES")
    parser.add_argument("--target_col", type=str, default="logBCF")
    parser.add_argument("--std_thresh", type=float, default=1e-4)
    parser.add_argument("--corr_thresh", type=float, default=0.95)
    parser.add_argument("--pearson_thresh", type=float, default=0.3)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--cv_folds", type=int, default=10)
    # 这次直接限定 5~12
    parser.add_argument("--min_n", type=int, default=5)
    parser.add_argument("--max_n", type=int, default=12)
    # 与最佳R²差值 <= delta_r2 时，选更少特征
    parser.add_argument("--delta_r2", type=float, default=0.03)
    # 稳定特征池阈值
    parser.add_argument("--stable_min_freq", type=int, default=1)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
    )

    args = parser.parse_args()
    ensure_dir(args.outdir)

    subset_sizes = list(range(args.min_n, args.max_n + 1))

    # -----------------------------------------------------
    # 读原始数据
    # -----------------------------------------------------
    raw_df = pd.read_csv(args.input)
    raw_df = raw_df.dropna(subset=[args.id_col, args.target_col]).reset_index(drop=True)

    # -----------------------------------------------------
    # 第0步：全数据无监督基础过滤
    # -----------------------------------------------------
    global_filtered_df, global_stats = global_unsupervised_filter(
        raw_df,
        id_col=args.id_col,
        target_col=args.target_col,
        std_thresh=args.std_thresh,
        corr_thresh=args.corr_thresh
    )

    global_filtered_path = os.path.join(args.outdir, "global_unsupervised_filtered.csv")
    global_filtered_df.to_csv(global_filtered_path, index=False, encoding="utf-8-sig")

    with open(os.path.join(args.outdir, "global_unsupervised_filter_stats.json"), "w", encoding="utf-8") as f:
        json.dump(global_stats, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------
    # 外部10次 8:2 分层划分
    # -----------------------------------------------------
    strat_bins = make_strat_bins(global_filtered_df[args.target_col], n_bins=args.n_bins)
    feature_counter = Counter()
    membership_df = global_filtered_df[[args.id_col, args.target_col]].copy()
    split_summary_rows = []

    for i, seed in enumerate(args.seeds[:args.n_splits], start=1):
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=seed
        )
        train_idx, test_idx = next(splitter.split(global_filtered_df, strat_bins))

        train_df = global_filtered_df.iloc[train_idx].reset_index(drop=True)
        test_df = global_filtered_df.iloc[test_idx].reset_index(drop=True)

        split_dir = os.path.join(args.outdir, f"split_{i:02d}_seed_{seed}")
        ensure_dir(split_dir)

        result = run_one_split(
            train_df=train_df,
            test_df=test_df,
            id_col=args.id_col,
            target_col=args.target_col,
            pearson_thresh=args.pearson_thresh,
            subset_sizes=subset_sizes,
            cv_folds=args.cv_folds,
            random_state=seed,
            delta_r2=args.delta_r2
        )

        final_features = result["final_features"]
        feature_counter.update(final_features)

        # 保存单个 split 的所有结果
        result["combined_ranking"].to_csv(
            os.path.join(split_dir, "combined_feature_ranking.csv"),
            index=True, encoding="utf-8-sig"
        )
        result["cv_fold_results"].to_csv(
            os.path.join(split_dir, "cv_fold_scores.csv"),
            index=False, encoding="utf-8-sig"
        )
        result["cv_summary"].to_csv(
            os.path.join(split_dir, "cv_summary.csv"),
            index=False, encoding="utf-8-sig"
        )
        with open(os.path.join(split_dir, "recommended_n.json"), "w", encoding="utf-8") as f:
            json.dump(result["recommended"], f, indent=2, ensure_ascii=False)
        with open(os.path.join(split_dir, "final_selected_features.txt"), "w", encoding="utf-8") as f:
            for feat in final_features:
                f.write(feat + "\n")
        result["train_final"].to_csv(
            os.path.join(split_dir, "train_final_selected.csv"),
            index=False, encoding="utf-8-sig"
        )
        result["test_final"].to_csv(
            os.path.join(split_dir, "test_final_selected.csv"),
            index=False, encoding="utf-8-sig"
        )

        membership = pd.Series(index=global_filtered_df.index, dtype="object")
        membership.iloc[train_idx] = "train"
        membership.iloc[test_idx] = "test"
        membership_df[f"split_{i:02d}_seed_{seed}"] = membership.values

        split_summary_rows.append({
            "split_id": i,
            "seed": seed,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_features_after_global_unsupervised": int(global_filtered_df.shape[1] - 2),
            "n_features_after_train_pearson": int(len(result["pearson_kept"])),
            "n_final_selected_features": int(len(final_features)),
            "recommended_n": int(result["recommended"]["n_features"]),
            "best_r2_in_split": float(result["recommended"]["best_r2"]),
            "delta_r2": float(result["recommended"]["delta_r2"])
        })

        print(
            f"[split {i:02d} | seed={seed}] "
            f"train={len(train_df)}, test={len(test_df)}, "
            f"pearson_kept={len(result['pearson_kept'])}, "
            f"recommended_n={len(final_features)}"
        )

    # -----------------------------------------------------
    # 汇总 phase 1 结果
    # -----------------------------------------------------
    pd.DataFrame(split_summary_rows).to_csv(
        os.path.join(args.outdir, "all_split_summary.csv"),
        index=False, encoding="utf-8-sig"
    )
    membership_df.to_csv(
        os.path.join(args.outdir, "split_membership_summary.csv"),
        index=False, encoding="utf-8-sig"
    )

    freq_df = pd.DataFrame({
        "feature": list(feature_counter.keys()),
        "frequency": list(feature_counter.values())
    }).sort_values(["frequency", "feature"], ascending=[False, True]).reset_index(drop=True)
    freq_df.to_csv(
        os.path.join(args.outdir, "final_feature_frequency_across_splits.csv"),
        index=False, encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # phase 2：稳定特征池再精炼
    # -----------------------------------------------------
    refine_dir = os.path.join(args.outdir, "stable_pool_refinement")
    ensure_dir(refine_dir)

    stable_df = freq_df[freq_df["frequency"] >= args.stable_min_freq].copy()
    stable_df.to_csv(
        os.path.join(refine_dir, "stable_pool_by_frequency.csv"),
        index=False, encoding="utf-8-sig"
    )

    if len(stable_df) >= args.min_n:
        stable_features = stable_df["feature"].tolist()

        # 汇总稳定特征跨 split 的 Average_Rank
        all_rank_df, rank_summary = collect_rank_summary(args.outdir, stable_features)
        all_rank_df.to_csv(
            os.path.join(refine_dir, "stable_pool_all_split_average_rank.csv"),
            index=False, encoding="utf-8-sig"
        )
        rank_summary.to_csv(
            os.path.join(refine_dir, "stable_pool_rank_summary.csv"),
            index=False, encoding="utf-8-sig"
        )

        ordered_stable_features = rank_summary["feature"].tolist()

        # 只比较 top5~top12
        stable_subset_sizes = [n for n in subset_sizes if n <= len(ordered_stable_features)]

        all_fold_df, all_summary_df, summary_final = evaluate_stable_pool_across_splits(
            global_filtered_df=global_filtered_df,
            membership_df=membership_df,
            ordered_stable_features=ordered_stable_features,
            subset_sizes=stable_subset_sizes,
            target_col=args.target_col,
            cv_folds=args.cv_folds
        )

        all_fold_df.to_csv(
            os.path.join(refine_dir, "all_splits_cv_fold_scores_topN.csv"),
            index=False, encoding="utf-8-sig"
        )
        all_summary_df.to_csv(
            os.path.join(refine_dir, "all_splits_cv_summary_topN.csv"),
            index=False, encoding="utf-8-sig"
        )
        summary_final.to_csv(
            os.path.join(refine_dir, "cv_summary_across_splits_topN.csv"),
            index=False, encoding="utf-8-sig"
        )

        recommended_final = choose_compact_n(
            summary_final.rename(columns={"mean_of_split_mean_r2": "mean_r2"}),
            delta_r2=args.delta_r2
        )
        with open(os.path.join(refine_dir, "recommended_n.json"), "w", encoding="utf-8") as f:
            json.dump(recommended_final, f, indent=2, ensure_ascii=False)

        final_n = int(recommended_final["n_features"])
        final_descriptors = ordered_stable_features[:final_n]

        pd.DataFrame({
            "rank": range(1, len(ordered_stable_features) + 1),
            "feature": ordered_stable_features
        }).to_csv(
            os.path.join(refine_dir, "stable_pool_ordered_features.csv"),
            index=False, encoding="utf-8-sig"
        )

        pd.DataFrame({
            "rank": range(1, len(final_descriptors) + 1),
            "feature": final_descriptors
        }).to_csv(
            os.path.join(refine_dir, "final_selected_descriptors.csv"),
            index=False, encoding="utf-8-sig"
        )

        print("\n===== Stable pool refinement done =====")
        print(f"Stable pool size (freq >= {args.stable_min_freq}): {len(stable_features)}")
        print(f"Recommended final n: {final_n}")

    else:
        print("\n===== Stable pool refinement skipped =====")
        print(f"因为 frequency >= {args.stable_min_freq} 的特征不足 {args.min_n} 个。")

    print("\nDone.")
    print(f"Results saved to: {args.outdir}")


if __name__ == "__main__":
    main()