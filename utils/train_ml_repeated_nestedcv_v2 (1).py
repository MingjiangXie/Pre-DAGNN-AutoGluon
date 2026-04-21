#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

# optional models
HAS_XGB = True
HAS_LGBM = True

try:
    from xgboost import XGBRegressor
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
except Exception:
    HAS_LGBM = False


# =========================================================
# utils
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred)
    }


def make_strat_bins(y, n_bins=10):
    """
    For continuous targets, create bins for stratified splitting.
    Prefer qcut; fallback to cut; final fallback to one bin.
    """
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


def format_mean_std(mean_val, std_val, ndigits=3):
    return f"{mean_val:.{ndigits}f} ± {std_val:.{ndigits}f}"


def build_model_spaces(random_state=42):
    model_spaces = {}

    # -------------------------
    # linear models
    # -------------------------
    model_spaces["LinearRegression"] = {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "param_grid": {}
    }

    model_spaces["Ridge"] = {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(random_state=random_state))
        ]),
        "param_grid": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    }

    model_spaces["Lasso"] = {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Lasso(random_state=random_state, max_iter=30000))
        ]),
        "param_grid": {
            "model__alpha": [0.0001, 0.0005, 0.001, 0.01, 0.1, 1.0]
        }
    }

    model_spaces["ElasticNet"] = {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(random_state=random_state, max_iter=30000))
        ]),
        "param_grid": {
            "model__alpha": [0.0001, 0.0005, 0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    }

    model_spaces["SVR"] = {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVR())
        ]),
        "param_grid": {
            "model__kernel": ["rbf", "linear"],
            "model__C": [1, 5, 10, 50, 100],
            "model__epsilon": [0.01, 0.05, 0.1, 0.2],
            "model__gamma": ["scale", "auto"]
        }
    }

    # -------------------------
    # tree models
    # -------------------------
    model_spaces["RandomForest"] = {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                random_state=random_state,
                n_jobs=-1
            ))
        ]),
        "param_grid": {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 6, 10, 15],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.8, 1.0]
        }
    }

    model_spaces["ExtraTrees"] = {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                random_state=random_state,
                n_jobs=-1
            ))
        ]),
        "param_grid": {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 6, 10, 15],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.8, 1.0]
        }
    }

    model_spaces["GradientBoosting"] = {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(random_state=random_state))
        ]),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0]
        }
    }

    model_spaces["MLP"] = {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                random_state=random_state,
                max_iter=3000,
                early_stopping=True
            ))
        ]),
        "param_grid": {
            "model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
            "model__alpha": [1e-5, 1e-4, 1e-3],
            "model__learning_rate_init": [1e-3, 5e-4]
        }
    }

    if HAS_XGB:
        model_spaces["XGBoost"] = {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBRegressor(
                    random_state=random_state,
                    n_jobs=-1,
                    objective="reg:squarederror",
                    eval_metric="rmse"
                ))
            ]),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 4, 6],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0]
            }
        }

    if HAS_LGBM:
        model_spaces["LightGBM"] = {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LGBMRegressor(
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1
                ))
            ]),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__num_leaves": [15, 31, 63],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0]
            }
        }

    return model_spaces


def get_inner_fold_indices(X_train, y_train, n_bins, n_splits, random_state):
    inner_bins = make_strat_bins(y_train, n_bins=n_bins)
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    fold_indices = list(skf.split(X_train, inner_bins))
    return fold_indices, inner_bins


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
    parser = argparse.ArgumentParser(
        description="Repeated external split + inner 10-fold CV ML training on selected descriptors"
    )
    parser.add_argument("--input", type=str, required=True, help="Input CSV")
    parser.add_argument("--outdir", type=str, default="ml_repeated_nestedcv_results")
    parser.add_argument("--id_col", type=str, default="SMILES")
    parser.add_argument("--target_col", type=str, default="logBCF")
    parser.add_argument("--feature_cols", nargs="+", default=[
        "TPSA_efficiency", "LOGP99", "SM1_Dz(p)", "Hy", "BLTA96"
    ])
    parser.add_argument("--n_bins", type=int, default=10, help="Bins for stratification")
    parser.add_argument("--test_size", type=float, default=0.2, help="Outer validation fraction")
    parser.add_argument("--cv_folds", type=int, default=10, help="Inner CV folds; keep consistent with feature selection")
    parser.add_argument("--outer_seeds", nargs="+", type=int,
                        default=[42, 52, 62, 72, 82, 92, 102, 112, 122, 132])
    parser.add_argument("--scoring", type=str, default="neg_root_mean_squared_error",
                        choices=["neg_root_mean_squared_error", "r2", "neg_mean_absolute_error"])
    args = parser.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "per_split"))
    ensure_dir(os.path.join(args.outdir, "predictions"))
    ensure_dir(os.path.join(args.outdir, "cv_details"))

    # -------------------------
    # load data
    # -------------------------
    df = pd.read_csv(args.input)
    needed_cols = [args.id_col] + args.feature_cols + [args.target_col]
    missing_cols = [c for c in needed_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input CSV: {missing_cols}")

    df = df[needed_cols].copy()
    df = df.dropna(subset=[args.id_col, args.target_col]).reset_index(drop=True)
    df.insert(0, "row_id", np.arange(len(df)))

    X_all = df[args.feature_cols].copy()
    y_all = df[args.target_col].copy()
    all_bins = make_strat_bins(y_all, n_bins=args.n_bins)

    run_info = {
        "input": args.input,
        "outdir": args.outdir,
        "n_samples": int(len(df)),
        "feature_cols": args.feature_cols,
        "target_col": args.target_col,
        "id_col": args.id_col,
        "n_bins": args.n_bins,
        "test_size": args.test_size,
        "cv_folds": args.cv_folds,
        "outer_seeds": args.outer_seeds,
        "scoring_for_gridsearch": args.scoring,
        "note": "tra = outer training set, val = outer hold-out validation set, cv = inner cross-validation mean on outer training set"
    }
    with open(os.path.join(args.outdir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    model_spaces = build_model_spaces(random_state=42)

    all_split_metrics = []
    all_best_params = []
    all_predictions = []
    all_split_membership = []
    all_inner_fold_metrics = []

    # -------------------------
    # outer repeated split
    # -------------------------
    for split_id, seed in enumerate(args.outer_seeds, start=1):
        print(f"\n========== Outer split {split_id:02d} | seed={seed} ==========")

        outer_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=args.test_size,
            random_state=seed
        )
        train_idx, val_idx = next(outer_splitter.split(X_all, all_bins))

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        X_train = train_df[args.feature_cols].copy()
        y_train = train_df[args.target_col].copy()
        X_val = val_df[args.feature_cols].copy()
        y_val = val_df[args.target_col].copy()

        fold_indices, inner_bins = get_inner_fold_indices(
            X_train, y_train, n_bins=args.n_bins, n_splits=args.cv_folds, random_state=seed
        )

        # save split membership
        train_membership = train_df[[args.id_col, "row_id", args.target_col]].copy()
        train_membership["split_id"] = split_id
        train_membership["seed"] = seed
        train_membership["set"] = "train"

        val_membership = val_df[[args.id_col, "row_id", args.target_col]].copy()
        val_membership["split_id"] = split_id
        val_membership["seed"] = seed
        val_membership["set"] = "val"

        all_split_membership.append(train_membership)
        all_split_membership.append(val_membership)

        # save inner fold assignment
        fold_assign_rows = []
        for fold_id, (_, va_idx) in enumerate(fold_indices, start=1):
            tmp = train_df.iloc[va_idx][[args.id_col, "row_id", args.target_col]].copy()
            tmp["split_id"] = split_id
            tmp["seed"] = seed
            tmp["inner_fold"] = fold_id
            fold_assign_rows.append(tmp)
        fold_assign_df = pd.concat(fold_assign_rows, axis=0, ignore_index=True)
        fold_assign_df.to_csv(
            os.path.join(args.outdir, "cv_details", f"split_{split_id:02d}_seed_{seed}_inner_fold_assignment.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        for model_name, spec in model_spaces.items():
            print(f"  -> {model_name}")

            pipeline = clone(spec["pipeline"])
            param_grid = deepcopy(spec["param_grid"])

            # -------------------------
            # inner CV tuning
            # -------------------------
            if len(param_grid) == 0:
                # no hyperparameter tuning
                best_estimator = clone(pipeline)
                best_estimator.fit(X_train, y_train)
                best_params = {}

            else:
                grid = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    scoring=args.scoring,
                    cv=fold_indices,
                    n_jobs=-1,
                    refit=True,
                    return_train_score=False
                )
                grid.fit(X_train, y_train)
                best_estimator = grid.best_estimator_
                best_params = grid.best_params_

            # -------------------------
            # inner CV metrics with best estimator
            # -------------------------
            fold_metric_rows = []
            for fold_id, (tr_idx, va_idx) in enumerate(fold_indices, start=1):
                X_tr = X_train.iloc[tr_idx]
                y_tr = y_train.iloc[tr_idx]
                X_va = X_train.iloc[va_idx]
                y_va = y_train.iloc[va_idx]

                est = clone(best_estimator)
                est.fit(X_tr, y_tr)

                pred_tr = est.predict(X_tr)
                pred_va = est.predict(X_va)

                tr_m = regression_metrics(y_tr, pred_tr)
                va_m = regression_metrics(y_va, pred_va)

                fold_metric_rows.append({
                    "split_id": split_id,
                    "seed": seed,
                    "Model": model_name,
                    "inner_fold": fold_id,
                    "n_fold_train": len(X_tr),
                    "n_fold_val": len(X_va),
                    "R2_fold_train": tr_m["R2"],
                    "RMSE_fold_train": tr_m["RMSE"],
                    "MAE_fold_train": tr_m["MAE"],
                    "R2_fold_val": va_m["R2"],
                    "RMSE_fold_val": va_m["RMSE"],
                    "MAE_fold_val": va_m["MAE"]
                })

            fold_df = pd.DataFrame(fold_metric_rows)
            all_inner_fold_metrics.append(fold_df)

            # inner CV summary on validation folds
            cv_r2 = fold_df["R2_fold_val"].mean()
            cv_rmse = fold_df["RMSE_fold_val"].mean()
            cv_mae = fold_df["MAE_fold_val"].mean()

            cv_r2_std = fold_df["R2_fold_val"].std(ddof=1) if len(fold_df) > 1 else 0.0
            cv_rmse_std = fold_df["RMSE_fold_val"].std(ddof=1) if len(fold_df) > 1 else 0.0
            cv_mae_std = fold_df["MAE_fold_val"].std(ddof=1) if len(fold_df) > 1 else 0.0

            # save per-model fold metrics
            fold_df.to_csv(
                os.path.join(args.outdir, "cv_details", f"split_{split_id:02d}_seed_{seed}_{model_name}_inner10fold_metrics.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            # -------------------------
            # fit best estimator on full outer training set
            # -------------------------
            best_estimator.fit(X_train, y_train)

            pred_train = best_estimator.predict(X_train)
            pred_val = best_estimator.predict(X_val)

            train_metrics = regression_metrics(y_train, pred_train)
            val_metrics = regression_metrics(y_val, pred_val)
            delta_r2 = train_metrics["R2"] - val_metrics["R2"]

            # -------------------------
            # predictions for plotting
            # -------------------------
            pred_train_df = train_df[[args.id_col, "row_id", args.target_col]].copy()
            pred_train_df["split_id"] = split_id
            pred_train_df["seed"] = seed
            pred_train_df["Model"] = model_name
            pred_train_df["set"] = "train"
            pred_train_df["y_true"] = y_train.values
            pred_train_df["y_pred"] = pred_train
            pred_train_df["residual"] = pred_train_df["y_true"] - pred_train_df["y_pred"]

            pred_val_df = val_df[[args.id_col, "row_id", args.target_col]].copy()
            pred_val_df["split_id"] = split_id
            pred_val_df["seed"] = seed
            pred_val_df["Model"] = model_name
            pred_val_df["set"] = "val"
            pred_val_df["y_true"] = y_val.values
            pred_val_df["y_pred"] = pred_val
            pred_val_df["residual"] = pred_val_df["y_true"] - pred_val_df["y_pred"]

            all_predictions.append(pred_train_df)
            all_predictions.append(pred_val_df)

            # save per split per model predictions
            pred_one = pd.concat([pred_train_df, pred_val_df], axis=0, ignore_index=True)
            pred_one.to_csv(
                os.path.join(args.outdir, "predictions", f"{model_name}_split_{split_id:02d}_seed_{seed}_predictions.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            # -------------------------
            # metrics table row
            # -------------------------
            metric_row = {
                "split_id": split_id,
                "seed": seed,
                "Model": model_name,
                "n_train": len(train_df),
                "n_val": len(val_df),

                "R2_tra": train_metrics["R2"],
                "RMSE_tra": train_metrics["RMSE"],
                "MAE_tra": train_metrics["MAE"],

                "R2_val": val_metrics["R2"],
                "RMSE_val": val_metrics["RMSE"],
                "MAE_val": val_metrics["MAE"],

                "R2_cv": cv_r2,
                "RMSE_cv": cv_rmse,
                "MAE_cv": cv_mae,

                "R2_cv_std": cv_r2_std,
                "RMSE_cv_std": cv_rmse_std,
                "MAE_cv_std": cv_mae_std,

                "Delta_R2": delta_r2
            }
            all_split_metrics.append(metric_row)

            # -------------------------
            # NEW: save single-model single-split metrics to per_split
            # -------------------------
            per_split_df = pd.DataFrame([{
                "split_id": split_id,
                "seed": seed,
                "Model": model_name,
                "n_train": len(train_df),
                "n_val": len(val_df),

                "R2_tra": train_metrics["R2"],
                "RMSE_tra": train_metrics["RMSE"],
                "MAE_tra": train_metrics["MAE"],

                "R2_val": val_metrics["R2"],
                "RMSE_val": val_metrics["RMSE"],
                "MAE_val": val_metrics["MAE"],

                "R2_cv": cv_r2,
                "Delta_R2": delta_r2
            }])

            per_split_df.to_csv(
                os.path.join(
                    args.outdir,
                    "per_split",
                    f"{model_name}_split_{split_id:02d}_seed_{seed}_metrics.csv"
                ),
                index=False,
                encoding="utf-8-sig"
            )

            all_best_params.append({
                "split_id": split_id,
                "seed": seed,
                "Model": model_name,
                "best_params_json": json.dumps(best_params, ensure_ascii=False)
            })

    # =========================================================
    # save all results
    # =========================================================
    metrics_df = pd.DataFrame(all_split_metrics)
    metrics_df.to_csv(
        os.path.join(args.outdir, "all_models_all_splits_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    best_params_df = pd.DataFrame(all_best_params)
    best_params_df.to_csv(
        os.path.join(args.outdir, "all_models_best_params.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    split_membership_df = pd.concat(all_split_membership, axis=0, ignore_index=True)
    split_membership_df.to_csv(
        os.path.join(args.outdir, "outer_split_membership.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_predictions_df = pd.concat(all_predictions, axis=0, ignore_index=True)
    all_predictions_df.to_csv(
        os.path.join(args.outdir, "all_models_all_predictions_long.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_inner_fold_df = pd.concat(all_inner_fold_metrics, axis=0, ignore_index=True)
    all_inner_fold_df.to_csv(
        os.path.join(args.outdir, "all_models_all_inner10fold_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # summary by model
    summary_df = summarize_mean_std(metrics_df, group_col="Model")
    summary_df.to_csv(
        os.path.join(args.outdir, "summary_mean_std_by_model.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # direct-use main table
    table_s8 = summary_df[[
        "Model",
        "R2_tra_mean±std",
        "RMSE_tra_mean±std",
        "MAE_tra_mean±std",
        "R2_val_mean±std",
        "RMSE_val_mean±std",
        "MAE_val_mean±std",
        "R2_cv_mean±std",
        "Delta_R2_mean±std"
    ]].copy()

    table_s8.columns = [
        "Model",
        "R2_tra",
        "RMSE_tra",
        "MAE_tra",
        "R2_val",
        "RMSE_val",
        "MAE_val",
        "R2_cv",
        "Delta_R2"
    ]
    table_s8.to_csv(
        os.path.join(args.outdir, "Table_S8_repeated_split_mean_std.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # additional plot-ready files
    # 1) validation-only metrics for boxplots
    val_plot_df = metrics_df[[
        "split_id", "seed", "Model",
        "R2_val", "RMSE_val", "MAE_val"
    ]].copy()
    val_plot_df.to_csv(
        os.path.join(args.outdir, "plot_ready_validation_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 2) cv-only fold-level metrics
    cv_plot_df = all_inner_fold_df[[
        "split_id", "seed", "Model", "inner_fold",
        "R2_fold_val", "RMSE_fold_val", "MAE_fold_val"
    ]].copy()
    cv_plot_df.to_csv(
        os.path.join(args.outdir, "plot_ready_inner10fold_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 3) parity plot data
    parity_df = all_predictions_df[[
        args.id_col, "row_id", "split_id", "seed", "Model", "set", "y_true", "y_pred", "residual"
    ]].copy()
    parity_df.to_csv(
        os.path.join(args.outdir, "plot_ready_parity_and_residuals.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDone.")
    print(f"Results saved to: {args.outdir}")
    print("Key files:")
    print("  - summary_mean_std_by_model.csv")
    print("  - Table_S8_repeated_split_mean_std.csv")
    print("  - all_models_all_splits_metrics.csv")
    print("  - all_models_all_predictions_long.csv")
    print("  - all_models_all_inner10fold_metrics.csv")
    print("  - outer_split_membership.csv")


if __name__ == "__main__":
    main()