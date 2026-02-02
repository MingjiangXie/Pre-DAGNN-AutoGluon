# -*- coding: utf-8 -*-
"""
从output2.csv中的预计算描述符进行特征筛选
使用以下方法从105个特征筛选到约10个最优特征：
- 随机森林特征重要性
- RFE递归特征消除
- 互信息分析
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')


def load_data(csv_path):
    """从CSV加载预计算的描述符"""
    df = pd.read_csv(csv_path)
    
    # 目标列是最后一列(logBCF)
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]
    
    print(f"数据加载完成: {len(df)}个样本, {len(feature_cols)}个特征")
    print(f"目标列: {target_col}")
    
    X = df[feature_cols].values
    y = df[target_col].values
    feature_names = list(feature_cols)
    
    # 检查缺失值
    n_missing = np.isnan(X).sum()
    if n_missing > 0:
        print(f"警告: 发现{n_missing}个缺失值，使用列中位数填充")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
    
    return X, y, feature_names


def compute_rf_importance(X, y, feature_names, n_estimators=200, random_state=42):
    """计算随机森林特征重要性"""
    print("\n" + "="*70)
    print("步骤1: 随机森林特征重要性")
    print("="*70)
    
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    
    rf.fit(X, y)
    importances = rf.feature_importances_
    
    # 创建重要性得分DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"随机森林训练完成 (R² = {rf.score(X, y):.4f})")
    print(f"\n重要性排名前20的特征:")
    for i, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']:30s} : {row['importance']:.6f}")
    
    return importance_df


def compute_mutual_info(X, y, feature_names, random_state=42):
    """计算互信息得分"""
    print("\n" + "="*70)
    print("步骤2: 互信息分析")
    print("="*70)
    
    mi_scores = mutual_info_regression(X, y, random_state=random_state, n_neighbors=5)
    
    mi_df = pd.DataFrame({
        'feature': feature_names,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    print(f"\n互信息得分排名前20的特征:")
    for i, row in mi_df.head(20).iterrows():
        print(f"  {row['feature']:30s} : {row['mi_score']:.6f}")
    
    return mi_df


def compute_pearson_correlation(X, y, feature_names):
    """计算与目标变量的皮尔逊相关系数"""
    print("\n" + "="*70)
    print("步骤3: 皮尔逊相关性分析")
    print("="*70)
    
    correlations = []
    for i, fname in enumerate(feature_names):
        corr, pval = pearsonr(X[:, i], y)
        correlations.append({
            'feature': fname,
            'pearson_r': corr,
            'abs_pearson_r': abs(corr),
            'p_value': pval
        })
    
    corr_df = pd.DataFrame(correlations).sort_values('abs_pearson_r', ascending=False)
    
    print(f"\n绝对相关系数排名前20的特征:")
    for i, row in corr_df.head(20).iterrows():
        print(f"  {row['feature']:30s} : r={row['pearson_r']:7.4f}, p={row['p_value']:.2e}")
    
    return corr_df


def perform_rfe(X, y, feature_names, n_features_to_select=10, step=5, random_state=42):
    """执行递归特征消除"""
    print("\n" + "="*70)
    print(f"步骤4: RFE递归特征消除 -> 选择{n_features_to_select}个特征")
    print("="*70)
    
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=random_state,
        n_jobs=-1
    )
    
    rfe = RFE(
        estimator=rf,
        n_features_to_select=n_features_to_select,
        step=step,
        verbose=0
    )
    
    print("正在运行RFE (可能需要几分钟)...")
    rfe.fit(X, y)
    
    # 获取选中的特征
    selected_mask = rfe.support_
    selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
    
    # 获取排名
    ranking_df = pd.DataFrame({
        'feature': feature_names,
        'rfe_ranking': rfe.ranking_,
        'selected': selected_mask
    }).sort_values('rfe_ranking')
    
    print(f"\nRFE完成。选中{len(selected_features)}个特征:")
    for i, fname in enumerate(selected_features, 1):
        print(f"  {i:2d}. {fname}")
    
    return selected_features, ranking_df


def combine_rankings(rf_importance_df, mi_df, corr_df, ranking_df, top_k=30):
    """综合所有排名方法并计算聚合得分"""
    print("\n" + "="*70)
    print(f"步骤5: 综合排名 (每种方法取前{top_k}名)")
    print("="*70)
    
    # 归一化得分到[0, 1]
    def normalize_scores(df, col):
        scores = df[col].values
        min_val, max_val = scores.min(), scores.max()
        if max_val > min_val:
            return (scores - min_val) / (max_val - min_val)
        else:
            return scores
    
    # 从每种方法获取前k个特征
    top_rf = set(rf_importance_df.head(top_k)['feature'].tolist())
    top_mi = set(mi_df.head(top_k)['feature'].tolist())
    top_corr = set(corr_df.head(top_k)['feature'].tolist())
    
    # 合并所有特征
    all_features = set(rf_importance_df['feature'].tolist())
    
    # 计算聚合得分
    results = []
    for feat in all_features:
        # 获取排名
        rf_rank = rf_importance_df[rf_importance_df['feature'] == feat].index[0] + 1
        mi_rank = mi_df[mi_df['feature'] == feat].index[0] + 1
        corr_rank = corr_df[corr_df['feature'] == feat].index[0] + 1
        rfe_rank = ranking_df[ranking_df['feature'] == feat]['rfe_ranking'].values[0]
        
        # 平均排名(越小越好)
        avg_rank = (rf_rank + mi_rank + corr_rank + rfe_rank) / 4.0
        
        # 统计在前k名中出现的次数
        top_count = sum([feat in top_rf, feat in top_mi, feat in top_corr])
        
        results.append({
            'feature': feat,
            'rf_rank': rf_rank,
            'mi_rank': mi_rank,
            'corr_rank': corr_rank,
            'rfe_rank': rfe_rank,
            'avg_rank': avg_rank,
            'top_count': top_count
        })
    
    combined_df = pd.DataFrame(results).sort_values(['top_count', 'avg_rank'], ascending=[False, True])
    
    print("\n综合排名前20的特征:")
    print(f"{'排名':<6} {'特征':<30} {'RF':<6} {'MI':<6} {'相关':<6} {'RFE':<6} {'平均':<8} {'前K':<4}")
    print("-" * 74)
    for i, row in combined_df.head(20).iterrows():
        print(f"{i+1:<6d} {row['feature']:<30s} {row['rf_rank']:<6d} {row['mi_rank']:<6d} "
              f"{row['corr_rank']:<6d} {row['rfe_rank']:<6d} {row['avg_rank']:<8.2f} {row['top_count']:<4d}")
    
    return combined_df


def validate_feature_subsets(X, y, feature_names, combined_df, max_features=15):
    """使用交叉验证验证不同特征子集大小"""
    print("\n" + "="*70)
    print("步骤6: 验证特征子集大小 (5折交叉验证)")
    print("="*70)
    
    results = []
    
    for n_features in range(5, max_features + 1):
        selected = combined_df.head(n_features)['feature'].tolist()
        selected_indices = [feature_names.index(f) for f in selected]
        X_subset = X[:, selected_indices]
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
        cv_scores = cross_val_score(rf, X_subset, y, cv=5, scoring='r2', n_jobs=-1)
        
        results.append({
            'n_features': n_features,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        })
        
        print(f"  特征数={n_features:2d}: R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    results_df = pd.DataFrame(results)
    
    # 找到最优数量
    best_idx = results_df['cv_r2_mean'].idxmax()
    best_n = results_df.iloc[best_idx]['n_features']
    best_r2 = results_df.iloc[best_idx]['cv_r2_mean']
    
    print(f"\n最优特征数量: {best_n} (R² = {best_r2:.4f})")
    
    return results_df, best_n


def plot_results(rf_importance_df, mi_df, corr_df, combined_df, validation_df, output_dir):
    """生成可视化图表"""
    print("\n" + "="*70)
    print("步骤7: 生成可视化图表")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 图1: 随机森林重要性前20
    plt.figure(figsize=(10, 8))
    top20_rf = rf_importance_df.head(20)
    plt.barh(range(20), top20_rf['importance'].values[::-1])
    plt.yticks(range(20), top20_rf['feature'].values[::-1])
    plt.xlabel('Importance Score')
    plt.title('Random Forest Feature Importance Top 20')
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, 'rf_importance_top20.png')
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {plot1_path}")
    
    # 图2: 综合排名前20
    plt.figure(figsize=(10, 8))
    top20_combined = combined_df.head(20)
    plt.barh(range(20), 1.0 / top20_combined['avg_rank'].values[::-1])
    plt.yticks(range(20), top20_combined['feature'].values[::-1])
    plt.xlabel('Score (1/Average Rank)')
    plt.title('Combined Ranking Top 20')
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, 'combined_ranking_top20.png')
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {plot2_path}")
    
    # 图3: 验证曲线(特征数量 vs R²)
    plt.figure(figsize=(10, 6))
    plt.errorbar(validation_df['n_features'], validation_df['cv_r2_mean'], 
                 yerr=validation_df['cv_r2_std'], marker='o', capsize=5)
    plt.xlabel('Number of Features')
    plt.ylabel('Cross-Validation R²')
    plt.title('Number of Features vs Model Performance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, 'validation_curve.png')
    plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {plot3_path}")
    
    print(f"\n所有图表已保存到: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='从预计算描述符中进行特征筛选')
    parser.add_argument('--csv', type=str, default='output2.csv',
                        help='包含预计算描述符的CSV文件路径')
    parser.add_argument('--n-features', type=int, default=10,
                        help='目标特征数量 (默认: 10)')
    parser.add_argument('--rfe-step', type=int, default=5,
                        help='RFE步长 (默认: 5)')
    parser.add_argument('--top-k', type=int, default=30,
                        help='每种方法合并的前k个特征 (默认: 30)')
    parser.add_argument('--max-validate', type=int, default=15,
                        help='验证的最大特征数量 (默认: 15)')
    parser.add_argument('--output-dir', type=str, default='feature_selection_final',
                        help='输出目录 (默认: feature_selection_final)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("从预计算描述符进行特征筛选")
    print("="*70)
    print(f"输入CSV: {args.csv}")
    print(f"目标特征数: {args.n_features}")
    print(f"输出目录: {args.output_dir}")
    print(f"随机种子: {args.random_seed}")
    
    # 加载数据
    X, y, feature_names = load_data(args.csv)
    
    # 步骤1: 随机森林重要性
    rf_importance_df = compute_rf_importance(X, y, feature_names, random_state=args.random_seed)
    
    # 步骤2: 互信息
    mi_df = compute_mutual_info(X, y, feature_names, random_state=args.random_seed)
    
    # 步骤3: 皮尔逊相关性
    corr_df = compute_pearson_correlation(X, y, feature_names)
    
    # 步骤4: RFE
    selected_features, ranking_df = perform_rfe(
        X, y, feature_names,
        n_features_to_select=args.n_features,
        step=args.rfe_step,
        random_state=args.random_seed
    )
    
    # 步骤5: 综合排名
    combined_df = combine_rankings(rf_importance_df, mi_df, corr_df, ranking_df, top_k=args.top_k)
    
    # 步骤6: 验证特征子集
    validation_df, best_n = validate_feature_subsets(
        X, y, feature_names, combined_df, max_features=args.max_validate
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存结果
    print("\n" + "="*70)
    print("保存结果")
    print("="*70)
    
    # 保存选中的特征(使用验证得到的最优数量)
    final_selected = combined_df.head(int(best_n))['feature'].tolist()
    output_json = os.path.join(args.output_dir, 'selected_features_final.json')
    with open(output_json, 'w') as f:
        json.dump(final_selected, f, indent=2)
    print(f"  已选特征 ({len(final_selected)}个): {output_json}")
    
    # 保存详细结果
    rf_importance_df.to_csv(os.path.join(args.output_dir, 'rf_importance.csv'), index=False)
    mi_df.to_csv(os.path.join(args.output_dir, 'mi_scores.csv'), index=False)
    corr_df.to_csv(os.path.join(args.output_dir, 'pearson_correlation.csv'), index=False)
    ranking_df.to_csv(os.path.join(args.output_dir, 'rfe_ranking.csv'), index=False)
    combined_df.to_csv(os.path.join(args.output_dir, 'combined_ranking.csv'), index=False)
    validation_df.to_csv(os.path.join(args.output_dir, 'validation_results.csv'), index=False)
    print(f"  详细结果已保存到: {args.output_dir}/")
    
    # 生成图表
    plot_results(rf_importance_df, mi_df, corr_df, combined_df, validation_df, args.output_dir)
    
    # 最终总结
    print("\n" + "="*70)
    print("最终总结")
    print("="*70)
    print(f"初始特征数: {len(feature_names)}")
    print(f"选中特征数: {len(final_selected)}")
    print(f"预估交叉验证R²: {validation_df[validation_df['n_features']==int(best_n)]['cv_r2_mean'].values[0]:.4f}")
    print(f"\n最终选中的特征:")
    for i, feat in enumerate(final_selected, 1):
        print(f"  {i:2d}. {feat}")
    print(f"\n结果已保存到: {args.output_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()
