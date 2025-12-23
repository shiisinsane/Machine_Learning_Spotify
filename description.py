import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pygam import LinearGAM, s  # 广义可加模型
import os
from feature_process import read_df
from statsmodels.nonparametric.smoothers_lowess import lowess  # 用于非线性拟合曲线
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def split_target(df_train, df_test):
    # 分离X/y
    # 目标变量popularity
    X_train = df_train.drop(columns=['popularity'])
    y_train = df_train['popularity']
    X_test = df_test.drop(columns=['popularity'])
    y_test = df_test['popularity']

    print(f"训练集 X: {X_train.shape}, y: {y_train.shape}")
    print(f"测试集 X: {X_test.shape}, y: {y_test.shape}")
    return X_train, y_train, X_test, y_test

def basic_statistics(X, y):
    """
    基础统计量
    均值、标准差、中位数等
    """
    # 合并特征和目标变量
    data = pd.concat([X, y.rename('popularity')], axis=1)
    stats = data.describe().T[['mean', 'std', '50%', 'min', 'max']]
    stats.columns = ['均值', '标准差', '中位数', '最小值', '最大值']
    print("基础统计量：")
    print(stats.round(4))
    return stats


def visualization_basic(X, y, save_path):
    """
    基础可视化
    直方图、相关性热力图
    """
    plt.rcParams['font.sans-serif'] = ['Arial']
    data = pd.concat([X, y.rename('popularity')], axis=1)

    # 目标变量直方图
    plt.figure(figsize=(8, 4))
    plt.hist(y, bins=50, alpha=0.7, color='grey', edgecolor='black')
    plt.title('Distribution of Spotify Music Popularity')
    plt.xlabel('Popularity (Standardized)')
    plt.ylabel('Count')
    plt.savefig(f"{save_path}/popularity_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 数值类特征相关性热力图
    core_features = [
        'duration_ms', 'acousticness', 'danceability', 'energy',
        'instrumentalness', 'liveness', 'loudness', 'speechiness',
        'tempo', 'valence', 'key', 'singer_num', 'name_len','popularity'
    ]
    corr = data[core_features].corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.xticks(range(len(core_features)), core_features, rotation=45)
    plt.yticks(range(len(core_features)), core_features)
    # 添加相关系数文本
    for i in range(len(core_features)):
        for j in range(len(core_features)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center',
                     color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
    plt.title('Correlation Heatmap of Core Features')
    plt.savefig(f"{save_path}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_distributions(X, save_path, cat_threshold=2):
    """
    绘制所有特征的分布直方图
    - 数值变量：绘制直方图（单张图包含所有数值特征子图），并输出偏度和峰度
    - 分类变量（unique值<=阈值）：绘制柱状图（单张图包含所有分类特征子图）

    参数:
        X: 特征数据DataFrame
        save_path: 图片保存路径
        cat_threshold: 判断分类变量的唯一值数量阈值
    """
    print("\n===== 绘制特征分布直方图 =====")

    # 区分数值变量和分类变量
    numeric_features = []
    categorical_features = []
    for col in X.columns:
        if X[col].nunique() <= cat_threshold:
            categorical_features.append(col)
        else:
            numeric_features.append(col)

    print(f"检测到数值变量 {len(numeric_features)} 个：{numeric_features}")
    print(f"检测到分类变量 {len(categorical_features)} 个：{categorical_features}")

    # 绘制数值变量直方图并计算偏度、峰度
    if numeric_features:
        n = len(numeric_features)
        n_cols = 4
        n_rows = (n + n_cols - 1) // n_cols  # 向上取整
        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        plt.rcParams['font.sans-serif'] = ['Arial']

        # 打印数值变量的偏度和峰度
        print("\n===== 数值变量偏度与峰度 =====")
        for feat in numeric_features:
            skew_val = X[feat].skew()  # 偏度：>0右偏，<0左偏，接近0对称
            kurt_val = X[feat].kurt()  # 峰度：>0尖峰，<0平峰，接近0正态
            print(f"{feat}: 偏度 = {skew_val:.4f}, 峰度 = {kurt_val:.4f}")

        for i, feat in enumerate(numeric_features):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            # 绘制直方图
            ax.hist(X[feat], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            # 添加统计信息
            mean_val = X[feat].mean()
            median_val = X[feat].median()

            ax.axvline(mean_val, color='r', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='g', linestyle='-.', linewidth=1.5, label=f'Median: {median_val:.2f}')

            ax.set_title(
                f'Distribution of {feat}',
                fontsize=10,
                fontweight='bold'
            )
            ax.set_xlabel(feat, fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        # 隐藏多余子图
        for idx in range(n + 1, n_rows * n_cols + 1):
            plt.subplot(n_rows, n_cols, idx).axis('off')

        fig.suptitle('Distributions of Numeric Features', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        save_file = os.path.join(save_path, 'numeric_features_distribution.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n数值变量分布图已保存到：{save_file}")

    # 绘制分类变量柱状图
    if categorical_features:
        n = len(categorical_features)
        n_cols = 4
        n_rows = (n + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        plt.rcParams['font.sans-serif'] = ['Arial']

        for i, feat in enumerate(categorical_features):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            # 计算类别数量
            counts = X[feat].value_counts().sort_index()
            # 绘制柱状图
            ax.bar(counts.index.astype(str), counts.values, alpha=0.7, color='salmon', edgecolor='black')
            # 添加数值标签
            for x, y in zip(counts.index, counts.values):
                ax.text(x, y + 0.5, f'{y}', ha='center', fontsize=7)

            ax.set_title(f'Distribution of {feat}', fontsize=10, fontweight='bold')
            ax.set_xlabel(feat, fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.grid(alpha=0.3, axis='y')

        # 隐藏多余子图
        for idx in range(n + 1, n_rows * n_cols + 1):
            plt.subplot(n_rows, n_cols, idx).axis('off')

        fig.suptitle('Distributions of Categorical Features', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        save_file = os.path.join(save_path, 'categorical_features_distribution.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"分类变量分布图已保存到：{save_file}")

def gam_analysis(X, y, save_path):
    """
    GAM分析
    """
    print("=====GAM分析=====")

    # 数值型特征
    numeric_cols = [
        'duration_ms', 'acousticness', 'danceability', 'energy',
        'instrumentalness', 'liveness', 'loudness', 'speechiness',
        'tempo', 'valence', 'key', 'singer_num', 'name_len'
    ]
    gam_features = [c for c in numeric_cols if c in X.columns]
    X_gam = X[gam_features].values
    y_gam = y.values
    n_features = len(gam_features)

    print(f"GAM分析特征共{n_features}个：{gam_features}")

    # ========== 1. 构建GAM ==========
    terms = s(0, n_splines=10, spline_order=3)
    for i in range(1, n_features):
        terms += s(i, n_splines=10, spline_order=3)

    gam = LinearGAM(terms, fit_intercept=True)
    gam.fit(X_gam, y_gam)

    # ========== 2. 模型指标 ==========
    stats = gam.statistics_

    r2 = stats.get('pseudo_r2', {}).get('explained_deviance', np.nan)
    aic = stats.get('AIC', np.nan)

    print("=" * 50)
    print("GAM模型核心指标")
    print("=" * 50)
    print(f"样本量：{X_gam.shape[0]}")
    print(f"特征数：{n_features}")
    print(f"R²（pseudo）：{r2:.4f}")
    print(f"AIC：{aic:.2f}")
    print("=" * 50)

    # ========== 3. 特征重要性（EDF） ==========

    print("\n特征重要性分析 基于 Partial Dependence 方差")

    feature_importance = {}
    n_grid_points = 100

    for i, feat in enumerate(gam_features):
        XX = gam.generate_X_grid(term=i, n=n_grid_points)
        pdp = gam.partial_dependence(term=i, X=XX).ravel()

        # 使用PD的方差作为重要性
        importance = np.var(pdp)
        feature_importance[feat] = importance

        print(f"  {feat}: PD variance = {importance:.6f}")

    # ========== 4. 部分依赖图 ==========
    print("\n生成部分依赖图...")

    fig = plt.figure(figsize=(22, 18))
    n_grid_points = 100

    for i, feature in enumerate(gam_features):
        ax = plt.subplot(4, 4, i + 1)

        # pyGAM官方PD
        XX = gam.generate_X_grid(term=i, n=n_grid_points)
        pdp, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

        feature_vals = XX[:, i]

        # 主PD曲线
        ax.plot(feature_vals, pdp,
                color='darkblue', linewidth=2, label='Partial Dependence')

        # 置信区间
        ax.fill_between(feature_vals,
                        confi[:, 0], confi[:, 1],
                        color='blue', alpha=0.15, label='95% CI')

        # 直方图（数据分布）
        ax_hist = ax.twinx()
        hist_counts, hist_bins = np.histogram(X_gam[:, i], bins=30)
        hist_height = hist_counts / hist_counts.max() * 0.3
        ax_hist.bar(hist_bins[:-1], hist_height,
                    width=np.diff(hist_bins),
                    alpha=0.3, color='gray', align='edge')
        ax_hist.set_ylim(0, 0.35)
        ax_hist.set_ylabel('Density', fontsize=7)

        # 线性趋势线（用于展示方向性）
        if len(pdp) > 1:
            z = np.polyfit(feature_vals, pdp, 1)
            p = np.poly1d(z)
            ss_tot = np.sum((pdp - pdp.mean()) ** 2)
            ss_res = np.sum((pdp - p(feature_vals)) ** 2)
            r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            ax.plot(feature_vals, p(feature_vals),
                    'r--', linewidth=1.5,
                    label=f'Linear (R²={r2_lin:.3f})')

        # 标签
        ax.set_title(feature, fontsize=10, fontweight='bold')
        ax.set_xlabel(feature, fontsize=8)
        ax.set_ylabel('PD Value', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for idx in range(len(gam_features) + 1, 17):
        plt.subplot(4, 4, idx).axis('off')

    fig.suptitle('Prior Method I: GAM Partial Dependence Plots',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(f"{save_path}/gam_partial_dependence_all_features.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"依赖图已保存到: {save_path}/gam_partial_dependence_all_features.png")

    # ========== 5. 返回结构（与你原来一致） ==========
    results = {
        'model': gam,
        'r2': r2,
        'aic': aic,
        'feature_importance': feature_importance,
        'gam_features': gam_features
    }

    print("\nGAM分析完成！")
    return results


# def plot_feature_popularity_relation(X: pd.DataFrame, y: pd.Series, save_path: str):
#     """
#     绘制所有数值特征与popularity的直接关系图（13个子图，1张图整合）
#     每个子图包含：散点图（原始数据）+ Lowess非线性拟合曲线（趋势）+ 相关系数
#     """
#     print("=====生成特征与流行度直接关系图=====")
#
#     # 定义要分析的特征
#     feature_list = [
#         'duration_ms', 'acousticness', 'danceability', 'energy',
#         'instrumentalness', 'liveness', 'loudness', 'speechiness',
#         'tempo', 'valence', 'key', 'singer_num', 'name_len'
#     ]
#     # 过滤存在的特征
#     feature_list = [feat for feat in feature_list if feat in X.columns]
#     n_features = len(feature_list)
#     print(f"最终分析特征（共{n_features}个）：{feature_list}")
#
#     # 设置画布（4行4列，13个特征，剩余子图隐藏）
#     fig = plt.figure(figsize=(22, 18))
#     plt.rcParams['font.sans-serif'] = ['Arial']
#     plt.rcParams['axes.unicode_minus'] = False
#
#     for i, feat in enumerate(feature_list):
#         ax = plt.subplot(4, 4, i + 1)
#
#         # 获取当前特征数据（避免极端值干扰可视化，限制在99%分位数内）
#         x_data = X[feat].values
#         y_data = y.values
#         # 过滤极端值（仅用于可视化，不改变分析结论）
#         q1 = np.percentile(x_data, 1)
#         q99 = np.percentile(x_data, 99)
#         mask = (x_data >= q1) & (x_data <= q99)
#         x_plot = x_data[mask]
#         y_plot = y_data[mask]
#
#         # 绘制散点图
#         ax.scatter(x_plot, y_plot, alpha=0.1, color='steelblue', s=5, edgecolor='none')
#
#         # 绘制Lowess非线性拟合曲线
#         lowess_result = lowess(y_plot, x_plot, frac=0.1)  # frac=0.1控制平滑度
#         ax.plot(lowess_result[:, 0], lowess_result[:, 1], color='darkred', linewidth=2.5, label='Trend Curve')
#
#         # 计算并显示Pearson相关系数
#         corr = np.corrcoef(x_data, y_data)[0, 1]
#         corr_text = f"Pearson Corr: {corr:.3f}"
#         ax.text(0.05, 0.9, corr_text, transform=ax.transAxes, fontsize=8,
#                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), fontweight='bold')
#
#         # 子图标签与格式
#         ax.set_title(f'Feature: {feat}', fontsize=10, fontweight='bold', pad=8)
#         ax.set_xlabel(feat, fontsize=8)
#         ax.set_ylabel('Popularity (Standardized)', fontsize=8)
#         ax.grid(True, alpha=0.3)
#         ax.legend(loc='best', fontsize=7)
#
#     # 隐藏多余子图
#     for idx in range(n_features + 1, 17):
#         plt.subplot(4, 4, idx).axis('off')
#
#     # 总标题与布局调整
#     fig.suptitle('Relationship Between Numeric Features and Spotify Music Popularity',
#                  fontsize=16, fontweight='bold', y=0.98)
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.94)
#
#     # 保存图片
#     save_path = f"{save_path}/feature_popularity_relation.png"
#     plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
#     print(f"特征与流行度关系图已保存到：{save_path}")



def prior_method_two_analysis(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_grid: int = 100
):
    """
    先验方法二：
    强模型 -> 单变量函数 g_k -> 线性回归
    """

    print("===== Prior Method II: Strong Model --> Additive Approximation =====")
    # === 保证特征顺序一致（XGBoost 必须）===
    feature_order = X_train.columns.tolist()
    X_train = X_train[feature_order]
    X_test = X_test[feature_order]

    # ========== Step 1: 强泛化模型 ==========

    # strong_model = GradientBoostingRegressor(
    #     n_estimators=300,
    #     max_depth=5,
    #     learning_rate=0.05,
    #     subsample=0.8,
    #     random_state=42
    # )
    # strong_model.fit(X_train, y_train)

    strong_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42
    )

    strong_model.fit(X_train, y_train)

    y_pred_strong = strong_model.predict(X_test)
    print(f"[Strong Model] R^2 = {r2_score(y_test, y_pred_strong):.4f}")

    # ========== Step 2: 构造 g_k(x^(k)) ==========
    X_mean = X_train.mean()
    g_functions = {}
    design_train = []
    design_test = []

    for col in X_train.columns:
        # 在特征 col 上构造网格
        grid = np.linspace(
            X_train[col].quantile(0.01),
            X_train[col].quantile(0.99),
            n_grid
        )

        X_tmp = pd.DataFrame(
            np.tile(X_mean.values, (n_grid, 1)),
            columns=X_train.columns
        )
        X_tmp[col] = grid

        g_k = strong_model.predict(X_tmp)
        g_functions[col] = (grid, g_k)

        # === 将 g_k 映射回训练 / 测试 ===
        design_train.append(np.interp(X_train[col], grid, g_k))
        design_test.append(np.interp(X_test[col], grid, g_k))

    Phi_train = np.column_stack(design_train)
    Phi_test = np.column_stack(design_test)

    # ========== Step 3: 线性回归 ==========
    linear_model = LinearRegression()
    linear_model.fit(Phi_train, y_train)

    y_pred_additive = linear_model.predict(Phi_test)

    print(f"[Additive Approximation] R^2 = {r2_score(y_test, y_pred_additive):.4f}")
    print(f"[Additive Approximation] RMSE = {mean_squared_error(y_test, y_pred_additive, squared=False):.4f}")

    return {
        "strong_model": strong_model,
        "linear_additive_model": linear_model,
        "g_functions": g_functions,
        "r2_strong": r2_score(y_test, y_pred_strong),
        "r2_additive": r2_score(y_test, y_pred_additive),
    }

# def plot_prior2_partial_dependence(
#     prior2_results: dict,
#     save_path: str,
#     max_cols: int = 4
# ):
#     """
#     绘制 Prior Method II 的部分依赖图（g_k(x_k)）
#     - 基于强模型 + 固定均值
#     - 数值安全（防 SVD / 常数函数）
#     """
#
#     numeric_features = [
#         'duration_ms', 'acousticness', 'danceability', 'energy',
#         'instrumentalness', 'liveness', 'loudness', 'speechiness',
#         'tempo', 'valence', 'key', 'singer_num', 'name_len'
#     ]
#
#     print("===== 绘制 Prior Method II 的部分依赖图 =====")
#
#     g_functions = prior2_results["g_functions"]
#
#     # 只画数值变量（与你 GAM 保持一致）
#     features = [f for f in numeric_features if f in g_functions]
#     n_features = len(features)
#
#     # === 自动计算行列 ===
#     n_cols = max_cols
#     n_rows = int(np.ceil(n_features / n_cols))
#
#     fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows))
#     plt.rcParams['font.sans-serif'] = ['Arial']
#     plt.rcParams['axes.unicode_minus'] = False
#
#     for i, feat in enumerate(features):
#         ax = plt.subplot(n_rows, n_cols, i + 1)
#
#         grid, gk = g_functions[feat]
#         grid = np.asarray(grid)
#         gk = np.asarray(gk)
#
#         # === 主曲线 g_k(x_k) ===
#         ax.plot(
#             grid,
#             gk,
#             color="darkblue",
#             linewidth=2,
#             label=r"$g_k(x_k)$"
#         )
#
#         # === 尝试线性趋势（安全版）===
#         try:
#             mask = np.isfinite(grid) & np.isfinite(gk)
#             x_fit = grid[mask]
#             y_fit = gk[mask]
#
#             # 防止退化（常数 / 点太少）
#             if len(x_fit) > 5 and len(np.unique(x_fit)) > 1:
#                 z = np.polyfit(x_fit, y_fit, 1)
#                 p = np.poly1d(z)
#
#                 ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
#                 ss_res = np.sum((y_fit - p(x_fit)) ** 2)
#                 r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0
#
#                 ax.plot(
#                     x_fit,
#                     p(x_fit),
#                     "r--",
#                     linewidth=1.5,
#                     label=f"Linear fit (R²={r2_lin:.3f})"
#                 )
#         except Exception as e:
#             print(f"[Warning] Linear fit failed for '{feat}': {e}")
#
#         # === 图形细节 ===
#         ax.set_title(feat, fontsize=11, fontweight="bold")
#         ax.set_xlabel(feat, fontsize=9)
#         ax.set_ylabel("Additive Effect", fontsize=9)
#         ax.grid(True, alpha=0.3)
#         ax.legend(fontsize=8)
#
#     # === 隐藏多余子图 ===
#     for idx in range(n_features + 1, n_rows * n_cols + 1):
#         plt.subplot(n_rows, n_cols, idx).axis("off")
#
#     fig.suptitle(
#         "Prior Method II: Additive Effects from Strong Model",
#         fontsize=16,
#         fontweight="bold",
#         y=0.98
#     )
#
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.93)
#
#     os.makedirs(save_path, exist_ok=True)
#     save_file = os.path.join(save_path, "prior2_partial_dependence.png")
#     plt.savefig(save_file, dpi=300, bbox_inches="tight")
#     plt.close()
#
#     print(f"Prior Method II 部分依赖图已保存到：{save_file}")

def plot_prior2_partial_dependence(
    prior2_results: dict,
    save_path: str,
    max_cols: int = 4,
    lowess_frac: float = 0.1  # LOWESS平滑参数
):
    """
    绘制 Prior Method II 的部分依赖图（带 LOWESS 平滑）
    """
    numeric_features = [
        'duration_ms', 'acousticness', 'danceability', 'energy',
        'instrumentalness', 'liveness', 'loudness', 'speechiness',
        'tempo', 'valence', 'key', 'singer_num', 'name_len'
    ]

    print("===== 绘制 Prior Method II 的部分依赖图（带LOWESS平滑） =====")

    g_functions = prior2_results["g_functions"]
    features = [f for f in numeric_features if f in g_functions]
    n_features = len(features)

    n_cols = max_cols
    n_rows = int(np.ceil(n_features / n_cols))
    fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows))
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

    for i, feat in enumerate(features):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        grid, gk = g_functions[feat]
        grid = np.asarray(grid)
        gk = np.asarray(gk)

        # 原始曲线
        ax.plot(grid, gk, color="darkblue", linewidth=2, label=r"$g_k(x_k)$")

        # LOWESS平滑曲线
        lowess_result = lowess(gk, grid, frac=lowess_frac)
        ax.plot(lowess_result[:, 0], lowess_result[:, 1],
                color="green", linewidth=1, label="LOWESS Smooth")

        # 线性趋势（安全版）
        try:
            mask = np.isfinite(grid) & np.isfinite(gk)
            x_fit = grid[mask]
            y_fit = gk[mask]

            if len(x_fit) > 5 and len(np.unique(x_fit)) > 1:
                z = np.polyfit(x_fit, y_fit, 1)
                p = np.poly1d(z)
                ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
                ss_res = np.sum((y_fit - p(x_fit)) ** 2)
                r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                ax.plot(x_fit, p(x_fit), "r--", linewidth=1.5,
                        label=f"Linear fit (R²={r2_lin:.3f})")
        except Exception as e:
            print(f"[Warning] Linear fit failed for '{feat}': {e}")

        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("Additive Effect", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(n_features + 1, n_rows * n_cols + 1):
        plt.subplot(n_rows, n_cols, idx).axis("off")

    fig.suptitle(
        "Prior Method II: Partial Dependence With Strong Generalization Model",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "prior2_partial_dependence_lowess.png")
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Prior Method II 部分依赖图（平滑版）已保存到：{save_file}")






def descriptive_analysis_pipeline(X_train: pd.DataFrame, y_train: pd.Series, save_path: str):
    """描述性分析"""

    # 基础统计量
    print("\n1. 计算基础统计量...")
    basic_stats = basic_statistics(X_train, y_train)
    basic_stats.to_csv(f"{save_path}/basic_statistics.csv", encoding='utf-8-sig')
    print(f"基础统计量已保存到: {save_path}/basic_statistics.csv")

    # 基础可视化
    print("\n2. 生成基础可视化...")
    visualization_basic(X_train, y_train, save_path)
    print(f"基础可视化图已保存到: {save_path}/")


    # 新增：特征分布直方图
    print("\n3. 绘制特征分布直方图...")
    plot_feature_distributions(X_train, save_path)


    # GAM分析
    print("\n4. 进行GAM分析...")
    gam_results = gam_analysis(X_train, y_train, save_path)

    # print("\n5. 绘制特征与流行度直接关系图...")
    # plot_feature_popularity_relation(X_train, y_train, save_path=f"{save_root}/descriptive_analysis")

    return basic_stats, gam_results


if __name__=="__main__":
    train_path = "train_processed.csv"
    test_path = "test_processed.csv"
    save_root = "experiment_results"  # 结果根目录
    random_state = 42

    df_train = read_df(train_path)
    df_test = read_df(test_path)

    X_train, y_train, X_test, y_test = split_target(df_train, df_test)

    # 创建保存目录
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(f"{save_root}/descriptive_analysis", exist_ok=True)


    # 数据描述性分析
    print("\n===== 步骤2：数据描述性分析 =====")
    basic_stats, gam_model = descriptive_analysis_pipeline(
        X_train, y_train, save_path=f"{save_root}/descriptive_analysis"
    )

    # ===============================
    # 方法二：强模型 → 加性近似
    # ===============================
    print("\n===== Prior Method II: Strong Model Based Additive =====")

    prior2_results = prior_method_two_analysis(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    plot_prior2_partial_dependence(
        prior2_results=prior2_results,
        save_path=f"{save_root}/descriptive_analysis"
    )
