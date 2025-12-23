import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import os


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def poison_data(X, y, poison_ratio=0.1, poison_strength=0.5, random_state=42, mode="target", label_flip=False):
    """
    对数据进行投毒
    mode:
        "target" -> 修改y值
        "feature" -> 修改高影响特征
        "both" -> 同时修改
    label_flip:
        True -> 对投毒样本进行标签反转
    """
    np.random.seed(random_state)
    X_p = deepcopy(X)
    y_p = deepcopy(y)
    n_samples = X.shape[0]
    n_poison = int(n_samples * poison_ratio)
    idx = np.random.choice(n_samples, n_poison, replace=False)

    if mode in ["target", "both"]:
        if label_flip:
            # 标签反转，映射到极端值
            y_min, y_max = np.min(y), np.max(y)
            # 将选中的样本反转到相对最大值
            y_p[idx] = y_max + y_min - y_p[idx]
        else:
            # 在目标变量上加噪声
            y_std = np.std(y)
            y_p[idx] += np.random.randn(n_poison) * y_std * poison_strength

    if mode in ["feature", "both"]:
        # 在随机特征上加噪声
        n_features = X.shape[1]
        f_idx = np.random.choice(n_features, max(1, n_features // 10), replace=False)
        for f in f_idx:
            X_std = np.std(X[:, f])
            X_p[idx, f] += np.random.randn(n_poison) * X_std * poison_strength

    return X_p, y_p


def evaluate_robustness(models_dict, X_train, y_train, X_test, y_test,
                        poison_ratios=[0.0, 0.05, 0.1, 0.2],
                        poison_strength=0.5,
                        mode="target",
                        label_flip=False):
    """
    对多模型在不同投毒强度下评估鲁棒性
    """
    results = []

    for ratio in poison_ratios:
        X_train_p, y_train_p = poison_data(
            X_train, y_train,
            poison_ratio=ratio,
            poison_strength=poison_strength,
            mode=mode,
            label_flip=label_flip
        )
        for name, func in models_dict.items():
            try:
                pred_test = func(X_train_p, y_train_p, X_test)
            except Exception as e:
                print(f"Error in {name} with poison_ratio={ratio}: {e}")
                pred_test = np.full_like(y_test, np.nan)

            rmse_val = rmse(y_test, pred_test)
            r2_val = r2_score(y_test, pred_test)
            results.append({
                "model": name,
                "poison_ratio": ratio,
                "rmse": rmse_val,
                "r2": r2_val
            })

    return pd.DataFrame(results)

def plot_robustness(df, metric="rmse"):
    """
    绘制不同投毒比例下，各模型的性能变化折线图
    df: evaluate_robustness 输出的 DataFrame
    metric: "rmse" 或 "r2"
    """
    plt.figure(figsize=(8, 5))
    sns.set(style="whitegrid", font_scale=1.1)

    for model_name in df['model'].unique():
        sub_df = df[df['model'] == model_name]
        plt.plot(sub_df['poison_ratio'], sub_df[metric], marker='o', label=model_name)

    # 路径配置
    save_root = "experiment_results"  # 结果根目录
    save_dir = os.path.join(save_root, "results")  # 拼接子目录路径
    os.makedirs(save_dir, exist_ok=True)  # 创建目录

    plt.xlabel("Poison Ratio")
    plt.ylabel(metric.upper())
    plt.title(f"Model Robustness under Poisoning ({metric.upper()})")
    plt.xticks(df['poison_ratio'].unique())
    plt.legend()
    save_path = os.path.join(save_dir, f'{metric}_poisoning_robustness.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    from Modelling_final import load_data, localized_gam, fpr_admm_regression, incremental_dictionary_learning, \
        xgboost_model

    X_train, X_test, y_train, y_test, feature_names = load_data()

    models = {
        "L-GAM": localized_gam,
        "FPR-ADMM": fpr_admm_regression,
        "IDL-Greedy": incremental_dictionary_learning,
        "XGBoost": xgboost_model
    }

    # 测试标签反转投毒
    df_robust_flip = evaluate_robustness(
        models, X_train, y_train, X_test, y_test,
        poison_ratios=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        poison_strength=0.5,
        mode="both",
        label_flip=False
    )
    print(df_robust_flip)

    # 可视化RMSE
    plot_robustness(df_robust_flip, metric="rmse")

    # 可视化R2
    plot_robustness(df_robust_flip, metric="r2")
