import time
import numpy as np
import pandas as pd

from functools import reduce
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from pygam import LinearGAM, s
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold

RANDOM_STATE = 42
TARGET = "popularity"
N_SPLITS = 5  # 交叉验证折数

# =====================================================
# Utils
# =====================================================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def load_data():
    train = pd.read_csv("train_processed.csv")
    test = pd.read_csv("test_processed.csv")

    feature_names = train.drop(columns=[TARGET]).columns.tolist()

    X_train = train.drop(columns=[TARGET]).values
    y_train = train[TARGET].values

    X_test = test.drop(columns=[TARGET]).values
    y_test = test[TARGET].values

    return X_train, X_test, y_train, y_test, feature_names


# =====================================================
# Cross-Validation
# =====================================================
def cross_val_score_model(model_func, X, y, random_state=RANDOM_STATE):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    rmse_list, r2_list = [], []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        out = model_func(X_tr, y_tr, X_val)
        pred = out["pred"] if isinstance(out, dict) else out

        rmse_list.append(rmse(y_val, pred))
        r2_list.append(r2_score(y_val, pred))

    return np.mean(rmse_list), np.mean(r2_list)


# =====================================================
# Model A: Localized GAM
# =====================================================
def localized_gam(X_train, y_train, X_val, return_model=False):
    np.random.seed(RANDOM_STATE)
    n_clusters = 10

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(X_train)

    terms = reduce(lambda a, b: a + b, [s(i) for i in range(X_train.shape[1])])
    gam_models = {}

    for c in range(n_clusters):
        idx = clusters == c
        gam = LinearGAM(terms).fit(X_train[idx], y_train[idx])
        gam_models[c] = gam

    clusters_val = kmeans.predict(X_val)
    preds_val = np.zeros(len(X_val))

    for c, gam in gam_models.items():
        idx = clusters_val == c
        preds_val[idx] = gam.predict(X_val[idx])

    if return_model:
        return {
            "pred": preds_val,
            "gam_models": gam_models,
            "kmeans": kmeans
        }
    return preds_val


def explain_localized_gam(gam_models, feature_names, top_n=3):
    """
    返回每个簇中前 top_n 个影响最大的特征及其 effect_strength
    """
    records = []

    for c, gam in gam_models.items():
        cluster_effects = []
        for i, fname in enumerate(feature_names):
            XX = gam.generate_X_grid(term=i)
            pdp = gam.partial_dependence(term=i, X=XX)
            effect = np.max(pdp) - np.min(pdp)
            cluster_effects.append((fname, effect))

        # 对该簇按 effect_strength 降序排序
        cluster_effects.sort(key=lambda x: x[1], reverse=True)

        # 取前 top_n 个特征
        for rank, (fname, effect) in enumerate(cluster_effects[:top_n], 1):
            records.append({
                "cluster": c,
                "rank": rank,
                "feature": fname,
                "effect_strength": effect
            })

    return pd.DataFrame(records)



# =====================================================
# Model B: FPR-ADMM
# =====================================================
def fpr_admm_regression(
    X_train, y_train, X_val,
    degree=2, n_anchors=300, lam=1.0, rho=1.0, n_iter=30,
    return_model=False
):
    np.random.seed(RANDOM_STATE)
    n_samples = X_train.shape[0]

    anchor_idx = np.random.choice(n_samples, n_anchors, replace=False)
    anchors = X_train[anchor_idx]

    A = (1 + X_train @ anchors.T) ** degree
    u = np.zeros(n_anchors)
    v = np.zeros_like(u)
    w = np.zeros_like(u)

    AtA = A.T @ A
    Aty = A.T @ y_train
    I = np.eye(n_anchors)

    for _ in range(n_iter):
        u_new = np.linalg.solve(AtA + rho * I, Aty + rho * (v - w))
        v_new = np.sign(u_new + w) * np.maximum(np.abs(u_new + w) - lam / rho, 0)
        w += u_new - v_new
        if np.linalg.norm(u_new - u) < 1e-4:
            break
        u, v = u_new, v_new

    A_val = (1 + X_val @ anchors.T) ** degree
    pred_val = A_val @ u

    if return_model:
        return {
            "pred": pred_val,
            "u": u,
            "anchors": anchors
        }
    return pred_val

# def fpr_admm_regression(
#     X_train, y_train, X_val,
#     degree=2, n_anchors=300, lam=1.0, rho=1.0, n_iter=30,
#     n_runs=1,  # 随机锚点重复次数
#     return_model=False
# ):
#     """
#     Fast Polynomial kernel Regression with multiple random anchor averaging
#     """
#     np.random.seed(RANDOM_STATE)
#     n_samples = X_train.shape[0]
#     preds_val_total = np.zeros(X_val.shape[0])
#
#     # 多次随机锚点
#     for run in range(n_runs):
#         anchor_idx = np.random.choice(n_samples, n_anchors, replace=False)
#         anchors = X_train[anchor_idx]
#
#         # 训练集特征映射
#         A = (1 + X_train @ anchors.T) ** degree
#         u = np.zeros(n_anchors)
#         v = np.zeros_like(u)
#         w = np.zeros_like(u)
#
#         AtA = A.T @ A
#         Aty = A.T @ y_train
#         I = np.eye(n_anchors)
#
#         for _ in range(n_iter):
#             u_new = np.linalg.solve(AtA + rho * I, Aty + rho * (v - w))
#             v_new = np.sign(u_new + w) * np.maximum(np.abs(u_new + w) - lam / rho, 0)
#             w += u_new - v_new
#             if np.linalg.norm(u_new - u) < 1e-4:
#                 break
#             u, v = u_new, v_new
#
#         # 验证集预测
#         A_val = (1 + X_val @ anchors.T) ** degree
#         preds_val_total += A_val @ u
#
#     # 取平均
#     pred_val = preds_val_total / n_runs
#
#     if return_model:
#         return {
#             "pred": pred_val,
#             "n_runs": n_runs,
#         }
#     return pred_val



def explain_fpr(u, threshold=1e-3):
    active = np.sum(np.abs(u) > threshold)
    return {
        "active_kernels": active,
        "sparsity_ratio": active / len(u)
    }


# =====================================================
# Model C: Incremental Dictionary Learning
# =====================================================
def incremental_dictionary_learning(
    X_train, y_train, X_val,
    n_atoms=50, nu=0.9,
    return_model=False
):
    np.random.seed(RANDOM_STATE)
    n_samples, n_features = X_train.shape

    selected_idx = []
    Z_train = np.zeros((n_samples, n_atoms))
    R = X_train.copy()

    for k in range(n_atoms):
        scores = np.sum(R ** 2, axis=0)
        max_score = np.max(scores)
        candidate_idx = np.where(scores >= nu * max_score)[0]
        idx = np.random.choice(candidate_idx)

        selected_idx.append(idx)

        atom = np.zeros(n_features)
        atom[idx] = 1.0
        atom /= np.linalg.norm(atom)

        z_k = R @ atom
        Z_train[:, k] = z_k
        R -= np.outer(z_k, atom)

    lasso = Lasso(alpha=0.01, random_state=RANDOM_STATE)
    lasso.fit(Z_train, y_train)

    Z_val = np.zeros((X_val.shape[0], n_atoms))
    for k, idx in enumerate(selected_idx):
        Z_val[:, k] = X_val[:, idx]

    pred_val = lasso.predict(Z_val)

    if return_model:
        return {
            "pred": pred_val,
            "selected_idx": selected_idx,
            "lasso_coef": lasso.coef_
        }
    return pred_val


def explain_idl(selected_idx, coef, feature_names):
    records = []
    for i, j in enumerate(selected_idx):
        if coef[i] != 0:
            records.append({
                "feature": feature_names[j],
                "weight": coef[i]
            })
    return pd.DataFrame(records).sort_values(
        by="weight", key=np.abs, ascending=False
    )


# =====================================================
# Model D: XGBoost
# =====================================================
def xgboost_model(X_train, y_train, X_val):
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model.predict(X_val)


# =====================================================
# Model E: LightGBM
# =====================================================
def lightgbm_model(X_train, y_train, X_val):
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model.predict(X_val)



if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_data()

    models = {
        "Localized GAM": localized_gam,
        "FPR-ADMM": fpr_admm_regression,
        "IDL-Greedy": incremental_dictionary_learning,
        "XGBoost": xgboost_model,
        "LightGBM": lightgbm_model
    }

    for name, func in models.items():
        print(f"\n===== Model {name} 5-Fold CV =====")
        start = time.time()
        cv_rmse, cv_r2 = cross_val_score_model(func, X_train, y_train)
        print(f"CV RMSE: {cv_rmse:.4f}, R²: {cv_r2:.4f}")
        print("Time:", time.time() - start)

        # 全训练集训练并预测
        pred_train = func(X_train, y_train, X_train)
        pred_test = func(X_train, y_train, X_test)

        print(f"Full Train RMSE: {rmse(y_train, pred_train):.4f}, R²: {r2_score(y_train, pred_train):.4f}")
        print(f"Test RMSE: {rmse(y_test, pred_test):.4f}, R²: {r2_score(y_test, pred_test):.4f}")

    # ================== 解释性 ==================
    print("\n===== IDL-Greedy Interpretability =====")
    out = incremental_dictionary_learning(
        X_train, y_train, X_test, return_model=True
    )
    print(explain_idl(out["selected_idx"], out["lasso_coef"], feature_names).head(10))


    print("\n===== L-GAM Interpretability =====")
    out = localized_gam(X_train, y_train, X_test, return_model=True)

    df_top3 = explain_localized_gam(out["gam_models"], feature_names, top_n=3)

    # 按簇排序显示，每簇前3特征
    df_top3.sort_values(["cluster", "rank"], inplace=True)
    print(df_top3)

    print("\n===== FPR-ADMM Interpretability =====")
    out = fpr_admm_regression(
        X_train, y_train, X_test, return_model=True
    )
    print(explain_fpr(out["u"]))


