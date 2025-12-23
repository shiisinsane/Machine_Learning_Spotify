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
from itertools import product

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

    X_train = train.drop(columns=[TARGET]).values
    y_train = train[TARGET].values

    X_test = test.drop(columns=[TARGET]).values
    y_test = test[TARGET].values

    return X_train, X_test, y_train, y_test

# =====================================================
# Cross-Validation Wrapper
# =====================================================
def cross_val_score_model(model_func, X, y, random_state=RANDOM_STATE):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    rmse_list = []
    r2_list = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        pred_val = model_func(X_tr, y_tr, X_val)
        rmse_list.append(rmse(y_val, pred_val))
        r2_list.append(r2_score(y_val, pred_val))

    return np.mean(rmse_list), np.mean(r2_list)

def grid_search_cv(model_class, param_grid, X, y,
                   n_splits=5, random_state=42):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    best_rmse = np.inf
    best_params = None

    keys = list(param_grid.keys())
    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        rmse_list = []

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = model_class(**params)
            model.fit(X_tr, y_tr)
            pred_val = model.predict(X_val)
            rmse_list.append(rmse(y_val, pred_val))

        mean_rmse = np.mean(rmse_list)

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params

    # 用最优参数在全训练集 refit
    best_model = model_class(**best_params)
    best_model.fit(X, y)

    return best_model, best_params, best_rmse

# =====================================================
# Model A: Localized GAM
# =====================================================
def localized_gam(X_train, y_train, X_val):
    np.random.seed(RANDOM_STATE)
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(X_train)

    preds_train = np.zeros(len(y_train))
    for c in range(n_clusters):
        idx = clusters == c
        X_c = X_train[idx]
        y_c = y_train[idx]
        terms = reduce(lambda a, b: a + b, [s(i) for i in range(X_train.shape[1])])
        gam = LinearGAM(terms)
        gam.fit(X_c, y_c)
        preds_train[idx] = gam.predict(X_c)

    # 验证集预测
    clusters_val = kmeans.predict(X_val)
    preds_val = np.zeros(len(X_val))
    for c in range(n_clusters):
        idx_val = clusters_val == c
        X_c_val = X_val[idx_val]
        # 对应聚类的训练GAM
        idx_tr = clusters == c
        gam = LinearGAM(terms).fit(X_train[idx_tr], y_train[idx_tr])
        preds_val[idx_val] = gam.predict(X_c_val)

    return preds_val

# =====================================================
# Model B: FPC-ADMM
# =====================================================
def fpc_admm_regression(X_train, y_train, X_val, degree=2, n_anchors=300, lam=1.0, rho=1.0, n_iter=20):
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
        #v_new = (u_new + w) / (1 + lam / rho)
        v_new = np.sign(u_new + w) * np.maximum(np.abs(u_new + w) - lam / rho, 0)
        w += u_new - v_new
        if np.linalg.norm(u_new - u) < 1e-4:
            break
        u = u_new
        v = v_new

    A_val = (1 + X_val @ anchors.T) ** degree
    pred_val = A_val @ u
    return pred_val

# =====================================================
# Model C: Incremental Dictionary Learning + RGA + Lasso
# =====================================================
def incremental_dictionary_lasso_rga(X_train, y_train, X_val, n_atoms=50, nu=0.9):
    np.random.seed(RANDOM_STATE)
    n_samples, n_features = X_train.shape
    D = []
    Z_train = np.zeros((n_samples, n_atoms))
    R = X_train.copy()

    for k in range(n_atoms):
        scores = np.sum(R ** 2, axis=0)
        max_score = np.max(scores)
        candidate_idx = np.where(scores >= nu * max_score)[0]
        idx = np.random.choice(candidate_idx)
        atom = np.zeros(n_features)
        atom[idx] = 1.0
        atom = atom / np.linalg.norm(atom)
        D.append(atom)
        z_k = R @ atom
        Z_train[:, k] = z_k
        R -= np.outer(z_k, atom)

    D = np.array(D)
    lasso = Lasso(alpha=0.01, random_state=RANDOM_STATE)
    lasso.fit(Z_train, y_train)

    Z_val = np.zeros((X_val.shape[0], n_atoms))
    for k in range(n_atoms):
        Z_val[:, k] = X_val @ D[k]
    pred_val = lasso.predict(Z_val)
    return pred_val

# def incremental_dictionary_lasso_rga(
#     X_train, y_train, X_val,
#     n_atoms=50, nu=0.9, alpha=0.5, lasso_alpha=0.01
# ):
#     np.random.seed(RANDOM_STATE)
#
#     n_samples, n_features = X_train.shape
#
#     selected_idx = []
#     Z_train = np.zeros((n_samples, n_atoms))
#
#     # 初始化
#     f_train = np.zeros(n_samples)
#     residual = y_train.copy()
#
#     for k in range(n_atoms):
#         # === 1. 监督贪婪选择（与预测残差相关） ===
#         scores = np.abs(X_train.T @ residual)
#         max_score = np.max(scores)
#         candidate_idx = np.where(scores >= nu * max_score)[0]
#         j_k = np.random.choice(candidate_idx)
#
#         selected_idx.append(j_k)
#
#         # === 2. 最优系数（一维最小二乘） ===
#         x_j = X_train[:, j_k]
#         c_k = (x_j @ residual) / (x_j @ x_j + 1e-12)
#
#         # === 3. 松弛更新 ===
#         f_train = (1 - alpha) * f_train + alpha * c_k * x_j
#         residual = y_train - f_train
#
#         Z_train[:, k] = x_j
#
#     # === 4. Lasso 再稀疏化 ===
#     lasso = Lasso(alpha=lasso_alpha, random_state=RANDOM_STATE)
#     lasso.fit(Z_train, y_train)
#
#     # === 验证 ===
#     Z_val = X_val[:, selected_idx]
#     pred_val = lasso.predict(Z_val)
#
#     return pred_val


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
    pred_val = model.predict(X_val)
    return pred_val

def xgboost_grid_search(X_train, y_train):
    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],

        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [1, 2, 5],

        "random_state": [RANDOM_STATE],
        "n_jobs": [-1]
    }

    model, best_params, best_rmse = grid_search_cv(
        xgb.XGBRegressor,
        param_grid,
        X_train,
        y_train
    )

    return model, best_params, best_rmse


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
    pred_val = model.predict(X_val)
    return pred_val

def lightgbm_grid_search(X_train, y_train):
    param_grid = {
        "n_estimators": [500, 1000],
        "learning_rate": [0.03, 0.05],
        "num_leaves": [31, 63],
        "max_depth": [-1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "random_state": [RANDOM_STATE],
        "n_jobs": [-1]
    }

    model, best_params, best_rmse = grid_search_cv(
        lgb.LGBMRegressor,
        param_grid,
        X_train,
        y_train
    )

    return model, best_params, best_rmse


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    models = {
        "Localized GAM": localized_gam,
        "FPC-ADMM": fpc_admm_regression,
        "Dictionary+Lasso": incremental_dictionary_lasso_rga,
        "XGBoost": xgboost_model,
        "LightGBM": lightgbm_model
    }

    for name, func in models.items():
        print(f"\n===== Model {name} 5-Fold CV =====")
        start = time.time()
        cv_rmse, cv_r2 = cross_val_score_model(func, X_train, y_train)
        print(f"CV Train RMSE: {cv_rmse:.4f}, CV Train R²: {cv_r2:.4f}")
        print("Time:", time.time() - start)

        # 全训练集训练并预测（修正tuple判断错误）
        pred_train = func(X_train, y_train, X_train)
        pred_test = func(X_train, y_train, X_test)

        print(f"Full Train RMSE: {rmse(y_train, pred_train):.4f}, R²: {r2_score(y_train, pred_train):.4f}")
        print(f"Test RMSE: {rmse(y_test, pred_test):.4f}, R²: {r2_score(y_test, pred_test):.4f}")


    # print("\n" + "=" * 60)
    # print(" Grid Search Models (5-Fold CV)")
    # print("=" * 60)
    # # ---------- XGBoost ----------
    # print("\n>>> XGBoost Grid Search")
    # start = time.time()
    # xgb_best, xgb_params, xgb_cv_rmse = xgboost_grid_search(X_train, y_train)
    # print("Best Params:", xgb_params)
    # print(f"CV RMSE: {xgb_cv_rmse:.4f}")
    # print("Time:", time.time() - start)
    #
    # pred_train = xgb_best.predict(X_train)
    # pred_test = xgb_best.predict(X_test)
    # print(f"Train RMSE: {rmse(y_train, pred_train):.4f}, R²: {r2_score(y_train, pred_train):.4f}")
    # print(f"Test  RMSE: {rmse(y_test, pred_test):.4f}, R²: {r2_score(y_test, pred_test):.4f}")
    #
    # # ---------- LightGBM ----------
    # print("\n>>> LightGBM Grid Search")
    # start = time.time()
    # lgb_best, lgb_params, lgb_cv_rmse = lightgbm_grid_search(X_train, y_train)
    # print("Best Params:", lgb_params)
    # print(f"CV RMSE: {lgb_cv_rmse:.4f}")
    # print("Time:", time.time() - start)
    #
    # pred_train = lgb_best.predict(X_train)
    # pred_test = lgb_best.predict(X_test)
    # print(f"Train RMSE: {rmse(y_train, pred_train):.4f}, R²: {r2_score(y_train, pred_train):.4f}")
    # print(f"Test  RMSE: {rmse(y_test, pred_test):.4f}, R²: {r2_score(y_test, pred_test):.4f}")


