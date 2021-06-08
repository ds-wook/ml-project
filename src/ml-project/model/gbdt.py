from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def stratified_kfold_lgbm(
    params: Dict[str, Union[int, float, str]],
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    verbose: Union[int, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:

    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    splits = folds.split(X, y)

    lgb_oof = np.zeros(X.shape[0])
    lgb_preds = np.zeros(X_test.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(splits):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=verbose,
        )
        lgb_oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
        lgb_preds += model.predict_proba(X_test)[:, 1] / n_fold

    auc_score = roc_auc_score(y, lgb_oof)
    print(f"ROC-AUC valid score: {auc_score:.5f}")

    return lgb_oof, lgb_preds


def stratified_kfold_cat(
    params: Dict[str, Union[int, float, str, List[str]]],
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    verbose: Union[int, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    splits = folds.split(X, y)
    cat_oof = np.zeros(X.shape[0])
    cat_preds = np.zeros(X_test.shape[0])
    cat_cols = [c for c in X.columns if X[c].dtypes == "int64"]

    for fold, (train_idx, valid_idx) in enumerate(splits):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        train_data = Pool(data=X_train, label=y_train, cat_features=cat_cols)
        valid_data = Pool(data=X_valid, label=y_valid, cat_features=cat_cols)

        model = CatBoostClassifier(**params)

        model.fit(
            train_data,
            eval_set=valid_data,
            early_stopping_rounds=100,
            use_best_model=True,
            verbose=verbose,
        )

        cat_oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
        cat_preds += model.predict_proba(X_test)[:, 1] / n_fold

    auc_score = roc_auc_score(y, cat_oof)
    print(f"ROC-AUC valid score: {auc_score:.5f}")

    return cat_oof, cat_preds


def stratified_kfold_xgb(
    params: Dict[str, Union[int, float, str]],
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    verbose: Union[int, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:

    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    splits = folds.split(X, y)
    xgb_oof = np.zeros(X.shape[0])
    xgb_preds = np.zeros(X_test.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(splits):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=verbose,
        )

        xgb_oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
        xgb_preds += model.predict_proba(X_test)[:, 1] / n_fold

    fig, ax = plt.subplots(figsize=(20, 14))
    xgb.plot_importance(model, ax=ax, max_num_features=len(X_test.columns))
    plt.savefig("../../graph/xgb_import.png")
    auc_score = roc_auc_score(y, xgb_oof)
    print(f"ROC-AUC valid score: {auc_score:.5f}")

    return xgb_oof, xgb_preds
