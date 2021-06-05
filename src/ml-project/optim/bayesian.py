from typing import List

import joblib
import optuna
import pandas as pd
from model.gbdt import stratified_kfold_cat, stratified_kfold_lgbm, stratified_kfold_xgb
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score


class BayesianOptimizer:
    def __init__(self, objective_function: object):
        self.objective_function = objective_function

    def build_study(self, trials: int, verbose: bool = False):
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            study_name="parameter_opt",
            direction="maximize",
            sampler=sampler,
        )
        study.optimize(self.objective_function, n_trials=trials)
        if verbose:
            self.display_study_statistics(study)
        return study

    def display_study_statistics(study: optuna.create_study):
        print("Best Score:", study.best_value)
        print("Best trial:", study.best_trial.params)

    @staticmethod
    def lgbm_save_params(study: optuna.create_study, params_name: str):
        params = study.best_trial.params
        params["random_state"] = 42
        params["boosting_type"] = "gbdt"
        params["learning_rate"] = 0.05
        params["n_estimators"] = 10000
        params["objective"] = "binary"
        params["metric"] = "auc"
        joblib.dump(params, "../../parameters/" + params_name)

    @staticmethod
    def cat_save_params(
        study: optuna.create_study, params_name: str, cat_cols: List[str]
    ):
        params = study.best_trial.params
        params["random_state"] = 42
        params["eval_metric"] = "AUC"
        params["loss_function"] = "Logloss"
        params["od_type"] = "Iter"
        params["od_wait"] = 500
        params["iterations"] = 10000
        params["cat_features"] = cat_cols
        joblib.dump(params, "../../parameters/" + params_name)

    @staticmethod
    def xgb_save_params(study: optuna.create_study, params_name: str):
        params = study.best_trial.params
        params["random_state"] = 42
        params["n_estimators"] = 10000
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "auc"
        joblib.dump(params, "../../parameters/" + params_name)

    @staticmethod
    def plot_optimization_history(study: optuna.create_study) -> optuna.visualization:
        return optuna.visualization.plot_optimization_history(study)

    @staticmethod
    def plot_param_importances(study: optuna.create_study) -> optuna.visualization:
        return optuna.visualization.plot_param_importances(study)

    @staticmethod
    def plot_edf(study: optuna.create_study) -> optuna.visualization:
        return optuna.visualization.plot_edf(study)


def lgbm_objective(
    trial: Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    args: int,
) -> float:
    params_lgb = {
        "random_state": 42,
        "verbosity": -1,
        "learning_rate": 0.05,
        "n_estimators": 10000,
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 3e-5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "max_bin": trial.suggest_int("max_bin", 200, 500),
    }
    lgb_oof, lgb_preds = stratified_kfold_lgbm(
        params_lgb, args, X_train, y_train, X_test
    )
    return roc_auc_score(y_train, lgb_oof)


def cat_objective(
    trial: Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cat_cols: List[str],
    args: int,
) -> float:
    params_cat = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "od_type": "Iter",
        "od_wait": 500,
        "random_seed": 42,
        "iterations": 10000,
        "cat_features": cat_cols,
        "learning_rate": trial.suggest_uniform("learning_rate", 1e-5, 1.0),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 1.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "bagging_temperature": trial.suggest_int("bagging_temperature", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "max_bin": trial.suggest_int("max_bin", 200, 500),
    }

    cat_oof, cat_preds = stratified_kfold_cat(
        params_cat, args, X_train, y_train, X_test
    )
    return roc_auc_score(y_train, cat_oof)


def xgb_objective(
    trial: Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    args: int,
) -> float:
    params_xgb = {
        "random_state": 42,
        "n_estimators": 10000,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.1,
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "max_leaves": trial.suggest_int("max_leaves", 2, 256),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 100),
        "gamma": trial.suggest_float("gamma", 0.5, 1),
    }

    xgb_oof, xgb_preds = stratified_kfold_xgb(
        params_xgb, args.fold, X_train, y_train, X_test
    )

    return roc_auc_score(y_train, xgb_oof)
