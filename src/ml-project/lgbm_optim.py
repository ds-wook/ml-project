import argparse
from functools import partial

import pandas as pd
from optuna import Trial
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data.dataset import load_dataset
from model.gbdt import stratified_kfold_lgbm
from optim.bayesian import BayesianOptimizer


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


def define_argparser():
    parse = argparse.ArgumentParser("Optimize")
    parse.add_argument("--fold", type=int, default=10)
    parse.add_argument("--trials", type=int, default=360)
    parse.add_argument("--params", type=str, default="params.pkl")
    args = parse.parse_args()
    return args


def _main(args: argparse.Namespace):
    path = "../../input/siim-isic-melanoma-classification/"
    train = load_dataset(path)

    features = [
        "age_approx",
        "age_enc",
        "age_approx_mean_enc",
        "age_id_min",
        "age_id_max",
        "sex_enc",
        "anatom_enc",
        "n_images",
        "n_images_enc",
        "image_size_scaled",
        "image_size_enc",
    ]

    X = train[features]
    y = train["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    objective = partial(
        lgbm_objective, X_train=X_train, y_train=y_train, X_test=X_test, args=args.fold
    )

    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=args.trials)
    bayesian_optim.save_params(study, args.params)


if __name__ == "__main__":
    args = define_argparser()
    _main(args)
