import argparse

import joblib
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data.dataset import load_dataset
from model.gbdt import stratified_kfold_cat

path = "../../input/siim-isic-melanoma-classification/"
train = load_dataset(path)

features = [
    "age_approx",
    "age_enc",
    "age_approx_mean_enc",
    "age_id_min",
    "age_id_max",
    "sex_enc",
    "sex_mean_enc",
    "anatom_enc",
    "anatom_mean_enc",
    "n_images",
    "n_images_mean_enc",
    "n_images_enc",
    "image_size_scaled",
    "image_size_enc",
    "image_size_mean_enc",
]

X = train[features]
y = train["target"]
cat_cols = [c for c in X.columns if X[c].dtypes == "int64"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def objective(trial: Trial) -> float:
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
        params_cat, args.fold, X_train, y_train, X_test
    )
    return roc_auc_score(y_train, cat_oof)


if __name__ == "__main__":
    parse = argparse.ArgumentParser("Optimize")
    parse.add_argument("--fold", type=int, default=10)
    parse.add_argument("--trials", type=int, default=360)
    parse.add_argument("--params", type=str, default="params.pkl")
    args = parse.parse_args()
    study = optuna.create_study(
        study_name="cat_parameter_opt",
        direction="maximize",
        sampler=TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=args.trials)
    print("Best Score:", study.best_value)
    print("Best trial:", study.best_trial.params)

    params = study.best_trial.params
    params["random_state"] = 42
    params["eval_metric"] = "AUC"
    params["loss_function"] = "Logloss"
    params["od_type"] = "Iter"
    params["od_wait"] = 500
    params["iterations"] = 10000
    params["cat_features"] = cat_cols
    joblib.dump(params, "../../parameters/" + args.params)
