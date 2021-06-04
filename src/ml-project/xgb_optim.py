import argparse
import warnings

import joblib
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data.dataset import load_dataset
from model.gbdt import stratified_kfold_xgb

warnings.filterwarnings("ignore")

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def objective(trial: Trial) -> float:
    params_xgb = {
        "random_state": 42,
        "n_estimators": 10000,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.15,
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


if __name__ == "__main__":
    parse = argparse.ArgumentParser("Optimize")
    parse.add_argument("--fold", type=int, default=10)
    parse.add_argument("--trials", type=int, default=360)
    parse.add_argument("--params", type=str, default="params.pkl")
    args = parse.parse_args()

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name="xgb_parameter_opt",
        direction="maximize",
        sampler=sampler,
    )

    study.optimize(objective, n_trials=args.trials)
    print("Best Score:", study.best_value)
    print("Best trial:", study.best_trial.params)

    params = study.best_trial.params
    params["random_state"] = 42
    params["n_estimators"] = 10000
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "auc"
    joblib.dump(params, "../../parameters/" + args.params)
