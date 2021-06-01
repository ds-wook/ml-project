import argparse
import warnings

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data.dataset import load_dataset
from model.gbdt import stratified_kfold_cat, stratified_kfold_lgbm, stratified_kfold_xgb

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

if __name__ == "__main__":
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument("--fold", type=int, default=10)
    args = parse.parse_args()

    lgb_params = pd.read_pickle("../../parameters/best_lgbm_params.pkl")

    lgb_oof, lgb_preds = stratified_kfold_lgbm(
        lgb_params, args.fold, X_train, y_train, X_test, 100
    )

    eff_preds = pd.read_csv("../../submit/efficent_net.csv")["target"]

    xgb_params = {
        "eta": 0.023839252347297356,
        "reg_alpha": 6.99554614267605e-06,
        "reg_lambda": 0.010419988953061583,
        "max_depth": 15,
        "max_leaves": 159,
        "colsample_bytree": 0.4515469593932409,
        "subsample": 0.7732694309118915,
        "min_child_weight": 5,
        "gamma": 0.6847131315687576,
        "random_state": 42,
        "n_estimators": 10000,
        "objective": "binary:logistic",
        "eval_metric": "auc",
    }

    xgb_oof, xgb_preds = stratified_kfold_xgb(
        xgb_params, args.fold, X_train, y_train, X_test, 100
    )

    y_preds = 0.5 * eff_preds + 0.4 * lgb_preds + 0.1 * xgb_preds

    print("#### Scores ####")
    print(f"LGBM ROC-AUC Test Score: {roc_auc_score(y_test, lgb_preds):.5f}")
    print(f"XGB ROC-AUC Test Score: {roc_auc_score(y_test, xgb_preds):.5f}")
    print(f"Effiecent Net Test Score: {roc_auc_score(y_test, eff_preds.values):.5f}")
    print(f"Ensemble Test Score: {roc_auc_score(y_test, y_preds):.5f}")
