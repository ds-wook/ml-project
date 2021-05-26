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

    cat_params = pd.read_pickle("../../parameters/best_cat_params.pkl")
    cat_oof, cat_preds = stratified_kfold_cat(
        cat_params, args.fold, X_train, y_train, X_test, 100
    )

    xgb_params = pd.read_pickle("../../parameters/best_xgb_params.pkl")
    xgb_oof, xgb_preds = stratified_kfold_xgb(
        xgb_params, args.fold, X_train, y_train, X_test, 100
    )
    print("#### Scores ####")
    print(f"LGBM ROC-AUC Test Score: {roc_auc_score(y_test, lgb_preds):.5f}")
    print(f"CAT ROC-AUC Test Score: {roc_auc_score(y_test, cat_preds):.5f}")
    print(f"XGB ROC-AUC Test Score: {roc_auc_score(y_test, xgb_preds): .5f}")
