import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.dataset import load_dataset
from model.gbdt import stratified_kfold_lgbm

warnings.filterwarnings("ignore")


def define_argparser():
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument("--fold", type=int, default=10)
    parse.add_argument("--params", type=str, default="best_lgbm_params.pkl")
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
        # "sex_enc",
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

    lgb_params = pd.read_pickle("../../parameters/" + args.params)

    lgb_oof, lgb_preds = stratified_kfold_lgbm(
        lgb_params, args.fold, X_train, y_train, X_test, 100
    )
    eff_preds = pd.read_csv("../../submit/submission.csv")["target"]

    y_preds = 0.8 * eff_preds + 0.2 * lgb_preds

    X_test_scale = StandardScaler().fit_transform(X_test.values)
    pca = PCA(n_components=2)
    pca.fit(X_test_scale)
    pca_test = pca.transform(X_test_scale)
    pca_columns = ["pca_component1", "pca_component2"]
    pca_data = pd.DataFrame(pca_test, columns=pca_columns)
    pca_data["target"] = y_test
    pca_data["preds"] = np.where(y_preds > 0.2, 1, 0)

    plt.figure(figsize=(15, 15))
    sns.scatterplot(x="pca_component1", y="pca_component2", hue="target", data=pca_data)
    plt.savefig("../../graph/pca_figure.jpg")
    plt.close()

    print("#### Scores ####")
    print(f"LGBM ROC-AUC Test Score: {roc_auc_score(y_test, lgb_preds):.5f}")
    print(f"Effiecent Net Test Score: {roc_auc_score(y_test, eff_preds.values):.5f}")
    print(f"Ensemble Test Score: {roc_auc_score(y_test, y_preds):.5f}")


if __name__ == "__main__":
    args = define_argparser()
    _main(args)
