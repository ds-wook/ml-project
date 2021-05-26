import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, MinMaxScaler
from tqdm import tqdm


def load_dataset(path: str) -> pd.DataFrame:
    train = pd.read_csv(path + "train.csv")

    # 결측치 처리
    train["sex"].fillna("unknown", inplace=True)
    train["age_approx"].fillna(train["age_approx"].mode().values[0], inplace=True)
    train["anatom_site_general_challenge"].fillna("unknown", inplace=True)

    label_enc = LabelEncoder()
    train["sex_enc"] = label_enc.fit_transform(train["sex"].astype(str))
    train["age_enc"] = label_enc.fit_transform(train.age_approx.astype(str))
    train["anatom_enc"] = label_enc.fit_transform(
        train.anatom_site_general_challenge.astype(str)
    )
    train["age_approx"] = train["age_approx"].fillna(
        train["age_approx"].mode().values[0]
    )
    train["age_enc"] = label_enc.fit_transform(train["age_approx"].astype("str"))
    train["n_images"] = train["patient_id"].map(
        train.groupby(["patient_id"])["image_name"].count()
    )

    categorize = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform")
    train["n_images_enc"] = (
        categorize.fit_transform(train["n_images"].values.reshape(-1, 1))
        .astype(int)
        .squeeze()
    )
    train_images = train["image_name"].values

    train_sizes = np.zeros(train_images.shape[0])
    for i, img_path in enumerate(tqdm(train_images)):
        train_sizes[i] = os.path.getsize(
            os.path.join(
                path + "jpeg/train/",
                f"{img_path}.jpg",
            )
        )

    train["image_size"] = train_sizes

    min_max_scale = MinMaxScaler()
    train["image_size_scaled"] = min_max_scale.fit_transform(
        train["image_size"].values.reshape(-1, 1)
    )

    train["image_size_enc"] = (
        categorize.fit_transform(train["image_size_scaled"].values.reshape(-1, 1))
        .astype(int)
        .squeeze()
    )

    train["age_id_min"] = train["patient_id"].map(
        train.groupby(["patient_id"])["age_approx"].min()
    )
    train["age_id_max"] = train["patient_id"].map(
        train.groupby(["patient_id"])["age_approx"].max()
    )
    train["age_approx_mean_enc"] = train["age_approx"].map(
        train.groupby(["age_approx"])["target"].mean()
    )

    train["sex_mean_enc"] = train.sex_enc.map(
        train.groupby(["sex_enc"])["target"].mean()
    )
    train["n_images_mean_enc"] = train["n_images_enc"].map(
        train.groupby(["n_images_enc"])["target"].mean()
    )
    train["image_size_mean_enc"] = train["image_size_enc"].map(
        train.groupby(["image_size_enc"])["target"].mean()
    )
    train["anatom_mean_enc"] = train["anatom_enc"].map(
        train.groupby(["anatom_enc"])["target"].mean()
    )

    return train
