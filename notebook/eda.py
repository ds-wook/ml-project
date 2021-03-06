# %%

import os
from os import listdir
from typing import Optional

import chart_studio.plotly as py
import cufflinks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import plotly.express as px
import plotly.graph_objs as go
import pydicom
from PIL import Image, ImageDraw
from plotly.offline import iplot
from scipy.stats import boxcox
from skimage.io import imread
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler

cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme="pearl")

import seaborn as sns

sns.set(style="whitegrid")

import warnings

warnings.filterwarnings("ignore")


plt.style.use("fivethirtyeight")
plt.show()


# %%

path = "../input/siim-isic-melanoma-classification/"
train = pd.read_csv(path + "train.csv")
print("Shape of train :", train.shape)


# %% [markdown]


"""
+ image_name: 이미지 파일 이름.
+ patient_id: 환자 개인의 고유값.
+ sex: 성별(Nan값은 성별을 알 수 없음)
+ age_approx: 환자 나이.
+ anatom_site_general_challenge: 흑색종 위치
+ diagnosis: maligant의 진단명
+ benign_malignant: target값의 실제 명
"""

# 결측값 확인
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum() / train.isnull().count() * 100).sort_values(
    ascending=False
)
missing_train_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
missing_train_data.head()
# %%


def plot_count(df: pd.DataFrame, feature: str, title: str = "", size: float = 2.5):
    f, ax = plt.subplots(1, 1, figsize=(4 * size, 3 * size))
    total = float(len(df))
    sns.countplot(df[feature], order=df[feature].value_counts().index, palette="Set2")
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2.0,
            height + 3,
            "{:1.2f}%".format(100 * height / total),
            ha="center",
        )
    plt.show()


# %%
plot_count(train, "benign_malignant")

# %%
plot_count(train, "sex")

# %%
plot_count(train, "anatom_site_general_challenge")

# %%
train["diagnosis"].value_counts(normalize=True).sort_values().iplot(
    kind="barh",
    xTitle="Percentage",
    linecolor="black",
    opacity=0.7,
    color="blue",
    theme="pearl",
    bargap=0.2,
    gridcolor="white",
    title="Distribution in the training set",
)

# %%


def plot_relative_distribution(
    df: pd.DataFrame, feature: str, hue: str, title: str = "", size: float = 2
):
    f, ax = plt.subplots(1, 1, figsize=(4 * size, 3 * size))
    total = float(len(df))
    sns.countplot(x=feature, hue=hue, data=df, palette="Set2")
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2.0,
            height + 3,
            "{:1.2f}%".format(100 * height / total),
            ha="center",
        )
    plt.show()


# %%
plot_relative_distribution(
    df=train,
    feature="sex",
    hue="benign_malignant",
    title="relative count plot of sex with benign_malignant",
    size=2.8,
)

# %%
plot_relative_distribution(
    df=train,
    feature="anatom_site_general_challenge",
    hue="benign_malignant",
    title="relative count plot of anatom_site_general_challenge with benign_malignant",
    size=3,
)

# %%
train["age_approx"].iplot(
    kind="hist",
    bins=30,
    color="blue",
    xTitle="Age",
    yTitle="Count",
    title="Age Distribution",
)


def display_images(images: np.ndarray, title: Optional[str] = None):
    f, ax = plt.subplots(5, 3, figsize=(18, 22))
    if title:
        f.suptitle(title, fontsize=30)

    for i, image_id in enumerate(images):
        image_path = os.path.join(path, f"jpeg/train/{image_id}.jpg")
        image = Image.open(image_path)

        ax[i // 3, i % 3].imshow(image)
        image.close()
        ax[i // 3, i % 3].axis("off")

        benign_malignant = train[train["image_name"] == image_id][
            "benign_malignant"
        ].values[0]
        ax[i // 3, i % 3].set_title(
            f"image_name: {image_id}\nSource: {benign_malignant}", fontsize="15"
        )

    plt.show()


# %%
df = train.groupby(["age_approx", "target"])["target"].count().unstack(level="target")
df.plot(kind="bar", figsize=(12, 10), title="Age Target")
plt.show()
# %%
df["age_approx"]
# %%
df = train.groupby(["age_approx", "target"])["target"].agg(["count"])
df["percent"] = df.groupby(level=0).transform(lambda x: (x / x.sum()))
df = df.unstack(level="target")
df["percent"].plot(kind="bar", figsize=(12, 10), title="Age Target")
plt.show()
# %%
df = train.groupby(["age_approx", "target"])["target"].agg(["count"])
df["percent"] = df.groupby(level=0).transform(lambda x: (x / x.sum()))
df
# %%
