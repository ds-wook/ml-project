# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()

import plotly.offline as py

py.init_notebook_mode(connected=True)
from os import listdir

import plotly.graph_objects as go
import pydicom
from PIL import Image
from scipy.stats import boxcox
from skimage.io import imread
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler

listdir("../input/")

# %%
base_path = "../input/siim-isic-melanoma-classification/"
train_image_path = base_path + "train/"

# %%
train = pd.read_csv(base_path + "train.csv")
train.head()
# %%
train["dcm_path"] = train_image_path + train.image_name + ".dcm"
train.head()
# %%
train.shape

# %%[markdown]
"""
## Image Statistics data
"""

from scipy.stats import skew
from tqdm.notebook import tqdm


def extract_shapes(df: pd.DataFrame) -> pd.DataFrame:
    all_paths = df.dcm_path.values
    image_eda = pd.DataFrame(
        index=np.arange(len(df)),
        columns=[
            "path",
            "rows",
            "columns",
            "channels",
            "img_mean",
            "img_std",
            "img_skew",
            "red_mean",
            "green_mean",
            "blue_mean",
        ],
    )

    for i in tqdm(range(len(df))):
        path = all_paths[i]
        dcm_file = pydicom.dcmread(path)
        image = dcm_file.pixel_array

        image_eda.iloc[i]["path"] = path
        image_eda.iloc[i]["rows"] = image.shape[0]
        image_eda.iloc[i]["columns"] = image.shape[1]
        image_eda.iloc[i]["channels"] = image.shape[2]

        image_eda.iloc[i]["img_mean"] = np.mean(image.flatten())
        image_eda.iloc[i]["img_std"] = np.std(image.flatten())
        image_eda.iloc[i]["img_skew"] = skew(image.flatten())

        image_eda.iloc[i]["red_mean"] = np.mean(image[:, :, 0].flatten())
        image_eda.iloc[i]["green_mean"] = np.mean(image[:, :, 1].flatten())
        image_eda.iloc[i]["blue_mean"] = np.mean(image[:, :, 2].flatten())

    return image_eda


# %%

# train_shapes = extract_shapes(train)
# train_shapes.to_csv("../res/train_stats_and_meta.csv")

train_image_stats = pd.read_csv("../res/train_stats_and_meta.csv")
train_image_stats.head()

# %%

train_image_names = train_image_stats.image_name.values
train_image_stats["img_area"] = train_image_stats["rows"] * train_image_stats["columns"]

# %%

train_image_stats.head()

# %%

jpeg_path = base_path + "/jpeg/train/"

fig, ax = plt.subplots(3, 2, figsize=(20, 15))
for n in range(3):
    dcm_file = pydicom.dcmread(train.dcm_path.values[n])
    pixel_array = dcm_file.pixel_array

    image_path = jpeg_path + train.image_name.values[n] + ".jpg"
    image = imread(image_path)

    sns.histplot(pixel_array.flatten(), ax=ax[n, 0], color="sienna", kde=True)
    ax[n, 0].set_title("Distribution of values in dicom pixelarrays")

    sns.histplot(image.flatten(), ax=ax[n, 1], color="deepskyblue", kde=True)
    ax[n, 1].set_title("Distribution of values in jpeg images")

# %%

fig, ax = plt.subplots(1, 1, figsize=(20, 5))
ax.scatter(
    train_image_stats["rows"].values, train_image_stats["columns"].values, c="orangered"
)
ax.set_title("Images")
ax.set_xlabel("row value")
ax.set_ylabel("column value")
ax.set_xlim([0, 6500])
ax.set_ylim([0, 6500])

# %%

from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=1)


trace0 = go.Scatter(
    x=train_image_stats.img_std.values,
    y=train_image_stats.img_mean.values,
    mode="markers",
    text=train_image_stats["rows"].values,
    marker=dict(
        color=train_image_stats["columns"].values,
        colorscale="Jet",
        opacity=0.4,
        size=4,
        colorbar=dict(thickness=10, len=1.1, title="image columns"),
    ),
)

fig.add_trace(trace0, row=1, col=1)
fig.update_xaxes(title_text="Image std", row=1, col=1)
fig.update_yaxes(title_text="Image mean", row=1, col=1)
fig.update_xaxes(title_text="Image std", row=1, col=2)
fig.update_yaxes(title_text="Image mean", row=1, col=2)

fig.update_layout(height=425, width=850, showlegend=False)
fig.show()

# %%


plt.figure(figsize=(12, 9))
sns.histplot(train_image_stats.img_area, bins=30, color="orangered")
plt.xlabel("image area")
plt.ylabel("count")
plt.title("Image Area distribution")
plt.show()


# %%

fig, ax = plt.subplots(2, 3, figsize=(20, 12))

sns.histplot(train_image_stats.img_mean, ax=ax[0, 0], color="lightseagreen", kde=True)
sns.histplot(train_image_stats.img_std, ax=ax[0, 1], color="lightseagreen", kde=True)
sns.histplot(train_image_stats.img_skew, ax=ax[0, 2], color="lightseagreen", kde=True)


sns.histplot(train_image_stats.red_mean, ax=ax[1, 0], color="lightseagreen", kde=True)
sns.histplot(train_image_stats.green_mean, ax=ax[1, 1], color="lightseagreen", kde=True)
sns.histplot(train_image_stats.blue_mean, ax=ax[1, 2], color="lightseagreen", kde=True)


for n in range(3):
    for m in range(2):
        ax[m, n].set_ylabel("density")

ax[0, 0].set_title("Image means")
ax[0, 1].set_title("Image stds")
ax[0, 2].set_title("Image skewnesses")

ax[1, 0].set_title("Red channel mean")
ax[1, 1].set_title("Green channel mean")
ax[1, 2].set_title("Blue channel mean")

# %%


def preprocess_k_means(
    data: pd.DataFrame, feature: str, constant: int, lam: int
) -> pd.DataFrame:
    min_max_scaler = MinMaxScaler()
    scaled_train_feature = min_max_scaler.fit_transform(
        data[feature].values.reshape(-1, 1)
    )

    boxcox_train_feature = boxcox(scaled_train_feature[:, 0] + constant, lam)

    scaler = StandardScaler()
    preprocessed_train_feature = scaler.fit_transform(
        boxcox_train_feature.reshape(-1, 1)
    )

    data.loc[:, "preprocessed_" + feature] = preprocessed_train_feature
    return data


# %%
image_stats = preprocess_k_means(train_image_stats, "red_mean", constant=1, lam=10)
image_stats = preprocess_k_means(train_image_stats, "green_mean", constant=0.5, lam=2)
image_stats = preprocess_k_means(train_image_stats, "img_skew", constant=0.05, lam=2)

# %%
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

sns.histplot(
    train_image_stats.preprocessed_red_mean, ax=ax[0], color="lightseagreen", kde=True
)

sns.histplot(
    train_image_stats.preprocessed_green_mean, ax=ax[1], color="lightseagreen", kde=True
)

sns.histplot(
    train_image_stats.preprocessed_img_skew, ax=ax[2], color="lightseagreen", kde=True
)


for n in range(3):
    ax[n].set_ylabel("density")

ax[0].set_title("Image means")
ax[1].set_title("Image stds")
ax[2].set_title("Image skewnesses")

# %%
fig = make_subplots(
    rows=1,
    cols=1,
    subplot_titles=("Preprocessed image stats", "Preprocessed image stats"),
)

trace = go.Scatter(
    x=train_image_stats.img_mean.values,
    y=train_image_stats.img_std.values,
    mode="markers",
    text=train_image_stats["columns"].values,
    marker=dict(
        color=train_image_stats.preprocessed_img_skew.values,
        colorbar=dict(thickness=10, len=1.1, title="preprocessed skewness"),
        colorscale="Jet",
        opacity=0.4,
        size=2,
    ),
)

fig.add_trace(trace, row=1, col=1)

fig.update_xaxes(title_text="Image std", row=1, col=1)
fig.update_yaxes(title_text="Image mean", row=1, col=1)
fig.update_xaxes(title_text="Image std", row=1, col=2)
fig.update_yaxes(title_text="Image mean", row=1, col=2)

fig.update_layout(height=425, width=850, showlegend=False)
fig.show()


# %%

from sklearn.metrics import silhouette_score

x = train_image_stats.loc[
    :,
    [
        "img_mean",
        "img_std",
        "preprocessed_img_skew",
        "preprocessed_red_mean",
        "preprocessed_green_mean",
        "blue_mean",
    ],
].values
distortions = []
silhouettes = []
K = range(2, 11)
for k in K:
    kmean_model = KMeans(n_clusters=k)
    kmean_model.fit(x)
    cluster_labels = kmean_model.fit_predict(x)
    train_image_stats["cluster_label"] = cluster_labels
    distortions.append(kmean_model.inertia_)
    silhouettes.append(silhouette_score(x, train_image_stats.cluster_label))


# %%

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
ax1.set_xlabel("K")
ax1.set_ylabel("Distortion")
ax1.plot(K, distortions, "bx-", label="Distortion")

ax2.set_ylabel("Silhouettes")

ax2.plot(K, silhouettes, "rx-", label="silhouette")

ax1.set_title("The Elbow Method showing the optimal k")
ax2.set_title("Silhoutte")
plt.show()
# %%

num_clusters = 5

kmeans = KMeans(n_clusters=num_clusters, random_state=0)

x = train_image_stats.loc[
    :,
    [
        "img_mean",
        "img_std",
        "preprocessed_img_skew",
        "preprocessed_red_mean",
        "preprocessed_green_mean",
        "blue_mean",
    ],
].values

cluster_labels = kmeans.fit_predict(x)
train_image_stats["cluster_label"] = cluster_labels

# %%

fig = make_subplots(rows=1, cols=1, subplot_titles=("image stats", "image stats"))


trace = go.Scatter(
    x=train_image_stats.img_std.values,
    y=train_image_stats.img_mean.values,
    mode="markers",
    text=train_image_stats["cluster_label"].values,
    marker=dict(
        color=train_image_stats.cluster_label.values,
        colorbar=dict(thickness=10, len=1.1, title="cluster label"),
        colorscale="Jet",
        opacity=0.4,
        size=2,
    ),
)


fig.add_trace(trace, row=1, col=1)

fig.update_xaxes(title_text="Image std", row=1, col=1)
fig.update_yaxes(title_text="Image mean", row=1, col=1)
fig.update_xaxes(title_text="Image std", row=1, col=2)
fig.update_yaxes(title_text="Image mean", row=1, col=2)

fig.update_layout(height=425, width=425, showlegend=False)
fig.show()

# %%

fig, ax = plt.subplots(num_clusters, 8, figsize=(20, 2.5 * num_clusters))

for cluster in range(num_clusters):
    selection = np.random.choice(
        train_image_stats[train_image_stats.cluster_label == cluster].image_path.values,
        size=8,
        replace=False,
    )
    m = 0
    for path in selection:
        image = imread(path)
        ax[cluster, m].imshow(image)
        ax[cluster, m].set_title("K-Means cluster {}".format(cluster))
        ax[cluster, m].axis("off")
        m += 1

# %%

df = train_image_stats.groupby(["cluster_label", "target"])["target"].agg(["count"])
df["percent"] = df.groupby(level=0).transform(lambda x: (x / x.sum()))
df

# %%
num_clusters = 5

kmeans = KMeans(n_clusters=num_clusters, random_state=0)

x = train_image_stats.loc[
    :,
    [
        "img_mean",
        "img_std",
        "preprocessed_img_skew",
        "preprocessed_red_mean",
        "preprocessed_green_mean",
        "blue_mean",
    ],
].values

cluster_labels = kmeans.fit_predict(x)
train_image_stats["cluster_label"] = cluster_labels

# %%

fig = make_subplots(rows=1, cols=1, subplot_titles=("image stats", "image stats"))


trace = go.Scatter(
    x=train_image_stats.img_std.values,
    y=train_image_stats.img_mean.values,
    mode="markers",
    text=train_image_stats["cluster_label"].values,
    marker=dict(
        color=train_image_stats.cluster_label.values,
        colorbar=dict(thickness=10, len=1.1, title="cluster label"),
        colorscale="Jet",
        opacity=0.4,
        size=2,
    ),
)


fig.add_trace(trace, row=1, col=1)

fig.update_xaxes(title_text="Image std", row=1, col=1)
fig.update_yaxes(title_text="Image mean", row=1, col=1)
fig.update_xaxes(title_text="Image std", row=1, col=2)
fig.update_yaxes(title_text="Image mean", row=1, col=2)

fig.update_layout(height=425, width=850, showlegend=False)
fig.show()

# %%

fig, ax = plt.subplots(num_clusters, 8, figsize=(20, 2.5 * num_clusters))

for cluster in range(num_clusters):
    selection = np.random.choice(
        train_image_stats[train_image_stats.cluster_label == cluster].image_path.values,
        size=8,
        replace=False,
    )
    m = 0
    for path in selection:
        image = imread(path)
        ax[cluster, m].imshow(image)
        ax[cluster, m].set_title("K-Means cluster {}".format(cluster))
        ax[cluster, m].axis("off")
        m += 1

# %%

df = train_image_stats.groupby(["cluster_label", "target"])["target"].agg(["count"])
df["percent"] = df.groupby(level=0).transform(lambda x: (x / x.sum()))
df

# %%
