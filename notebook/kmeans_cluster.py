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
some_files = listdir(train_image_path)[:5]
some_files
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
extract_train0 = False
extract_train1 = False
extract_train2 = False

# %%
from tqdm.notebook import tqdm
from scipy.stats import skew


def extract_shapes(df: pd.DataFrame) -> pd.DataFrame:
    all_path = df.dcm_path.values
    image_eda = pd.DataFrame(index=np.arange(len(df)), columns=)