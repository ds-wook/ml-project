# %%
import warnings

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Set Color Palettes for the notebook
colors_nude = ["#e0798c", "#65365a", "#da8886", "#cfc4c4", "#dfd7ca"]
sns.palplot(sns.color_palette(colors_nude))

# Set Style
sns.set_style("whitegrid")
sns.despine(left=True, bottom=True)
# %%
directory = "../input/siim-isic-melanoma-classification/"
train_df = pd.read_csv(directory + "train.csv")
path_train = directory + "jpeg/train/" + train_df["image_name"] + ".jpg"
train_df["path_jpeg"] = path_train
train_df.head()

# %%
image_list = train_df["path_jpeg"].values.tolist()
image_name = train_df["image_name"].values.tolist()

# %%
for name, path in tqdm(zip(image_name, image_list)):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    val = 50
    array = np.full(image.shape, (val, val, val), dtype=np.uint8)
    image = cv2.add(image, array)

    # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0) ,256/10), -4, 128)

    p = "../res/bright/" + name + ".jpeg"
    cv2.imwrite(p, image)

# %%
