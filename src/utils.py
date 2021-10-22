import requests
import yaml

import pandas as pd


def load_metadata(dataset_name):
    link = f"https://raw.githubusercontent.com/EpistasisLab/pmlb/master/datasets/{dataset_name}/metadata.yaml"
    f = requests.get(link)
    return yaml.safe_load(f.text)


mean_norm = lambda df: (df-df.mean())/df.std()
minmax_norm = lambda df: (df-df.min())/(df.max()-df.min())
min_zero = lambda df: df-df.min()


def one_hot_encode_df(df, columns):
    return pd.get_dummies(df, columns=columns)


def normalize_df(df, columns, method="minmax"):
    if method == "minmax": f = minmax_norm
    elif method == "mean": f = mean_norm
    elif method == "minzero": f = min_zero
    return f(df.loc[:, columns])
