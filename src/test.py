import json
from pathlib import Path

from apricot.functions import FeatureBasedSelection
import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils import *

DATASET_CACHE = Path(__file__).resolve().parent.parent.joinpath("datasets")

pmlb_data = pd.read_csv("metadata/pmlb_data_processed.csv")

# dataset_names = pmlb_data["Dataset"].values
dataset_names = pmlb_data[(pmlb_data["n_observations"] > 10_000) & (pmlb_data["Task"] == "classification")]["Dataset"].values

results = []

for dataset in dataset_names:
    print(f"Dataset: {dataset}")
    m = load_metadata(dataset)
    X = fetch_data(dataset, local_cache_dir=DATASET_CACHE)
    categorical_fs = [f["name"] for f in m["features"] if f["type"] == "categorical"]
    continuous_fs = [f["name"] for f in m["features"] if f["type"] == "continuous"]

    # OHE categorical features
    X = one_hot_encode_df(X, columns=categorical_fs)
    # Standardize continuous features
    X.loc[:, continuous_fs] = normalize_df(X, columns=continuous_fs)

    y = X["target"]
    X = X.drop("target", axis=1)

    # Split the data into training/testing
    train_X, test_X, train_y, test_y = train_test_split(X.values, y.values)

    results_row = {}
    # Train on full dataset
    model = LogisticRegression(max_iter=1000)
    model.fit(train_X, train_y)
    results_row["dataset"] = dataset
    results_row["fraction"] = 1.0
    results_row["score"] = model.score(test_X, test_y)
    # Count the prevalence of the most and least representative classes
    (_, counts) = np.unique(y, return_counts=True)
    results_row["most_prevalent_cls"] = np.max(counts/len(y))
    results_row["least_prevalent_cls"] = np.min(counts/len(y))
    results.append(results_row)

    # TODO: use sklearn function to combine possibilities
    for opt in ["naive", "lazy", "stochastic"]:
        for n_subset in [0.001]:
            print(f"Optimizer: {opt}\nSample fraction: {n_subset}")
            results_row = {}
            results_row["dataset"] = dataset
            results_row["fraction"] = n_subset
            results_row["optimizer"] = opt

            fb_select = FeatureBasedSelection(n_samples=len(train_X)*n_subset, optimizer=opt)
            fb_select.fit(train_X)
            train_X_subset = train_X[fb_select.ranking, :]
            train_y_subset = train_y[fb_select.ranking]

            # Train on subset
            model = LogisticRegression(max_iter=10000, n_jobs=-1)
            model.fit(train_X_subset, train_y_subset)
            results_row["score"] = model.score(test_X, test_y)

            # Count the prevalence of the most and least representative classes
            (_, counts) = np.unique(train_y_subset, return_counts=True)
            results_row["most_prevalent_cls"] = np.max(counts/len(train_y_subset))
            results_row["least_prevalent_cls"] = np.min(counts/len(train_y_subset))
            results.append(results_row)

    json.dump(results, open("results.json", "w"))