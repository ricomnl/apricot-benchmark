import pandas as pd

from pmlb import fetch_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

pmlb_data = pd.read_csv("metadata/Penn Machine Learning Benchmarks.csv")
# 417 rows
pmlb_data = pmlb_data[pmlb_data["n_observations"] > 100]
# 146 rows
pmlb_data = pmlb_data[~pmlb_data["Dataset"].str.contains("feynman")]
# 26 rows
pmlb_data = pmlb_data.reset_index(drop=True)
# 13 classification, 13 regression
# len(pmlb_data[pmlb_data["Task"] == "classification"])
cls_dataset_names = pmlb_data[pmlb_data["Task"] == "classification"]["Dataset"].values

logit_test_scores = {}

for cls_dataset in cls_dataset_names:
    # Read in the datasets and split them into training/testing
    X, y = fetch_data(cls_dataset, return_X_y=True, local_cache_dir="datasets")
    train_X, test_X, train_y, test_y = train_test_split(X, y)

    lr = LogisticRegression()
    lr.fit(train_X, train_y)

    # Log the performance score on the test set
    logit_test_scores[cls_dataset] = lr.score(test_X, test_y)

data = fetch_data(cls_dataset_names[0])
data.describe().transpose()