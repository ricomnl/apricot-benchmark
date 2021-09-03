import requests

import pandas as pd
import streamlit as st
import yaml

from pmlb import fetch_data

st.set_page_config(
    page_title="Penn Machine Learning Benchmarks",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache
def load_data(dataset_name):
    return fetch_data(dataset_name, local_cache_dir="datasets")

@st.cache()
def load_metadata(dataset_name):
    link = f"https://raw.githubusercontent.com/EpistasisLab/pmlb/master/datasets/{dataset_name}/metadata.yaml"
    f = requests.get(link)
    return yaml.safe_load(f.text)

def main():
    st.title("Penn Machine Learning Benchmarks")

    pmlb_data = pd.read_csv("metadata/Penn Machine Learning Benchmarks.csv")
    # 417 rows
    pmlb_data = pmlb_data[pmlb_data["n_observations"] > 100]
    # 384 rows
    pmlb_data = pmlb_data[~pmlb_data["Dataset"].str.contains("feynman")]
    # 26 rows
    pmlb_data = pmlb_data.reset_index(drop=True)
    pmlb_data.loc[pmlb_data["n_classes"] == 2, "Endpoint"] = "binary"

    st.sidebar.title("Pick a dataset")

    tasks = list(pmlb_data["Task"].unique())
    task_filter = st.sidebar.multiselect("Task", options=tasks, default=tasks)

    filtered_data = pmlb_data[pmlb_data["Task"].isin(set(task_filter))]
    dataset_name = st.sidebar.selectbox("Dataset", options=filtered_data["Dataset"].values)

    data = load_data(dataset_name)

    st.subheader("Datasets")
    if not filtered_data.empty:
        st.dataframe(filtered_data)

    if not data.empty:
        st.subheader("Describe")
        st.dataframe(data.describe().transpose())

        st.subheader("Visualize")
        st.dataframe(data.head(20))

        st.subheader("Metadata")
        metadata = load_metadata(dataset_name)
        st.write(metadata)

if __name__ == "__main__":
    main()