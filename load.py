import pandas as pd

# Load the dataset
file_path = "E:/Main/data/dataset.csv"
df = pd.read_csv(file_path)

# Assuming the dataset has two columns: "query" (SQL queries) and "label" (1=malicious, 0=normal)
queries = df["query"].tolist()
labels = df["label"].tolist()
