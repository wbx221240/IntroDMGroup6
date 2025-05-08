import pandas as pd
import numpy as np
import json
import os

def label_merge():
    with open("./datafile.json", "r") as file:
        files = json.load(file)
    for i, (file, anomaly) in enumerate(files.items()):
        print(f"----Loading {file} ----")
        df = pd.read_csv(os.path.join("./data/", file))
        df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y-%m-%d %H:%M:%S')
        df["label"] = df["timestamp"].isin(anomaly)
        print(f"Anomaly Points: {df["label"].sum()}")
        df.to_csv(os.path.join("./data", file))
    

if __name__ == "__main__":
    label_merge()