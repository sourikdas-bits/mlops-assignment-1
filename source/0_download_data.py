#!/usr/bin/env python
from ucimlrepo import fetch_ucirepo
from tqdm import tqdm
import pandas as pd

def download():
    """Download the Heart Disease dataset from UCI Machine Learning Repository and save it as a CSV file."""

    print("Downloading dataset from UCI Machine Learning Repository...")
    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features
    y = dataset.data.targets["num"].apply(lambda x: 1 if x > 0 else 0)

    X["target"] = y

    print("Saving dataset to CSV with progress bar...")
    # Use tqdm to show progress while saving rows to CSV
    with tqdm(total=len(X), desc="Saving", unit="rows") as pbar:
        # Save in chunks to show progress
        chunk_size = 1000
        for i in range(0, len(X), chunk_size):
            mode = 'w' if i == 0 else 'a'
            header = i == 0
            X.iloc[i:i+chunk_size].to_csv("./data/heart.csv", index=False, mode=mode, header=header)
            pbar.update(min(chunk_size, len(X) - i))

    print("Dataset saved to data/heart.csv")

if __name__ == "__main__":
    download()
