import os
import json
import requests
# Get labels
import pandas as pd

file = pd.read_csv("recordings_data.csv")
labels = file["english_name"].unique().tolist()
labels = [label for label in labels if label != "Unknown"]
print(f"Found {len(labels)} unique labels.")
print("Labels:", labels)