import os
import json
import requests
# Get labels
import pandas as pd

file = pd.read_csv("recordings_data.csv")
labels = file["english_name"].unique().tolist()
# Remove "New Zealand", "North Island", and "South Island" from labels
for i in range(len(labels)):
    labels[i] = labels[i].replace("New Zealand ", "").replace("North Island ", "").replace("South Island ", "")
print("Labels:", labels)

# Count for each label
label_counts = file["english_name"].value_counts()
label_counts = label_counts[label_counts.index != "Unknown"]
# Write in order of count (descending) (label, count, followed by generic_name + scientific_name)
with open("label_counts.txt", "w") as f:
    for label, count in label_counts.items():
        if count < 5:
            continue
        generic_name = file[file["english_name"] == label]["generic_name"].values[0]
        scientific_name = file[file["english_name"] == label]["scientific_name"].values[0]
        f.write(f"{label}: {count} ({generic_name} {scientific_name})\n")
        print(f"{label}: {count}")

# If count < 5, don't include in labels
