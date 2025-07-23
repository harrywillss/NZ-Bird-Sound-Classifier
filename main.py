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
with open("label_counts.txt", "w") as f:
    for label in labels:
        count = label_counts.get(label, 0)
        f.write(f"{label}: {count}\n")
        print(f"{label}: {count}")


# If count < 5, don't include in labels