import librosa
import os
from collections import Counter
import soundfile as sf

directory = "downloads"
sample_rates = []

print("Sample Rates of Audio Files:")
# Iterate through all files in the directory and then rewrite the sample rate
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        path = os.path.join(directory, filename)
        _, sr = librosa.load(path, sr=None)
        print(f"{filename}: {sr} Hz")
        sample_rates.append(sr)

# Rewrite the sample rate
for i, sr in enumerate(sample_rates):
    if sr != 44100:
        # Resample the audio file
        path = os.path.join(directory, os.listdir(directory)[i])
        y, _ = librosa.load(path, sr=sr)
        sf.write(path, y, 44100)
        print(f"Resampled {os.listdir(directory)[i]} to 44100 Hz")
        sample_rates[i] = 44100
        # Update the sample rate in the list
    else:
        print(f"{os.listdir(directory)[i]} is already at 44100 Hz")

# Print a summary of the sample rates
print("\nSample Rate Summary:")
sr_counts = Counter(sample_rates)
for sr, count in sorted(sr_counts.items()):
    print(f"{sr} Hz: {count} files")