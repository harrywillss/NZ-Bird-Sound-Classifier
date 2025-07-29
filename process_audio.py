import os
import shutil
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf

# Display imports
import matplotlib.pyplot as plt
from collections import Counter
import soundfile as sf

# def segment_audio_snr(audio, sr, segment_len_ms=2500, hop_len_ms=1000, noise_len_ms=500, snr_threshold=5):
#     segment_len_samples = int(sr * segment_len_ms / 1000)
#     hop_len_samples = int(sr * hop_len_ms / 1000)
#     noise_len_samples = int(sr * noise_len_ms / 1000)

#     def get_noise_level(samples):
#         abs_max = []
#         if len(samples) > noise_len_samples:
#             for i in range(0, len(samples) - noise_len_samples, noise_len_samples):
#                 abs_max.append(np.max(np.abs(samples[i:i + noise_len_samples])))
#         else:
#             abs_max.append(np.max(np.abs(samples)))
#         return min(abs_max) if abs_max else 1e-6

#     noise_level = get_noise_level(audio)
#     segments = []
#     for i in range(0, len(audio) - segment_len_samples, hop_len_samples):
#         segment = audio[i:i + segment_len_samples]
#         seg_abs_max = np.max(np.abs(segment))
#         snr = seg_abs_max / noise_level if noise_level != 0 else 0
#         if snr > snr_threshold:
#             hann = np.hanning(len(segment))
#             windowed = segment * hann
#             spectrum = np.abs(np.fft.rfft(windowed))
#             segments.append(segment)
#     return segments

# def load_audio_data(data_dir="resampled", duration=3, save_segments=True, segments_dir="segments"):
#     """
#     Loads audio files from the specified directory, detects onsets, and extracts clips of a specified duration.
#     """
#     if not os.path.exists(data_dir):
#         print(f"Directory {data_dir} does not exist.")

#     resample_audio()

#     audio_clips = []
#     labels = []
#     total_files = 0
#     processed_files = 0

#     # Create segments directory if saving segments
#     if save_segments and not os.path.exists(segments_dir):
#         os.makedirs(segments_dir)

#     print(f"Walking through directory: {data_dir}")
#     # Recursively walk through all subfolders and process .wav files
#     for root, dirs, files in os.walk(data_dir):
#         wav_files = [f for f in files if f.endswith('.wav')]
#         if wav_files:
#             print(f"Found {len(wav_files)} .wav files in {root}")
#             total_files += len(wav_files)
        
#         for file in wav_files:
#             file_path = os.path.join(root, file)
#             # Use the English name as the label (parent folder of scientific name)
#             label = os.path.basename(os.path.dirname(root))
#             print(f"Processing {file}, label: {label}")
            
#             # Load the audio file
#             audio, sr = librosa.load(file_path, sr=None)
#             print(f"  Audio length: {len(audio)/sr:.2f}s, sample rate: {sr}")
            
#             # Use more flexible segmentation - lower SNR threshold and shorter segments
#             segments = segment_audio_snr(audio, sr, segment_len_ms=3000, snr_threshold=2)
#             print(f"  Found {len(segments)} segments")
            
#             # If no segments found, use simple fixed-duration chunking as fallback
#             if len(segments) == 0:
#                 print("  No SNR segments found, using fixed chunking")
#                 chunk_size = int(duration * sr)
#                 for start in range(0, len(audio) - chunk_size, chunk_size // 2):  # 50% overlap
#                     chunk = audio[start:start + chunk_size]
#                     if len(chunk) == chunk_size:
#                         segments.append(chunk)
#                 print(f"  Created {len(segments)} fixed chunks")
            
#             for i, clip in enumerate(segments):
#                 # More flexible duration check (accept 50% to 150% of target)
#                 expected_length = duration * sr
#                 clip_duration = len(clip) / sr
#                 print(f"    Segment {i}: {clip_duration:.2f}s")
                
#                 if len(clip) >= expected_length * 0.5:
#                     # Pad or trim to exact duration
#                     if len(clip) < expected_length:
#                         clip = np.pad(clip, (0, expected_length - len(clip)), mode='constant')
#                     elif len(clip) > expected_length:
#                         clip = clip[:expected_length]
                    
#                     audio_clips.append(clip)
#                     labels.append(label)
#                     print(f"    Added segment {i} to dataset")
                    
#                     # Save segment if requested
#                     if save_segments:
#                         # Create nested structure in segments folder
#                         english_name = os.path.basename(os.path.dirname(root))
#                         scientific_name = os.path.basename(root)
#                         segment_folder = os.path.join(segments_dir, english_name, scientific_name)
#                         os.makedirs(segment_folder, exist_ok=True)
                        
#                         # Save segment with original filename + segment number
#                         segment_filename = f"{file[:-4]}_segment_{i}.wav"
#                         segment_path = os.path.join(segment_folder, segment_filename)
#                         sf.write(segment_path, clip, sr)
            
#             processed_files += 1

#     print(f"Total files found: {total_files}, processed: {processed_files}")
#     print(f"Total segments extracted: {len(audio_clips)}")
#     return audio_clips, labels

# # Create features using Mel-Spectrogram
# def extract_features(audio_files):
#     features = [librosa.feature.melspectrogram(y=audio) for audio in audio_files]
#     return features

# # Create image patches with the positional encodings
# def create_image_patches_with_position(
#     features, labels, patch_size=4, height=128, width=188, expected_sequence_length=1504
# ):
#     """
#     Creates image patches and appends grid positional encodings.
#     """
#     patches = []

#     # Calculate grid size
#     grid_height = height // patch_size
#     grid_width = width // patch_size

#     # Create positional encodings
#     pos_enc_height = np.repeat(
#         np.arange(grid_height)[:, np.newaxis], grid_width, axis=1
#     )
#     pos_enc_width = np.repeat(np.arange(grid_width)[np.newaxis, :], grid_height, axis=0)

#     sequences = []

#     for idx, feature in enumerate(features):
#         current_sequence = []

#         for i in range(0, height, patch_size):
#             for j in range(0, width, patch_size):
#                 patch = feature[i : i + patch_size, j : j + patch_size]

#                 # Get positional encoding for current patch
#                 pos_enc = np.array(
#                     [
#                         pos_enc_height[i // patch_size, j // patch_size],
#                         pos_enc_width[i // patch_size, j // patch_size],
#                     ]
#                 )

#                 # Append positional encoding to the patch
#                 patch_with_pos = np.concatenate([patch.flatten(), pos_enc])

#                 current_sequence.append(patch_with_pos)

#         # Pad sequence if needed
#         while len(current_sequence) < expected_sequence_length:
#             current_sequence.append(np.zeros_like(patch_with_pos))

#         sequences.append(current_sequence)

#     return np.array(sequences), labels

# def build_transformer_model(
#     input_shape, num_classes, num_heads, ff_dim, num_transformer_blocks
# ):
#     """
#     Builds a transformer-based model for processing image patches.
#     """
#     inputs = tf.keras.Input(shape=input_shape)

#     # Start with a Linear layer followed by Batch Normalization as shown in the diagram
#     x = tf.keras.layers.Dense(units=18, activation="relu")(inputs)
#     x = tf.keras.layers.BatchNormalization()(x)

#     # Transformer encoder
#     for _ in range(num_transformer_blocks):
#         # Multi-head attention
#         attention_output = tf.keras.layers.MultiHeadAttention(
#             num_heads=num_heads, key_dim=18, dropout=0.1
#         )(x, x)
#         x = tf.keras.layers.Add()([x, attention_output])
#         x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

#         # Feed-forward layer
#         ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
#         ffn_output = tf.keras.layers.Dense(18)(ffn_output)
#         x = tf.keras.layers.Add()([x, ffn_output])
#         x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

#     # Mean over the sequence length (across the time axis of the Mel-Spectrogram)
#     avg_out = tf.keras.layers.GlobalAveragePooling1D()(x)

#     # Classification head with Linear layer followed by a Cross-Entropy loss during training
#     outputs = tf.keras.layers.Dense(num_classes)(avg_out)

#     return tf.keras.Model(inputs=inputs, outputs=outputs)

# def resample_audio(download_dir="downloads", resampled_dir="resampled"):
#     """
#     For each .wav file in download_dir, resample to 44100 Hz if needed, and copy to resampled_dir
#     in subfolders for both English and scientific names.
#     """
#     if input("Do you want to resample audio files? (y/n): ").strip().lower() != 'y':
#         print("Skipping audio resampling.")
#         return
#     if not os.path.exists(resampled_dir):
#         os.makedirs(resampled_dir)
#     else:
#         print(f"Directory {resampled_dir} already exists. Deleting entire folder.")
#         shutil.rmtree(resampled_dir)
#     filenames = [f for f in os.listdir(download_dir) if f.endswith(".wav")]
#     for filename in filenames:
#         # Construct full path
#         path = os.path.join(download_dir, filename)
#         # Parse filename: e.g. '33763_tui_prosthemadera_novaeseelandiae_song.wav' -> "{file_id}_{english_name}_{scientific_name}.wav"
#         parts = filename[:-4].split('_')  # remove .wav, split
#         if len(parts) < 3:
#             print(f"Filename format not recognized: {filename}")
#             continue
#         file_id = parts[0]
#         english_name = parts[1]
#         scientific_name = '_'.join(parts[2:-1])
#         scientific_name_pretty = ' '.join([p.capitalize() for p in parts[2:-1]])
#         # Resample if needed
#         y, sr = librosa.load(path, sr=None)
#         if sr > 22050:
#             y = librosa.resample(y, orig_sr=sr, target_sr=44100)
#             sr = 44100
#         elif sr < 22050:
#             print(f"Skipping {filename} as it is already at a lower sample rate ({sr} Hz)")
#             continue
#         # If filename contains '_new_zealand', '_north_island', or '_south_island', remove it
#         if '_new_zealand' in filename or '_north_island' in filename or '_south_island' in filename:
#             filename = filename.replace('_new_zealand', '').replace('_north_island', '').replace('_south_island', '')
#             print(f"Renaming {filename} to remove island names.")
#         # Save to nested folder: resampled/english_name/scientific_name/filename.wav
#         nested_folder = os.path.join(resampled_dir, english_name, scientific_name)
#         os.makedirs(nested_folder, exist_ok=True)
#         out_path = os.path.join(nested_folder, filename)
#         sf.write(out_path, y, sr)
#         print(f"Saved {filename} to {nested_folder} (Scientific: {scientific_name_pretty})")

# if __name__ == "__main__":
#     print("Loading audio data clips...")
#     audio_clips, labels = load_audio_data(duration=3, save_segments=True)
#     print(f"Loaded {len(audio_clips)} audio clips.")

#     print("Encoding labels...")
#     label_encoder = LabelEncoder()
#     encoded_labels = label_encoder.fit_transform(labels)
#     print(f"Encoded {len(set(encoded_labels))} unique labels.")

#     print("Splitting data into train and test sets...")
#     X_train, X_test, y_train, y_test = train_test_split(
#         audio_clips, encoded_labels, test_size=0.2
#     )
#     print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

#     print("Extracting Mel-Spectrogram features...")
#     mel_features = extract_features(X_train)
#     print(f"Extracted features for {len(mel_features)} training samples.")

#     print("Creating image patches with positional encodings...")
#     grid_patched_features, labels = create_image_patches_with_position(
#         mel_features, y_train, patch_size=4, height=128, width=188
#     )
#     print(f"Created patched features with shape: {np.array(grid_patched_features).shape}")

#     # Show 10 first Mel-Spectrograms
#     print("Displaying Mel-Spectrograms of the first 10 audio clips...")
#     plt.figure(figsize=(20, 8))
#     for i in range(10):
#         plt.subplot(2, 5, i + 1)
#         librosa.display.specshow(
#             mel_features[i], sr=22050, x_axis="time", y_axis="mel"
#         )
#         plt.colorbar(format="%+2.0f dB")
#         plt.title(f"Mel-Spectrogram of clip {i + 1}")
#     plt.tight_layout()
#     plt.show()

#     """ # Build the model
#     model = build_transformer_model(
#         input_shape=(1504, 18),  # (Number of patches per input images, size of a patch)
#         num_classes=3,
#         num_heads=4,
#         ff_dim=64,
#         num_transformer_blocks=1,
#     )
#     model.summary()

#     # Compile the model
#     model.compile(
#         optimizer="adam",
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         metrics=["accuracy"],
#     )

#     # Train the model
#     history = model.fit(
#         grid_patched_features, labels, batch_size=32, epochs=3, validation_split=0.2
#     ) """

# Make into class

class AudioProcessor:
    def __init__(self, data_dir="resampled", duration=3, save_segments=True, segments_dir="segments"):
        self.data_dir = data_dir
        self.duration = duration
        self.save_segments = save_segments
        self.segments_dir = segments_dir

    def process(self):
        audio_clips, labels = self.load_audio_data(
            data_dir=self.data_dir,
            duration=self.duration,
            save_segments=self.save_segments,
            segments_dir=self.segments_dir
        )
        return audio_clips, labels
    
    
    def segment_audio_snr(self, audio, sr, segment_len_ms=2500, hop_len_ms=1000, noise_len_ms=500, snr_threshold=5):
        segment_len_samples = int(sr * segment_len_ms / 1000)
        hop_len_samples = int(sr * hop_len_ms / 1000)
        noise_len_samples = int(sr * noise_len_ms / 1000)

        def get_noise_level(samples):
            abs_max = []
            if len(samples) > noise_len_samples:
                for i in range(0, len(samples) - noise_len_samples, noise_len_samples):
                    abs_max.append(np.max(np.abs(samples[i:i + noise_len_samples])))
            else:
                abs_max.append(np.max(np.abs(samples)))
            return min(abs_max) if abs_max else 1e-6

        noise_level = get_noise_level(audio)
        segments = []
        for i in range(0, len(audio) - segment_len_samples, hop_len_samples):
            segment = audio[i:i + segment_len_samples]
            seg_abs_max = np.max(np.abs(segment))
            snr = seg_abs_max / noise_level if noise_level != 0 else 0
            if snr > snr_threshold:
                hann = np.hanning(len(segment))
                windowed = segment * hann
                spectrum = np.abs(np.fft.rfft(windowed))
                segments.append(segment)
        return segments

    def load_audio_data(self, data_dir="resampled", duration=3, save_segments=True, segments_dir="segments"):
        """
        Loads audio files from the specified directory, detects onsets, and extracts clips of a specified duration.
        """
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} does not exist.")

        self.resample_audio()

        audio_clips = []
        labels = []
        total_files = 0
        processed_files = 0

        # Create segments directory if saving segments
        if save_segments and not os.path.exists(segments_dir):
            os.makedirs(segments_dir)

        print(f"Walking through directory: {data_dir}")
        # Recursively walk through all subfolders and process .wav files
        for root, dirs, files in os.walk(data_dir):
            wav_files = [f for f in files if f.endswith('.wav')]
            if wav_files:
                print(f"Found {len(wav_files)} .wav files in {root}")
                total_files += len(wav_files)
            
            for file in wav_files:
                file_path = os.path.join(root, file)
                # Use the English name as the label (parent folder of scientific name)
                label = os.path.basename(os.path.dirname(root))
                print(f"Processing {file}, label: {label}")
                
                # Load the audio file
                audio, sr = librosa.load(file_path, sr=None)
                print(f"  Audio length: {len(audio)/sr:.2f}s, sample rate: {sr}")
                
                # Use more flexible segmentation - lower SNR threshold and shorter segments
                segments = self.segment_audio_snr(audio, sr, segment_len_ms=3000, snr_threshold=2)
                print(f"  Found {len(segments)} segments")
                
                # If no segments found, use simple fixed-duration chunking as fallback
                if len(segments) == 0:
                    print("  No SNR segments found, using fixed chunking")
                    chunk_size = int(duration * sr)
                    for start in range(0, len(audio) - chunk_size, chunk_size // 2):  # 50% overlap
                        chunk = audio[start:start + chunk_size]
                        if len(chunk) == chunk_size:
                            segments.append(chunk)
                    print(f"  Created {len(segments)} fixed chunks")
                
                for i, clip in enumerate(segments):
                    # More flexible duration check (accept 50% to 150% of target)
                    expected_length = duration * sr
                    clip_duration = len(clip) / sr
                    print(f"    Segment {i}: {clip_duration:.2f}s")
                    
                    if len(clip) >= expected_length * 0.5:
                        # Pad or trim to exact duration
                        if len(clip) < expected_length:
                            clip = np.pad(clip, (0, expected_length - len(clip)), mode='constant')
                        elif len(clip) > expected_length:
                            clip = clip[:expected_length]
                        
                        audio_clips.append(clip)
                        labels.append(label)
                        print(f"    Added segment {i} to dataset")
                        
                        # Save segment if requested
                        if save_segments:
                            # Create nested structure in segments folder
                            english_name = os.path.basename(os.path.dirname(root))
                            scientific_name = os.path.basename(root)
                            segment_folder = os.path.join(segments_dir, english_name, scientific_name)
                            os.makedirs(segment_folder, exist_ok=True)
                            
                            # Save segment with original filename + segment number
                            segment_filename = f"{file[:-4]}_segment_{i}.wav"
                            segment_path = os.path.join(segment_folder, segment_filename)
                            sf.write(segment_path, clip, sr)
                
                processed_files += 1

        print(f"Total files found: {total_files}, processed: {processed_files}")
        print(f"Total segments extracted: {len(audio_clips)}")
        return audio_clips, labels

    # Create features using Mel-Spectrogram
    def extract_features(self, audio_files):
        features = [librosa.feature.melspectrogram(y=audio) for audio in audio_files]
        return features

    # Create image patches with the positional encodings
    def create_image_patches_with_position(
        self, features, labels, patch_size=4, height=128, width=188, expected_sequence_length=1504
    ):
        """
        Creates image patches and appends grid positional encodings.
        """
        patches = []

        # Calculate grid size
        grid_height = height // patch_size
        grid_width = width // patch_size

        # Create positional encodings
        pos_enc_height = np.repeat(
            np.arange(grid_height)[:, np.newaxis], grid_width, axis=1
        )
        pos_enc_width = np.repeat(np.arange(grid_width)[np.newaxis, :], grid_height, axis=0)

        sequences = []

        for idx, feature in enumerate(features):
            current_sequence = []

            for i in range(0, height, patch_size):
                for j in range(0, width, patch_size):
                    patch = feature[i : i + patch_size, j : j + patch_size]

                    # Get positional encoding for current patch
                    pos_enc = np.array(
                        [
                            pos_enc_height[i // patch_size, j // patch_size],
                            pos_enc_width[i // patch_size, j // patch_size],
                        ]
                    )

                    # Append positional encoding to the patch
                    patch_with_pos = np.concatenate([patch.flatten(), pos_enc])

                    current_sequence.append(patch_with_pos)

            # Pad sequence if needed
            while len(current_sequence) < expected_sequence_length:
                current_sequence.append(np.zeros_like(patch_with_pos))

            sequences.append(current_sequence)

        return np.array(sequences), labels

    def build_transformer_model(
        self, input_shape, num_classes, num_heads, ff_dim, num_transformer_blocks
    ):
        """
        Builds a transformer-based model for processing image patches.
        """
        inputs = tf.keras.Input(shape=input_shape)

        # Start with a Linear layer followed by Batch Normalization as shown in the diagram
        x = tf.keras.layers.Dense(units=18, activation="relu")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)

        # Transformer encoder
        for _ in range(num_transformer_blocks):
            # Multi-head attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=18, dropout=0.1
            )(x, x)
            x = tf.keras.layers.Add()([x, attention_output])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

            # Feed-forward layer
            ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
            ffn_output = tf.keras.layers.Dense(18)(ffn_output)
            x = tf.keras.layers.Add()([x, ffn_output])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # Mean over the sequence length (across the time axis of the Mel-Spectrogram)
        avg_out = tf.keras.layers.GlobalAveragePooling1D()(x)

        # Classification head with Linear layer followed by a Cross-Entropy loss during training
        outputs = tf.keras.layers.Dense(num_classes)(avg_out)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def resample_audio(self, download_dir="downloads", resampled_dir="resampled"):
        """
        For each .wav file in download_dir, resample to 44100 Hz if needed, and copy to resampled_dir
        in subfolders for both English and scientific names.
        """
        if input("Do you want to resample audio files? (y/n): ").strip().lower() != 'y':
            print("Skipping audio resampling.")
            return
        if not os.path.exists(resampled_dir):
            os.makedirs(resampled_dir)
        else:
            print(f"Directory {resampled_dir} already exists. Deleting entire folder.")
            shutil.rmtree(resampled_dir)
        filenames = [f for f in os.listdir(download_dir) if f.endswith(".wav")]
        for filename in filenames:
            # Construct full path
            path = os.path.join(download_dir, filename)
            # Parse filename: e.g. '33763_tui_prosthemadera_novaeseelandiae_song.wav' -> "{file_id}_{english_name}_{scientific_name}.wav"
            parts = filename[:-4].split('_')  # remove .wav, split
            if len(parts) < 3:
                print(f"Filename format not recognized: {filename}")
                continue
            file_id = parts[0]
            english_name = parts[1]
            scientific_name = '_'.join(parts[2:-1])
            scientific_name_pretty = ' '.join([p.capitalize() for p in parts[2:-1]])
            # Resample if sr is less
            y, sr = librosa.load(path, sr=None)
            if sr > 22050:
                y = librosa.resample(y, orig_sr=sr, target_sr=44100)
                sr = 44100
            elif sr < 22050:
                continue
            # Save to nested folder: resampled/english_name/scientific_name/filename.wav
            nested_folder = os.path.join(resampled_dir, english_name, scientific_name)
            os.makedirs(nested_folder, exist_ok=True)
            out_path = os.path.join(nested_folder, filename)
            sf.write(out_path, y, sr)

def main():
    print("Loading audio data clips...")
    processor = AudioProcessor()
    audio_clips, labels = processor.load_audio_data(
        data_dir="resampled",
        duration=3,
        save_segments=True,
        segments_dir="segments"
    )
    print(f"Loaded {len(audio_clips)} audio clips.")

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    print(f"Encoded {len(set(encoded_labels))} unique labels.")

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        audio_clips, encoded_labels, test_size=0.2
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    print("Extracting Mel-Spectrogram features...")
    mel_features = processor.extract_features(X_train)
    print(f"Extracted features for {len(mel_features)} training samples.")

    print("Creating image patches with positional encodings...")
    grid_patched_features, labels = processor.create_image_patches_with_position(
        mel_features, y_train, patch_size=4, height=128, width=188
    )
    print(f"Created patched features with shape: {np.array(grid_patched_features).shape}")

    # Show 10 first Mel-Spectrograms
    print("Displaying Mel-Spectrograms of the first 10 audio clips...")
    plt.figure(figsize=(20, 8))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        librosa.display.specshow(
            mel_features[i], sr=22050, x_axis="time", y_axis="mel"
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Mel-Spectrogram of clip {i + 1}")
    plt.tight_layout()
    plt.show()

    """ # Build the model
    model = build_transformer_model(
        input_shape=(1504, 18),  # (Number of patches per input images, size of a patch)
        num_classes=3,
        num_heads=4,
        ff_dim=64,
        num_transformer_blocks=1,
    )
    model.summary()

    # Compile the model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Train the model
    history = model.fit(
        grid_patched_features, labels, batch_size=32, epochs=3, validation_split=0.2
    ) """

if __name__ == "__main__":
    main()