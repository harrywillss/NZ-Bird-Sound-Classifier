# 🐦 New Zealand Bird Sound Classifier

*A machine learning project for classifying New Zealand native bird species using audio recordings and Vision Transformer (ViT) models.*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

This project implements a novel approach to bird species classification by treating audio spectrograms as visual data and using Vision Transformer (ViT) models for classification. The system can identify 10+ species of New Zealand native birds from their calls and songs.

### Key Features

- 🎵 **Audio-to-Vision Approach**: Converts bird audio recordings to mel spectrograms for visual classification
- 🤖 **Vision Transformer Model**: Uses ViT/ViTHybrid architecture for state-of-the-art classification performance
- 🔧 **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning approach
- 🌿 **Native Species Focus**: Specializes in New Zealand endemic and native bird species
- 📊 **Comprehensive Dataset**: 555+ audio recordings across 10 primary species

## 🦜 Supported Bird Species

The model currently classifies the following New Zealand bird species:

| Species | Count | Scientific Name |
|---------|-------|----------------|
| Tūī | 167 | *Prosthemadera novaeseelandiae* |
| New Zealand Bellbird | 77 | *Anthornis melanura* |
| Whitehead | 44 | *Mohoua albicilla* |
| New Zealand Fantail | 43 | *Rhipidura fuliginosa* |
| Robin | 43 | *Petroica longipes* |
| Kākā | 41 | *Nestor meridionalis* |
| Saddleback | 39 | *Philesturnus rufusater* |
| Tomtit | 39 | *Petroica macrocephala* |
| Morepork | 31 | *Ninox novaeseelandiae* |
| Silvereye | 30 | *Zosterops lateralis* |

## 🏗️ Project Structure

```text
├── README.md                 # Project documentation
├── brainstorm.md            # Project planning and research notes
├── data_downloader.py       # Xeno-canto API data acquisition script
├── count_specie.py          # Species count analysis
├── sample_rate.py           # Audio preprocessing and resampling
├── finetuning.ipynb         # Model training notebook
├── recordings_data.csv      # Dataset metadata
├── recordings_metadata.json # Detailed recording information
├── label_counts.txt         # Species distribution summary
├── downloads/               # Raw audio files (.wav)
├── resampled/              # Preprocessed audio files
└── reports/                # Project documentation and reports
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- librosa
- pandas
- requests
- tqdm
- soundfile

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/harrywillss/Tio.git
   cd Tio
   ```

2. **Install dependencies**

   ```bash
   pip install torch torchvision transformers librosa pandas requests tqdm soundfile python-dotenv
   ```

3. **Set up environment variables**

   Create a `.env` file in the project root:

   ```env
   XC_API_KEY=your_xeno_canto_api_key
   ```

### Usage

1. **Download bird recording data**

   ```bash
   python data_downloader.py
   ```

2. **Preprocess audio files**

   ```bash
   python sample_rate.py
   ```

3. **Analyze species distribution**

   ```bash
   python count_specie.py
   ```

4. **Train the model**

   Open and run `finetuning.ipynb` in Jupyter Notebook or your preferred environment.

## 🔬 Methodology

### Data Collection

- **Source**: Xeno-canto bird sound database
- **Quality Filter**: Grade C or better recordings
- **Geographic Filter**: New Zealand recordings only
- **Type Filter**: Calls and songs
- **Total Dataset**: 555+ recordings from 10+ species

### Preprocessing Pipeline

1. **Audio Standardization**: Resample all recordings to 44.1kHz
2. **Mel Spectrogram Conversion**: Transform audio to visual representations
3. **Data Augmentation**: Potential audio segmentation for dataset expansion
4. **Quality Control**: Filter segments with insufficient data

### Model Architecture

- **Base Model**: Vision Transformer (ViT) or ViT Hybrid
- **Fine-tuning Strategy**: LoRA (Low-Rank Adaptation) for efficient training
- **Input Format**: Mel spectrograms treated as images
- **Output**: Multi-class classification (10+ bird species)

## 📊 Dataset Statistics

- **Total Recordings**: 555 audio files
- **Average Length**: ~80.61 seconds
- **Audio Format**: WAV files at 44.1kHz
- **Species Coverage**: 10 primary species (>30 recordings each)
- **Data Quality**: Xeno-canto grade C or better

## 🎯 Project Roadmap

- [x] Data collection from Xeno-canto API
- [x] Audio preprocessing and standardization
- [x] Species analysis and filtering
- [ ] Mel spectrogram generation
- [ ] ViT model implementation
- [ ] LoRA fine-tuning setup
- [ ] Model training and validation
- [ ] Performance evaluation
- [ ] Real-time classification application
- [ ] Web interface for audio upload and classification

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Xeno-canto**: For providing the comprehensive bird sound database
- **Hugging Face**: For the Transformers library and pre-trained models
- **New Zealand Department of Conservation**: For species information and conservation efforts
- **AIML339 Course**: Academic framework and guidance

## 📧 Contact

For questions or collaboration opportunities, please reach out through the GitHub repository issues.

---

*This project is part of AIML339 coursework focusing on innovative applications of machine learning in bioacoustics and conservation.*
