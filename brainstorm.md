# NZ Bird Classifier Model

*A model trained to classify New Zealand bird sounds by their Māori identity.*

## Data

Maybe just download the New Zealand files off of Xeno-canto and analyse the data to be cleaned. API calling should do the job...

By calling the Xeno-canto API i have been able to achieve obtaining 1,518 recordings of 144 unique New Zealand species. These recordings have an average length of 80.61s which seems like it will serve as a great base for the data. I intend to find some more resources across the web to obtain on top of the current sounds dataset. Following this, I will be cleaning and maintaining the data before transforming the .wav files into mel spectrogram representations. This will serve as training/finetuning data for the classification model and when the model is finalised, there will be the ability to upload videos to be tested against the model and the videos will be transformed to mel spectograms automatically.

Potentially segment to expand dataset and mitigating risk of overfitting.

## Models

It is an idea to use a vision model like ViT to classify the sounds, as it takes a cool perspective of using vision for sound, right?

## Research

- [ ] Training classification models on sound
  - [ ] <https://www.researchgate.net/publication/342055866_Machine_Learning_Approach_to_Classify_Birds_on_the_Basis_of_Their_Sound>
- [ ] Potential models
  - [ ] ViT / ViTHybrid
    - [ ] torchvision.models
    - [ ] <https://huggingface.co/docs/transformers/v4.53.1/en/model_doc/vit_hybrid#transformers.ViTHybridModel>
- [x] Ways to make the sound interpretable
