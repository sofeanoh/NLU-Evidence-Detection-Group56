---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/sofeanoh/NLU-Evidence-Detection-Group56

---

# Model Card for c38534kt-x32001nm-ED

<!-- Provide a quick summary of what the model is/does. -->

This is a Bidirectional LSTM classification model that was trained to
        detect the relevance of evidence to a given claim as part of the Evidence Detection (ED) track.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model utilizes a deep learning-based approach with a Bidirectional LSTM
        architecture. It was trained on over 23K claim-evidence pairs and does not employ transformer architectures.

- **Developed by:** Nur Mohmad Noh and Khairunnisa Talib
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Bidirectional LSTM
- **Finetuned from model [optional]:** [More Information Needed]

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://github.com/keras-team/keras/tree/v3.2.1/keras/layers/rnn
- **Paper or documentation:** https://arxiv.org/pdf/1506.00019.pdf

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

23K pairs of claim and evidence

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - max_words: 10000
      - max_len: 100
      - LSTM units: 128
      - Dropout: 0.5
      - Optimizer: Adam
      - Loss: Binary Crossentropy
      - Batch size: 64
      - Epochs: 10

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 11.7mins
      - duration per training epoch: 2.mins
      - model size: 18.6MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A development dataset provided for validation, amounting to 6K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy

### Results

The model obtained an F1-score of 80% and an accuracy of 80%.

## Technical Specifications

### Hardware


      - RAM: at least 8 GB
      - Storage: at least 2GB,
      - GPU: Optional but recommended for faster training

### Software


    - Keras 2.6.0
    - TensorFlow 2.6.0

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Since this model uses a Bidirectional LSTM, it might not capture long-range dependencies as effectively as transformer models.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The tokenizer used for this model has been saved and will be utilized for consistent text preprocessing during inference.
