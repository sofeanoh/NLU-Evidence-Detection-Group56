---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
- bert
- logistic-regression
- hybrid-model
repo: https://github.com/username/project_name

---

# Model Card for c38534kt-x32001-ED

<!-- Provide a quick summary of what the model is/does. -->

This is a pairwise sequence classification model that was trained to  determine if the evidence is relevant to the claim


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon a BERT model that was fine-tuned on 23K pairs of texts, and then the robust text representation is fed into Logistic Regression for classification

- **Developed by:** Khairunnisa Talib and Nur Mohmad Noh
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** BERT with Logistic Regression
- **Finetuned from model [optional]:** bert-base-uncased

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/google-bert/bert-base-uncased
- **Paper or documentation:** https://aclanthology.org/N19-1423.pdf

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

23K pairs of texts drawn from emails, news articles and blog posts.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


        - learning_rate: 2e-5
        - batch_size: 16
        - epochs: 1
        - max_seq_length: 256
        - optimizer: AdamW
        - loss: Cross-Entropy Loss
        - gradient_clipping: 1.0

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time:9 minutes
      - duration per training epoch: 9 minutes
      - model size: 417MB

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

The model obtained an accuracy of 87% and weighted average F1-score of 87%

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: L4

### Software


        - Python: 3.7 or higher
        - Transformers: 4.18.0
        - PyTorch: 1.11.0+cu113
        - Scikit-learn: 1.0.2
        - Joblib: 1.1.0
        - tqdm: 4.64.0

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      512 subwords will be truncated by the model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by experimentation
      with different values to balance between computational efficiency and model performance.'

