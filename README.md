# NLU-Evidence-Detection-Group56

modelID : c38534kt-x32001nm-ED

## Overview

This repository contains our group's implementations for the Natural Language Understanding shared task, focusing specifically on Track B: Evidence Detection (ED). We have developed two distinct solutions for the pairwise sequence classification problem, utilizing different deep learning methodologies that align with approaches B and C as outlined by the task guidelines.

## Intended Learning Outcomes

Our project's objectives were to:

- Build and implement two distinct solutions for the pairwise sequence classification problem.
- Describe and present our solutions through a flash presentation.
- Collaborate responsibly on project development and documentation, contributing to planning and organization.

## The Shared Task Tracks

We elected to participate in:

- Track B: Evidence Detection (ED), where the goal is to determine the relevance of evidence to a given claim.

The datasets provided for Track B include over 23K claim-evidence pairs for training and almost 6K pairs for validation.

## Approaches

- **Approach B**: A deep learning-based solution that does not employ transformer architectures, developed by Nur Mohmad Noh.
- **Approach C**: A deep learning-based solution underpinned by transformer architectures, developed by Khairunnisa Talib.

Each approach is encapsulated in its respective Jupyter notebook within this repository.

## Usage

Follow the steps outlined in each Jupyter notebook for the respective solutions:

1. Install all dependencies as indicated at the beginning of each notebook.
2. Load the trained models using the instructions provided within the notebooks.
3. Input the `test.csv` file and execute the notebook to generate predictions.
4. Predictions will be output to `Group_56_<B_or_C>.csv` following the specified format.

## Repository Structure

Notebooks for training models for each approach:

- `Group_56_LSTM.ipynb`: Jupyter notebook demonstrating the inference process for Approach B.
- `Group_56_C.ipynb`: Jupyter notebook demonstrating the inference process for Approach C.

The tokeniser or classifier used in our models:
- `tokenizer.pickle`: Serialized tokenizer used for text preprocessing in Approach B.
  
The predictions files:
- `Group_56_B.csv`: output file with predictions from Approach B.
- `Group_56_C.csv`: output file with predictions from Approach C.

Our trained models, due to their sizes, are as followings:

- Trained Model for Approach B: `bidirectional_lstm_model.h5`
- [Trained Model for Approach C]:
   Access these models using the link https://drive.google.com/drive/folders/177AI6UpgCXBN54_CiDVI1yt-Lg3olJTA?usp=drive_link
   `bert_for_sequence_classification.pth`
    `scaler.joblib`
    `lr_classifier.joblib`
  
  



## Model Cards

Model cards for each solution are provided in the repository, offering concise descriptions of the models and their performance metrics.

## Acknowledgements

We strictly adhered to the closed mode of the shared task, using only the datasets provided without incorporating any external datasets.

## Contributors

- Partner 1 (Approach B): x32001nm - Nur Mohmad Noh
- Partner 2 (Approach C): c38534kt - Khairunnisa Talib

## Authors

- Nur Mohmad Noh
- Khairunnisa Talib


