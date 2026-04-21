# NER-HMM

A simple Named Entity Recognition (NER) project using a Hidden Markov Model (HMM) and the Viterbi algorithm.

## What This Project Does

- Trains an HMM on tokenized text and NER tags.
- Computes:
  - start probabilities
  - transition probabilities
  - emission probabilities
- Predicts the best tag sequence for a sentence using Viterbi decoding.

## Project Structure

- `itegrate.ipynb`: main notebook for data loading, training, evaluation, and plots.
- `data/data.csv`: dataset used in the notebook.
- `model/HMM.py`: HMM class (training + inference wrapper).
- `model/label_maps.py`: NER label list and index mappings.
- `model/preprocessing.py`: preprocessing helper (`UNK` thresholding).
- `components/start_prob.py`: start probability calculation.
- `components/transition_prob.py`: transition matrix calculation.
- `components/emission_prob.py`: emission matrix calculation.
- `components/viterbi.py`: Viterbi decoding implementation.
- `ner_tags.txt`: list of supported NER tags.

## Requirements

Install Python packages used by the notebook:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn pydash
```

## How To Run

1. Open `itegrate.ipynb` in Jupyter/VS Code.
2. Run cells from top to bottom.
3. The notebook will:
   - read and prepare data
   - train the HMM
   - run predictions
   - show evaluation metrics (classification report, confusion matrix)

## Notes

- The notebook file name is currently `itegrate.ipynb`.
- Unknown or rare tokens are handled through `UNK` mapping in preprocessing.
