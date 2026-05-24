# Shortcut Learning Risky Intent Classification

This project studies shortcut learning in risky-intent classification.

## Main goal

We evaluate whether RoBERTa relies too much on shortcut-trigger words such as "die", "kill", "cut", and "jump".

## Experiments

- E0: TF-IDF baseline
- E1: RoBERTa baseline
- E2: RoBERTa + Keyword Masking
- E3: RoBERTa + Counterfactual Augmentation
- E4: RoBERTa + Counterfactual + Keyword Masking
- E5: RoBERTa + Counterfactual + Keyword Masking + Confidence Regularization
- E6: MC Dropout uncertainty analysis

## Main files

- `notebooks/experiment_flow.ipynb`: main experiment notebook
- `src/`: Python scripts
- `data/raw/`: original datasets
- `data/processed/`: processed train/test/OOD files
- `results/`: metrics, predictions, plots
- `saved_models/`: trained models
# shortcut-learning-risky-intent
