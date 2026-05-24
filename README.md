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

## Hypotheses

### H1: Shortcut Learning in Baseline

The RoBERTa baseline is expected to perform well on the in-distribution test set but worse on the OOD test set because it may rely on shortcut-trigger words such as `die`, `kill`, `cut`, and `jump`.

### H2: Counterfactual Augmentation

Counterfactual augmentation is expected to improve OOD performance by teaching the model that the same keyword can have different labels depending on context.

### H3: Keyword Masking

Keyword masking is expected to reduce shortcut reliance by forcing the model to use surrounding context instead of only focusing on risky keywords.

### H4: Confidence Regularization

Confidence regularization is expected to reduce overconfident wrong predictions on OOD examples.

### H5: Full Method

The full model, `RoBERTa-CF-KM-CR`, is expected to achieve better OOD Macro-F1, a smaller ID-OOD gap, lower keyword sensitivity, and fewer confident wrong predictions than the baseline.

### H6: MC Dropout

MC Dropout is expected to help analyze uncertainty on OOD examples, but it is used as an inference-time uncertainty baseline, not as the main mitigation method.
