# Disambiguating Suicidal Intent:
## Mitigating Shortcut Learning and Catastrophic Forgetting for Crisis NLP

Final Project for COSE461 — Natural Language Processing  
Korea University

## Team Members
- Julia Irsalina (2023320344)
- Mahirah Sofea (2023320033)
- Nadiah Nabilah (2023320093)
- Emira Syazwani (2023320334)

---

# Project Overview

This project investigates **shortcut learning** and **catastrophic forgetting** in risky-intent classification for crisis NLP systems.

Modern language models often achieve very high in-distribution accuracy while failing to correctly interpret context-dependent expressions such as:

- “I don’t want to die.”
- “I want to die of laughter.”
- “I won’t cut myself.”

These examples contain high-risk keywords but do not necessarily express suicidal intent. Our work evaluates whether models truly understand contextual intent or simply rely on superficial keyword correlations.

---

# Main Objectives

- Analyze shortcut learning behavior in RoBERTa-based classifiers
- Evaluate robustness under out-of-distribution (OOD) settings
- Investigate catastrophic forgetting after fine-tuning
- Improve contextual understanding using:
  - Keyword masking
  - Counterfactual augmentation
  - Experience replay
  - NLI initialization
- Analyze prediction behavior using:
  - SHAP interpretability
  - MC Dropout uncertainty estimation

---

# Dataset

We construct a custom ambiguous-intent dataset containing approximately **3,120 examples** using 20 ambiguous risk-related keywords such as:

- `die`
- `kill`
- `cut`
- `jump`

The dataset includes multiple linguistic categories:

- Direct intent
- Figurative language
- Negation
- Temporal context
- Negation + temporal
- Ambiguous intent

An additional OOD dataset is used to evaluate contextual robustness.

---

# Experiments

| Experiment | Description |
|---|---|
| E1 | RoBERTa fine-tuned baseline |
| E2 | RoBERTa + keyword masking |
| E3 | RoBERTa + counterfactual augmentation |
| E4 | RoBERTa + keyword masking + counterfactual augmentation |
| E5 | RoBERTa + experience replay |
| E6 | NLI-RoBERTa zero-shot |
| E7 | NLI-RoBERTa fine-tuned |
| E8 | NLI-RoBERTa + keyword masking + counterfactual augmentation |
| E9 | NLI-RoBERTa + experience replay + counterfactual augmentation |
| E10 | NLI-RoBERTa + experience replay + keyword masking + counterfactual augmentation |

---

# Key Findings

- Most models achieve near-perfect ID accuracy but struggle on OOD examples
- The RoBERTa baseline heavily relies on shortcut keyword associations
- Keyword masking alone is insufficient for robust contextual understanding
- Counterfactual augmentation significantly improves OOD robustness
- NLI fine-tuning alone suffers from catastrophic forgetting
- Combining NLI initialization with experience replay and contextual mitigation achieves the best robustness

### Best Model (E10)

| Metric | Score |
|---|---|
| ID Macro-F1 | 1.0000 |
| OOD Macro-F1 | 0.7531 |
| ID-OOD Gap | 0.2469 |

E10 improves OOD Macro-F1 by approximately **30 percentage points** compared with the RoBERTa baseline.

---

# Methodology

## 1. Shortcut Learning Mitigation

### Keyword Masking
Risk-related keywords are randomly masked during training to reduce direct keyword-label dependence.

### Counterfactual Augmentation
The model is exposed to the same keyword under both risky and non-risky contexts.

Example:
- Risky: “I want to die tonight.”
- Non-risky: “I want to die of laughter.”

---

## 2. Continual Learning (LIER)

We propose:

### LIER — Linguistically-Informed Experience Replay

LIER combines:
- NLI initialization
- Experience replay
- Structured linguistic replay buffers

This helps preserve contextual reasoning ability during fine-tuning.

---

## 3. SHAP Interpretability

SHAP analysis is used to inspect token-level prediction behavior.

Findings show:
- Baseline models rely heavily on isolated keywords
- Later mitigation models become more context-sensitive

---

## 4. MC Dropout Uncertainty Analysis

MC Dropout is used to estimate predictive uncertainty across multiple stochastic forward passes.

We observe:
- Shortcut-sensitive models can remain highly confident even when wrong
- Later mitigation models produce more stable and context-aware uncertainty behavior

---

# Repository Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── experiment_flow.ipynb
│
├── src/
│   ├── training/
│   ├── evaluation/
│   ├── shap/
│   └── mc_dropout/
│
├── results/
│   ├── metrics/
│   ├── plots/
│   ├── shap/
│   └── uncertainty/
│
├── report/
│   └── final_report.pdf
│
└── README.md
