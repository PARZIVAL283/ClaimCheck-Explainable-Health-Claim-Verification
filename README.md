# ClaimCheck: Explainable Health-Claim Verification

ClaimCheck is an educational NLP project that assesses short health claims and provides a verdict, confidence score, and plain-language explanation. It is designed to help users think critically about health information online—not to replace professional medical advice or authoritative fact-checking.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU-orange)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

Health misinformation can spread quickly on social media, especially when claims use absolute language such as “cures,” “prevents,” or “guarantees.” ClaimCheck addresses this problem by classifying health claims into evidence-based veracity categories and explaining the reasoning behind each prediction.

The system uses the PUBHEALTH dataset, a benchmark for explainable automated fact-checking of public health claims. Each claim is associated with one of four veracity labels—`true`, `false`, `unproven`, or `mixture`—and includes an explanation justifying the assigned label. [70][68]

## Features

- Classifies health claims into `true`, `false`, `mixture`, and `unproven`
- Provides a confidence score for each prediction
- Generates a plain-language explanation based on the claim’s predicted category
- Compares a traditional TF-IDF baseline with a fine-tuned DistilBERT model
- Reports accuracy, macro F1, per-class precision/recall, and a confusion matrix
- Includes a Gradio web interface for interactive demonstrations
- Clearly identifies uncertainty instead of forcing a misleading binary “real/fake” decision

## Project Motivation

A simple fake-news classifier can be useful, but it is insufficient for health-related misinformation because claims are often partially true, context-dependent, or unsupported by available evidence.

ClaimCheck uses four categories rather than a binary label:

| Label | Meaning |
|---|---|
| `true` | The claim is supported by available evidence |
| `false` | The claim conflicts with available evidence |
| `mixture` | The claim contains both supported and unsupported elements |
| `unproven` | There is not enough reliable evidence to verify the claim |

This design encourages more responsible AI use: uncertain claims are labeled as unproven rather than automatically marked as false.

## Dataset

This project uses the [PUBHEALTH dataset](https://huggingface.co/datasets/ImperialCollegeLondon/health_fact).

PUBHEALTH is designed for explainable automated fact-checking of public health claims. It includes claim text, veracity labels, explanation text, and source-related metadata. [70][68]

### Dataset fields used

| Field | Description |
|---|---|
| `claim` | The health claim to classify |
| `label` | Veracity label: `true`, `false`, `mixture`, or `unproven` |
| `explanation` | Human-written justification for the assigned label |
| `sources` | Supporting sources or references, where available |

## Model Architecture

### Baseline model

The baseline uses:

- TF-IDF vectorization
- Unigrams and bigrams
- Logistic Regression
- Class weighting to handle label imbalance

This provides a fast, interpretable reference point for evaluating the transformer model.

### Transformer model

The main model fine-tunes `distilbert-base-uncased` for four-class health-claim classification.

The pipeline is:

```text
Health claim
      ↓
Tokenization
      ↓
DistilBERT encoder
      ↓
Classification head
      ↓
Verdict + confidence score
      ↓
Explanation generator
```

## Results

| Model | Accuracy | Macro F1 | Notes |
|---|---:|---:|---|
| TF-IDF + Logistic Regression | `0.593` | `0.4605` | Baseline |
| DistilBERT fine-tuned | `0.637 ` | `0.3711` | Main model |

### Confusion matrix

<img width="657" height="590" alt="image" src="https://github.com/user-attachments/assets/132fef43-d02d-40a0-8bdd-bc9ba287fa14" />


### Key observations

- `[Add observation: e.g., The model performs best on false claims.]`
- `[Add observation: e.g., Unproven claims are harder to classify because evidence may be limited.]`
- `[Add observation: e.g., Some errors occur when claims contain partially correct information.]`

## How to Run

### 1. Open in Google Colab

1. Open Google Colab.
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`.
3. Upload or clone this repository.

### 2. Install directly in Colab:

```python
!pip install -q datasets transformers evaluate accelerate scikit-learn gradio
```

### 3. Train the baseline

### 4. Fine-tune DistilBERT

### 5. Launch the demo

## Example Input and Output

**Input claim:**

```text

```

**Example output:**

```text

```

## Tech Stack

- Python
- Google Colab
- Hugging Face Transformers
- Hugging Face Datasets
- DistilBERT
- scikit-learn
- pandas
- NumPy
- Matplotlib and Seaborn
- Gradio
- PyTorch

## Evaluation Methodology

The project evaluates models using:

- Accuracy
- Macro F1 score
- Per-class precision, recall, and F1
- Confusion matrix
- Confidence-score analysis
- Error analysis on misclassified examples

Macro F1 is emphasized because the dataset may contain class imbalance. Accuracy alone can hide poor performance on minority classes.

## Limitations

- ClaimCheck is not a medical diagnostic, treatment, or emergency tool.
- The model may make mistakes on new, ambiguous, or evolving health claims.
- A prediction should not be treated as a definitive determination of truth.
- The current version focuses on English-language health claims.
- The explanation is template-based in version 1; future versions may use retrieval-augmented generation with cited evidence.
- The model may inherit biases present in the training data and source material.

## Future Improvements

- Add evidence retrieval from trusted health sources
- Generate explanations with inline citations
- Add a “needs human fact-checking” category
- Evaluate retrieval quality using precision@k and recall@k
- Support Bangla and other low-resource languages
- Add adversarial and out-of-distribution claim testing
- Deploy the model through Hugging Face Spaces
- Add model cards and dataset documentation

## Responsible AI

ClaimCheck is intended to support media and information literacy, not to make final judgments about truth. Users should review cited evidence, consider context, and consult qualified professionals for personal health decisions.

The project follows a cautious approach by:

- Using multiple veracity categories instead of a binary fake/real label
- Showing confidence rather than presenting predictions as certainty
- Providing explanations for model decisions
- Explicitly labeling limitations and potential failure cases
- Avoiding medical advice or diagnosis

## Contributing

Contributions are welcome. If you want to improve the model, add data processing steps, or expand the project scope, please open an issue or submit a pull request.

## License

This project is released under the MIT License.

## Author

**Sourodip Das Gupta**  

Data Science / NLP / AI Enthusiast Student 

Email: sourodipdasgupta283@gmail.com

linkedIn: www.linkedin.com/in/sourodip-das-gupta-3b97a62a6

## Acknowledgements

- [PUBHEALTH Dataset](https://huggingface.co/datasets/ImperialCollegeLondon/health_fact)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Google Colab](https://colab.research.google.com/)
- [Gradio](https://www.gradio.app/)
