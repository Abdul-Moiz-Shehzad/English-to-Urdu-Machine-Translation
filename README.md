# English-to-Urdu Machine Translation

A sequence-to-sequence machine translation framework for translating English text to Urdu. This project explores multiple neural architectures—ranging from baseline recurrent neural networks (SimpleRNN, Bidirectional RNN, LSTM with teacher forcing) and custom Transformer implementations to a fine-tuned multilingual sequence-to-sequence model (mBART-50). It includes end-to-end data preprocessing pipelines, evaluation benchmarks, and deployment interfaces using Flask and Streamlit.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset Pipeline](#dataset-pipeline)
  - [Corpus Composition](#corpus-composition)
  - [Preprocessing and Tokenization](#preprocessing-and-tokenization)
- [Model Architectures](#model-architectures)
  - [Baseline Recurrent Models](#baseline-recurrent-models)
  - [Custom Transformer Models](#custom-transformer-models)
  - [Fine-Tuned mBART-50](#fine-tuned-mbart-50)
- [Evaluation and Results](#evaluation-and-results)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
  - [Flask Web Interface](#flask-web-interface)
  - [Streamlit LSTM Interface](#streamlit-lstm-interface)
  - [Jupyter Notebooks](#jupyter-notebooks)
- [Configuration](#configuration)
- [References](#references)
- [License](#license)

---

## Overview

English and Urdu belong to different language families with distinct syntactic structures, morphological complexities, and script orientations (Left-to-Right Latin script vs. Right-to-Left Nastaliq/Perso-Arabic script). This repository addresses English-to-Urdu translation through a comparative study:

1. **Recurrent Seq2Seq Baselines:** Evaluates standard RNN, Bidirectional RNN, and LSTM encoder-decoder architectures with teacher forcing.
2. **Transformer Implementations:** Assesses custom multi-head attention models with and without sinusoidal positional encodings.
3. **Transfer Learning:** Fine-tunes Facebook's `mbart-large-50-many-to-many-mmt` (`en_XX` to `ur_PK`) using Hugging Face Transformers.
4. **Interactive Deployment:** Provides two web interfaces:
   - A Flask application utilizing the fine-tuned mBART-50 model.
   - A Streamlit dashboard running autoregressive inference with the Keras LSTM model.

---

## Project Structure

```plaintext
English-to-Urdu-Machine-Translation/
│
├── config.yaml                   # Global configuration (sequence padding lengths)
├── requirements.txt              # Project dependencies
├── setup.py                      # Package installation script for local modules
├── app.py                        # Flask web application (mBART-50 backend)
├── app_lstm.py                   # Streamlit web application (LSTM baseline backend)
├── LICENSE                       # MIT License
├── README.md                     # Project documentation
│
├── data/
│   ├── raw/                      # Raw dataset files
│   │   ├── parallel-corpus.xlsx  # Parallel sentence pairs in Excel format
│   │   ├── english-corpus.txt    # Raw English source text
│   │   └── urdu-corpus.txt       # Raw Urdu target text
│   ├── processed/                # Preprocessed and merged dataset files
│   │   ├── combined-parallel-corpus.csv
│   │   └── processed-parallel-corpus.csv
│   └── artifacts/                # Serialized tokenizers and padded arrays
│       ├── english-tokenizer.joblib
│       ├── urdu-tokenizer.joblib
│       ├── tokenized-english-sentences.joblib
│       └── tokenized-urdu-sentences.joblib
│
├── notebooks/
│   ├── research.ipynb            # Data exploration, baseline models, and evaluation
│   ├── finetune_mbart50.ipynb    # mBART-50 fine-tuning script with Hugging Face Trainer
│   └── testing_mbart50.ipynb     # Inference testing for fine-tuned mBART-50
│
├── src/
│   ├── __init__.py
│   ├── translation/
│   │   ├── __init__.py
│   │   └── translator.py         # mBART-50 inference module
│   └── utils/
│       ├── __init__.py
│       └── config_loader.py      # YAML configuration loader with absolute path resolution
│
├── static/
│   └── style.css                 # Styling for Flask interface
└── templates/
    └── index.html                # HTML template for Flask interface
```

---

## Dataset Pipeline

### Corpus Composition

The dataset merges structured parallel sentence sources into an aligned corpus:

- **Source Files:**
  - `parallel-corpus.xlsx`: English-Urdu sentence pairs.
  - `english-corpus.txt` and `urdu-corpus.txt`: Parallel line-by-line sentence text.
- **Combined Corpus Size:** 54,689 parallel sentence pairs.

### Preprocessing and Tokenization

1. **Null and Anomaly Handling:** Strips whitespace, removes empty entries, and normalizes string encodings.
2. **Boundary Token Injection:** Adds `<start>` and `<end>` markers to both source (English) and target (Urdu) sentences for sequence delimiter learning.
3. **Vocabulary and Tokenization:**
   - Word-level tokenization built using Keras `Tokenizer`.
   - Serialized tokenizers stored in `data/artifacts/`.
4. **Length Distribution and Padding:**
   - English average length: ~10.8 tokens (max: 432 tokens).
   - Urdu average length: ~12.6 tokens (max: 950 tokens).
   - 95th percentile thresholding sets sequence truncation/padding limits to **38 tokens** for both English and Urdu (configured in `config.yaml`).

---

## Model Architectures

### Baseline Recurrent Models

Implemented in TensorFlow/Keras:

1. **SimpleRNN Seq2Seq:** Unidirectional RNN encoder and decoder with embedding layers.
2. **Bidirectional RNN / Bi-LSTM:** Bidirectional encoder capturing forward and backward context, feeding into a recurrent decoder.
3. **LSTM Seq2Seq with Teacher Forcing:** Multi-unit LSTM network with embedding layers, using shifted target tokens during training to stabilize convergence.

### Custom Transformer Models

- Built with custom multi-head attention layers, layer normalization, residual connections, and feed-forward sub-layers.
- Evaluated under two configurations:
  - Without positional encodings.
  - With sinusoidal positional encodings.

### Fine-Tuned mBART-50

- **Base Model:** `facebook/mbart-large-50-many-to-many-mmt`.
- **Language Codes:** Source: `en_XX`, Target: `ur_PK`.
- **Training Configuration:**
  - Fine-tuned using Hugging Face `Trainer` and `TrainingArguments`.
  - Learning rate: `2e-5` with weight decay `0.01`.
  - Mixed-precision training with checkpoint saving per epoch.
  - Subword BPE tokenization handling out-of-vocabulary terms and morphological variants.

---

## Evaluation and Results

Models were evaluated on held-out test splits using BLEU (with smoothing) and ROUGE metrics:

| Model Architecture | BLEU Score | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :---: | :---: | :---: | :---: |
| SimpleRNN Seq2Seq | 0.2081 | 0.4612 | 0.1853 | 0.4397 |
| Bidirectional RNN / Bi-LSTM | 0.2443 | 0.5571 | 0.3056 | 0.5371 |
| LSTM Seq2Seq (Teacher Forcing) | 0.3342 | 0.5886 | 0.3207 | 0.5768 |
| Custom Transformer (w/o Pos. Encoding) | 0.1693 | 0.5200 | 0.2155 | 0.4988 |
| Custom Transformer (w/ Pos. Encoding) | 0.0831 | 0.2990 | 0.0585 | 0.2969 |
| **Fine-Tuned mBART-50** | **0.4520** | **0.7130** | **0.4810** | **0.6940** |

*Note: The fine-tuned mBART-50 transformer demonstrates superior performance by leveraging large-scale multilingual pre-trained representations, outperforming custom sequence-to-sequence baselines on low-resource Urdu generation.*

---

## Installation and Setup

### Prerequisites

- Python 3.8, 3.9, 3.10, or 3.11
- CUDA-compatible GPU (recommended for mBART-50 fine-tuning and inference)

### 1. Clone the Repository

```bash
git clone https://github.com/Abdul-Moiz-Shehzad/English-to-Urdu-Machine-Translation.git
cd English-to-Urdu-Machine-Translation
```

### 2. Set Up Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Flask Web Interface

The Flask application uses the fine-tuned mBART-50 model located at `./data/models/fine_tuned_mbart50` to perform translation.

```bash
python app.py
```

Open a web browser and navigate to:
```plaintext
http://127.0.0.1:5000/
```

### Streamlit LSTM Interface

The Streamlit application runs autoregressive token-by-token translation using the trained Keras LSTM model (`./data/models/LSTM_seq2seq.keras`) and saved joblib tokenizers.

```bash
streamlit run app_lstm.py
```

Open a web browser and navigate to the displayed local URL (typically `http://localhost:8501`).

### Jupyter Notebooks

To reproduce data preparation, training, and evaluation:

1. Launch Jupyter Lab / Notebook:
   ```bash
   jupyter notebook
   ```
2. Run notebooks in the following order:
   - `notebooks/research.ipynb`: Preprocesses raw data, generates artifacts, trains baseline models, and outputs comparison metrics.
   - `notebooks/finetune_mbart50.ipynb`: Fine-tunes `facebook/mbart-large-50-many-to-many-mmt` on GPU.
   - `notebooks/testing_mbart50.ipynb`: Tests sample translations using the fine-tuned checkpoint.

---

## Configuration

Preprocessing sequence limits are configured in `config.yaml`:

```yaml
Preprocessing:
  Padding:
    English: 38
    Urdu: 38
```

The configuration is loaded dynamically via `src.utils.config_loader.load_config`, which resolves file paths relative to the project root.

---

## References

1. Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 311–318.
2. Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*. In Text Summarization Branches Out, pp. 74–81.
3. Tang, Y., Tran, C., Li, X., Chen, P.-J., Goyal, N., Chaudhary, V., Gu, J., & Fan, A. (2020). *Multilingual Translation with Extensible Multilingual Pretraining and Finetuning*. arXiv preprint arXiv:2008.00401.
4. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS 2017).
5. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks*. Advances in Neural Information Processing Systems (NeurIPS 2014).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete details.
