
# English-to-Urdu-Machine-Translation
## Table of Contents

- [Overview](#overview)  
- [Dataset](#dataset)  
- [Preprocessing](#preprocessing)  
- [Models](#models)  
- [Evaluation](#evaluation)  
- [Usage](#usage)  
- [License](#license)  
- [References](#references)  

## Overview

English-to-Urdu-Machine-Translation is a deep learning project for converting English text into Urdu. It implements sequence-to-sequence architectures (RNN, bidirectional RNN, LSTM, Transformer) and fine-tunes a pretrained multilingual transformer (mBART-50) on a dedicated English-Urdu corpus. The system includes data preprocessing, model training, evaluation, and a production-ready REST API for real-time translation. All experiments are tracked using Weights & Biases (W&B) to monitor training metrics and model performance.

- **Models:** Implements RNN, Bi-RNN, LSTM, Transformer architectures with attention, and a fine-tuned mBART-50 transformer.  
- **Deployment:** Provides a Flask-based web API for translation and a Docker configuration for containerized deployment.  
- **Monitoring:** Uses Weights & Biases for logging experiments and visualizing performance.  

## Dataset

The dataset consists of **54,689** parallel English–Urdu sentence pairs. The raw corpus was cleaned and normalized to remove extraneous symbols and ensure consistency (lowercasing, punctuation removal, etc.). After preprocessing, the data is split into **training (80%)** and **test (20%)** subsets for model development and evaluation.

- **Size:** 54,689 aligned sentence pairs (English & Urdu).  
- **Splits:** 80% (43,751 pairs) for training, 20% (10,938 pairs) for testing.  
- **Cleaning:** Removed noisy or unaligned sentences and applied language-specific normalization.  

## Preprocessing

Each sentence pair is prepared through several preprocessing steps to make it suitable for training:

- **Normalization:** Lowercase text and strip unwanted characters (digits, HTML tags, etc.).  
- **Tokenization:** Split sentences into tokens. We use subword tokenization (e.g., Byte-Pair Encoding or SentencePiece) to handle rare words and construct a shared vocabulary.  
- **Vocabulary Construction:** Build a consistent source (English) and target (Urdu) vocabulary for embedding and model input.  
- **Formatting:** Convert cleaned token sequences into numerical IDs and pad/truncate them for batch training.

These steps ensure that all models receive clean, standardized input for effective learning.

## Models

We train and compare several neural translation models:

- **RNN (Seq2Seq):** A standard encoder-decoder with a unidirectional recurrent neural network and attention mechanism.  
- **Bi-RNN:** A bidirectional RNN encoder (processing input in both directions) with an RNN decoder and attention.  
- **LSTM:** Encoder-decoder with LSTM units (with gating) to capture long-range dependencies.  
- **GRU:** Encoder-decoder with Gated Recurrent Units, an alternative gating architecture.  
- **mBART-50 (Transformer):** A pretrained multilingual encoder-decoder from Facebook AI. The mBART-50 model, introduced by Tang *et al.* (2020), supports translation across 50 languages. We fine-tune this transformer on our English-Urdu dataset to leverage transfer learning from multilingual pretraining.

Each model is implemented in PyTorch (using Hugging Face Transformers for mBART) and trained until convergence on the training data. All RNN/LSTM/GRU models use additive attention to align source and target sequences. 

## Evaluation

We evaluate translation quality on the test set using standard automated metrics:

- **BLEU (Bilingual Evaluation Understudy):** Measures n-gram precision.  
- **ROUGE-L:** Measures longest common subsequence recall.  
- **METEOR:** Computes unigram precision/recall with synonym matches.  
- **TER (Translation Edit Rate):** Calculates the number of edits needed (lower is better).  

Higher BLEU, ROUGE indicate better performance. In our experiments, the fine-tuned mBART-50 model achieves the highest scores, with **BLEU = 0.45**, **ROUGE-L = 0.69** on the test set. These results show that multilingual pre-training significantly improves English–Urdu translation quality compared to basic RNN/LSTM baselines.

## Usage

To run the translation service locally or in Docker, follow these steps:

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/yourusername/english-urdu-translation.git
   cd english-urdu-translation
   ```  
2. **Install dependencies:** (requires Python 3.8+)  
   ```bash
   pip install -r requirements.txt
   ```  
3. **Start the Flask server:**  
   ```bash
   python app.py
   ```  
   The API will run at `http://localhost:5000`. Send a POST request with JSON `{"text": "Your English sentence"}` to `/translate` to get an Urdu translation.


The project logs all training runs and metrics to Weights & Biases by default. To enable W&B tracking, sign up at [wandb.ai](https://wandb.ai) and set the `WANDB_API_KEY` environment variable before training.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. ACL.  
- Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*. ACL.  
- Banerjee, S., & Lavie, A. (2005). *METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments*. ACL Workshop.  
- Snover, M., Dorr, B., Schwartz, R., Micciulla, L., & Makhoul, J. (2006). *A Study of Translation Edit Rate with Targeted Human Annotation*. AMTA.  
- Tang, Y., Tran, C., Li, X., Chen, P.-J., Goyal, N., Chaudhary, V., Gu, J., & Fan, A. (2020). *Multilingual Translation with Extensible Multilingual Pretraining and Finetuning*. arXiv:2008.00401.  
- Weights & Biases (W&B). [Online]. Available: https://wandb.ai.
