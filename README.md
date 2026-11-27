# Inference-Time Optimization for Large Article Summarization

An NLP framework designed to overcome transformer context-window constraints during abstractive and extractive document summarization through semantic clustering.

---

## 📌 Problem & Motivation

Pretrained sequence-to-sequence transformers (BART, DistilBART, Pegasus, RoBERTa) are constrained by fixed positional embedding boundaries (typically 1024 tokens). Standard approaches often truncate excess text, discarding vital information from long documents. 

This project provides an inference-time optimization pipeline that preserves document semantics without requiring costly model retraining or fine-tuning.

---

## 🏗️ Architecture & Workflow

1. **Sentence Segmentation**: The input text is decomposed into discrete sentence units using NLTK tokenizers.
2. **Dense Representation**: Sentences are mapped to dense vector embeddings using `sentence-transformers/all-MiniLM-L6-v2` with mean pooling.
3. **Semantic Clustering**: The target cluster count is computed dynamically based on the input token volume and the model capacity. Sentences are clustered using Agglomerative Hierarchical Clustering or K-Means.
4. **Sub-document Summarization**: Each sentence cluster is individually summarized using the selected transformer model.
5. **Meta-Synthesis**: Individual cluster summaries are consolidated and processed through a final summarization pass to produce a unified, coherent abstract.

```
Long Input Document
        │
        ▼
Sentence Segmentation (NLTK)
        │
        ▼
Dense Sentence Embeddings (all-MiniLM-L6-v2)
        │
        ▼
Agglomerative Semantic Clustering
 ├── Cluster 1 ──► Local Summarizer (BART / DistilBART)
 ├── Cluster 2 ──► Local Summarizer (BART / DistilBART)
 └── Cluster N ──► Local Summarizer (BART / DistilBART)
        │
        ▼
Concatenation & Final Summary Synthesis
        │
        ▼
Final Coherent Summary
```

---

## 🚀 Key Features

- **Inference-Time Scaling**: Efficiently summarizes documents well beyond 1024 tokens without architectural modifications.
- **Model Flexibility**: Supports multiple transformer backends including BART Large, DistilBART, Pegasus, and RoBERTa.
- **Interactive Web Interface**: Streamlit application with adjustable summary length, model selection, and real-time generation progress.
- **Lightweight & Fast**: Sentence clustering executes in seconds using optimized vector operations and scikit-learn.

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.9 - 3.12
- PyTorch & Transformers

### 1. Clone the Repository
```bash
git clone https://github.com/sabhyatadas/cluster-summarization-nlp.git
cd cluster-summarization-nlp
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download Tokenizer Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## 🖥️ Usage

### Run the Web Interface
Launch the interactive Streamlit dashboard:
```bash
streamlit run Large_Article_Summarization.py
```

For multi-article processing:
```bash
streamlit run tmp_pages/Multi_Article_Summarization.py
```

### Run via Python CLI
Execute a sample summarization pipeline:
```bash
python summarize.py
```

Run test suite:
```bash
python test.py
```

---

## 📂 Project Structure

```
cluster-summarization-nlp/
├── Large_Article_Summarization.py   # Primary Streamlit dashboard
├── summarize.py                     # Core clustering & summarization pipeline
├── test.py                          # Validation test suite
├── try.py                           # Quick demonstration script
├── requirements.txt                 # Project dependencies
├── configs/
│   └── model_config.py              # Embedding & model configuration
├── data/
│   ├── large_sum.py                 # Long article dataset loader
│   └── multi_sum.py                 # Multi-document dataset loader
├── models/
│   └── summarizers.py               # Model loading & sequence chunking
├── utils/
│   ├── clustering.py                # Agglomerative & K-Means clustering
│   └── sentence_embedding.py        # Dense sentence vector generation
├── tmp_pages/
│   └── Multi_Article_Summarization.py # Multi-article Streamlit interface
└── v0/                              # Legacy baseline implementation & examples
```

---

## 👤 Author

**Sabhyata Das**
- GitHub: [@sabhyatadas](https://github.com/sabhyatadas)
