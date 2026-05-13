# 🤗 Hugging Face ML Pipeline

A modular machine learning pipeline for **sentiment analysis** using Hugging Face Transformers and PyTorch. Built to demonstrate end-to-end model fine-tuning, evaluation, and inference using the Trainer API.

---

## 📌 Project Overview

This project fine-tunes a pre-trained **DistilBERT** transformer model on the **IMDb movie review dataset** for binary sentiment classification (positive/negative). It showcases a clean, reproducible ML pipeline covering data loading, tokenisation, training, and inference.

**Use case:** Given a piece of text, the model predicts whether the sentiment is positive or negative — a technique applicable to customer feedback, reviews, and social media analysis.

---

## 🚀 Features

- ⚡ Fine-tune `distilbert-base-uncased` on the IMDb dataset using Hugging Face `Trainer` API
- 🧹 Automatic dataset tokenisation and preprocessing with `datasets` library
- 🔍 Quick inference pipeline using Hugging Face `pipeline()` abstraction
- 📁 Modular structure separating training (`train.py`) and evaluation (`evaluate.py`)
- 💾 Results and logs saved automatically during training

---

## 🧠 Model & Dataset

| Component | Detail |
|-----------|--------|
| Base Model | `distilbert-base-uncased` |
| Dataset | IMDb (via Hugging Face Datasets) |
| Task | Binary Sentiment Classification |
| Training Samples | 500 (subset for demonstration) |
| Epochs | 1 |
| Batch Size | 8 |

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Deep Learning:** PyTorch
- **NLP / Transformers:** Hugging Face `transformers`, `datasets`
- **Model:** DistilBERT (distilbert-base-uncased)

---

## 📂 Project Structure

```
hf-ml-pipeline/
│
├── train.py           # Fine-tuning pipeline using Trainer API
├── evaluate.py        # Model evaluation and inference
├── requirements.txt   # Project dependencies
└── README.md
```

---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/prekshashanbhag123/hf-ml-pipeline.git
cd hf-ml-pipeline
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python train.py
```

**4. Run evaluation / inference**
```bash
python evaluate.py
```

---

## 💡 Example Inference

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("This project is amazing!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

---

## 🌱 What I Learned

- How to load and preprocess NLP datasets using Hugging Face `datasets`
- How transformer tokenisation works (truncation, padding, attention masks)
- How to fine-tune a pre-trained model using the `Trainer` API
- The difference between using a pipeline for quick inference vs. a full training loop

---

## 🔮 Future Improvements

- Train on the full IMDb dataset (25,000 samples) for better accuracy
- Add evaluation metrics (accuracy, F1 score) to `evaluate.py`
- Experiment with other transformer models (BERT, RoBERTa)
- Add a simple web interface using Gradio or Streamlit

---

## 👩‍💻 Author

**Preksha Manjunath Shanbhag**  
MSc Artificial Intelligence — Queen's University Belfast  
[LinkedIn](https://www.linkedin.com/in/preksha-shanbhag) | [GitHub](https://github.com/prekshashanbhag123)
