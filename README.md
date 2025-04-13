# 📝 Yelp Sentiment Classifier: DistilBERT & LSTM

This project performs sentiment classification on Yelp restaurant reviews using two deep learning models:

- ✅ **DistilBERT** (Transformer-based classifier)
- 🔁 **LSTM** (Recurrent neural network-based classifier)

Both models are implemented from scratch using PyTorch.

---

## 📁 Project Structure

```
.
├── data/                            # Folder containing dataset (CSV files)
├── distilbert_classifier/          # DistilBERT classifier implementation
│   ├── dataloader.py
│   ├── main.py
│   ├── model.py
│   ├── tunning.py
│   ├── utils.py
│   ├── results/
│   └── AIGC_5000_ADL_Final_Project_DistilBERT.ipynb
├── lstm_classifier/                # LSTM classifier implementation
│   ├── dataloader.py
│   ├── main.py
│   ├── model.py
│   ├── tunning.py
│   ├── utils.py
│   ├── results/
│   └── AIGC_5000_ADL_Final_Project_LSTM.ipynb
├── main.ipynb                      # Comparison notebook
├── .gitignore
└── requirements.txt
```

---

## ⚙️ Installation Instructions

1. **Clone the Repository**

```bash
git clone <your-repository-url>
cd <project-folder>
```

2. **Create a Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

3. **Install Python Dependencies**

```bash
pip install -r requirements.txt
```

> If you don’t have a `requirements.txt`, generate it using:

```bash
pip freeze > requirements.txt
```

---

## 🚀 How to Run

### 📌 Step 1: Add Dataset

Place your Yelp review dataset (e.g., `yelp.csv`) inside the `data/` folder.  
Make sure both classifiers refer to the correct path in their respective `dataloader.py`.

---

### 🤖 Run DistilBERT Model

```bash
cd distilbert_classifier
python main.py
```

- Loads and tokenizes the Yelp reviews
- Trains and evaluates a DistilBERT-based sentiment classifier
- Modify hyperparameters in `tunning.py`

---

### 🔁 Run LSTM Model

```bash
cd lstm_classifier
python main.py
```

- Trains a custom LSTM classifier from scratch
- Outputs training logs and results to `results/`
- Tune hyperparameters via `tunning.py`

---

## ✍️ Author

- Shrey Patel
- Suyash Kulkarni
- Yash Modi
- Kaverappa K U

---
