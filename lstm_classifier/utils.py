import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Function to clean and normalize raw text
def normalize_text(text):
    text = text.lower()  # Convert to lowercase

    # Replace common contractions with full words
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

    tokens = word_tokenize(text)  # Tokenize the text
    stops = set(stopwords.words('english'))  # Load stop words

    # Remove stop words and non-alphabetic tokens
    filtered_tokens = [
        token for token in tokens
        if token not in stops and token.isalpha()
    ]

    lemmatizer = WordNetLemmatizer()
    # Lemmatize tokens (convert to base form)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    return ' '.join(lemmatized_tokens)  # Return cleaned sentence

# Read lines from a text file
def read_txt_file(file_path):
    with open(file_path, "r") as f:
        return f.readlines()

# Build vocabulary from dataset with minimum word frequency threshold
def create_vocab(lines, min_freq):
    total_word_counts = []
    word2index = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    word2count = {}

    for sample in lines:
        text = sample.split("\t")[0]
        word_count = len(text.split())
        total_word_counts.append(word_count)

        for w in text.split():
            if w not in word2count:
                word2count[w] = 1
            else:
                word2count[w] += 1

    # Add words meeting frequency threshold to vocabulary
    for w, c in word2count.items():
        if c >= min_freq:
            word2index[w] = len(word2index)

    return word2index, total_word_counts

# Convert numeric sentiment label into sentiment category
def convert_to_sentiment(label):
    if (int(label) <= 2):
        return 'negative'
    elif (int(label) == 3):
        return 'neutral'
    else:
        return 'positive'

# Preprocess and prepare dataset from CSV file
def prepare_dataset(csv_file):
    print("Preparing dataset...")

    df = pd.read_csv(csv_file)  # Load data from CSV
    df.columns = ['label', 'text']  # Rename columns
    df['label'] = df['label'].apply(convert_to_sentiment)  # Map to sentiment
    sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['label'].map(sentiment_mapping)  # Convert to numerical labels
    dataset = []

    # Normalize text and format label
    for _, row in df.iterrows():
        text = normalize_text(str(row["text"]))
        label = str(row["label"])
        sample = f"{text}\t{label}"
        dataset.append(sample)

    np.random.shuffle(dataset)  # Shuffle data
    word2index, total_word_counts = create_vocab(dataset, 25)  # Create vocabulary
    print(f"Vocab size:", len(word2index))

    # Split dataset into train/val/test
    n_train = int(len(dataset) * 0.8)
    n_val = int(len(dataset) * 0.1)
    dataset = {
        "train": dataset[:n_train],
        "val": dataset[n_train : n_train + n_val],
        "test": dataset[-n_val:],
    }
    return dataset, word2index, total_word_counts

# Training function
def train(dloader, model, criterion, optimizer):
    model.train()
    train_losses, train_accs = [], []

    for batch in tqdm(dloader, desc="Training"):
        y = batch["label"]  # True labels
        logits = model(batch)  # Forward pass
        loss = criterion(logits, y)  # Compute loss
        optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        train_losses.append(loss.item())  # Track loss
        preds = torch.argmax(logits, -1)  # Get predictions
        train_accs.append((preds == y).float().mean().item())  # Accuracy

    avg_loss = np.array(train_losses).mean()
    avg_acc = np.array(train_accs).mean()

    print(f"Train Loss: {avg_loss:.4f} | Train Accuracy: {avg_acc:.4f}")
    return avg_loss, avg_acc

# Evaluation function (no gradient calculation)
@torch.no_grad()
def test(dloader, model, criterion):
    model.eval()
    test_losses, test_accs = [], []

    for batch in tqdm(dloader, desc="Evaluating"):
        y = batch["label"]  # True labels
        logits = model(batch)  # Forward pass
        loss = criterion(logits, y)  # Compute loss

        test_losses.append(loss.item())  # Track loss
        preds = torch.argmax(logits, -1)  # Predictions
        test_accs.append((preds == y).float().mean().item())  # Accuracy

    avg_loss = np.array(test_losses).mean()
    avg_acc = np.array(test_accs).mean()

    print(f"Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")
    return avg_loss, avg_acc
