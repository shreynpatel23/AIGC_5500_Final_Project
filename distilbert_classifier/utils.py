import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import NLP tools
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('omw-1.4')

from tqdm import tqdm

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# Text Preprocessing
# ========================
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Expand common contractions
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

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and non-alphabetic tokens
    stops = set(stopwords.words('english'))
    filtered_tokens = [
        token for token in tokens
        if token not in stops and token.isalpha()
    ]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Return the cleaned, lemmatized text
    return ' '.join(lemmatized_tokens)

# ========================
# Convert Numeric Rating to Sentiment Class
# ========================
def convert_to_sentiment(label):
    if (int(label) <= 2):
        return 'negative'
    elif (int(label) == 3):
        return 'neutral'
    else:
        return 'positive'

# ========================
# Training Loop
# ========================
def train(dloader, model, criterion, optimizer):
    model.train()  # Set model to training mode
    train_losses, train_accs = [], []

    for batch in tqdm(dloader, desc="Training"):
        # Move input and labels to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Record metrics
        train_losses.append(loss.item())
        preds = torch.argmax(logits, -1)
        train_accs.append((preds == labels).float().mean().item())

    # Compute average metrics
    avg_loss = np.array(train_losses).mean()
    avg_acc = np.array(train_accs).mean()

    print(f"Train Loss: {avg_loss:.4f} | Train Accuracy: {avg_acc:.4f}")
    return avg_loss, avg_acc

# ========================
# Evaluation Loop (Validation / Testing)
# ========================
@torch.no_grad()
def test(dloader, model, criterion):
    model.eval()  # Set model to evaluation mode
    test_losses, test_accs = [], []

    for batch in tqdm(dloader, desc="Evaluating"):
        # Move input and labels to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        # Record metrics
        test_losses.append(loss.item())
        preds = torch.argmax(outputs, dim=1)
        test_accs.append((preds == labels).float().mean().item())

    # Compute average metrics
    avg_loss = np.array(test_losses).mean()
    avg_acc = np.array(test_accs).mean()

    print(f"Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")
    return avg_loss, avg_acc
