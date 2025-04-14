import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import DistilBertTokenizerFast
from utils import preprocess_text, convert_to_sentiment

# Custom Dataset class for handling Yelp reviews
class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.dataset = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Define mapping from sentiment strings to numeric labels
        self.label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}

    def __len__(self):
        # Return total number of samples
        return len(self.dataset)

    def __getitem__(self, idx):
        # Extract review and corresponding sentiment at given index
        review_text = self.dataset.iloc[idx, 1]
        sentiment = self.dataset.iloc[idx, 0]
        labels = self.label_dict[sentiment]

        # Tokenize review text using DistilBERT tokenizer
        encoding = self.tokenizer.encode_plus(
            review_text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        # Return dictionary containing tokenized inputs and label
        return {
            'review_text': review_text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),  # Attention mask to ignore padding tokens
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# Function to read CSV, preprocess text, tokenize, and return DataLoaders
def read_data(csv_file, max_seq_length=512):
    model_name = "distilbert-base-uncased"

    # Read the CSV file and rename columns
    df = pd.read_csv(csv_file)
    df.columns = ['label', 'text']

    # Convert numeric labels to sentiment (1-5 → pos/neg/neutral)
    df['label'] = df['label'].apply(convert_to_sentiment)

    # Clean the review text
    df['text'] = df['text'].apply(preprocess_text)

    # Preview the preprocessed data
    print(df.head())

    # Print dataset class distribution
    print("\nDataset Statistics:")
    print(df['label'].value_counts())

    # Split dataset: 70% train, 15% val, 15% test
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    test_size = len(df) - train_size - val_size

    # Load tokenizer and create dataset object
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    review_dataset = ReviewDataset(df, tokenizer, max_seq_length)

    # Randomly split dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        review_dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders for model training and evaluation
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    return train_loader, val_loader, test_loader
