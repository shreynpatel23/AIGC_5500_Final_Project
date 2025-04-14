import torch
import random
import numpy as np
from itertools import product
from dataloader import read_data
from model import CustomDistilBertForSequenceClassification
from utils import train, test

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to perform manual hyperparameter tuning
def hyperparameter_tunning(csv_file):
    # Define grid of hyperparameters to tune
    param_grid = {
        'max_seq_length': [256, 512],  # Sequence length for tokenizer
        'dropout': [0.3, 0.5],         # Dropout rate for regularization
    }

    # Create all combinations of hyperparameter values
    combinations = list(product(*param_grid.values()))
    random.shuffle(combinations)  # Shuffle to randomize trial order
    EPOCHS = 5  # Number of epochs per trial

    best_val_acc = 0      # Track the best validation accuracy
    best_pair = None      # Store best performing (seq_length, dropout) pair

    # Loop through all hyperparameter combinations
    for i, combo in enumerate(combinations):
        print(f"\n🔁 Trial {i+1}: Params = {combo}")
        max_seq_length, dropout = combo

        # Load training and validation data with current max_seq_length
        train_loader, val_loader, _ = read_data(csv_file, max_seq_length)

        # Initialize the model with current dropout
        model = CustomDistilBertForSequenceClassification(dropout=dropout)
        model.to(device)

        # Freeze DistilBERT layers to train only the classification head
        for param in model.distilbert.parameters():
            param.requires_grad = False

        # Define loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        train_accs, val_accs = [], []

        # Train the model for a few epochs
        for epoch in range(EPOCHS):
            print(f"===Epoch {epoch}===")
            print("Training...")
            _, train_acc = train(train_loader, model, criterion, optimizer)

            print("Validating...")
            _, val_acc = test(val_loader, model, criterion)

            train_accs.append(train_acc)
            val_accs.append(val_acc)

        # Compute average validation accuracy over epochs
        val_acc = np.mean(val_accs)
        print(f"Validation Accuracy: {val_acc}")

        # Update best pair if current combo performs better
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_pair = combo
            print(f"Best pair currently is {combo}")

    return best_pair  # Return the best hyperparameter pair
