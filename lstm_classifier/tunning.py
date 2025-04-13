# Import required libraries
import torch
import random
import numpy as np
from itertools import product
from dataloader import make_dataloader
from model import LSTMClassifier
from utils import prepare_dataset, train, test

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to perform hyperparameter tuning
def hyperparameter_tunning(csv_file):
    # Prepare the dataset and vocabulary
    dataset, word2index, _ = prepare_dataset(csv_file)

    # Define the hyperparameter grid to search over
    param_grid = {
        'embedding_dim': [128, 256],
        'hidden_dim': [64, 128],
        'dropout': [0.3, 0.5],
    }

    # Create all combinations of parameters and shuffle them
    combinations = list(product(*param_grid.values()))
    random.shuffle(combinations)

    EPOCHS = 5  # Number of epochs for each trial
    best_val_acc = 0  # Track best validation accuracy
    best_pair = {}    # Store best hyperparameter combination

    # Iterate through each combination
    for i, combo in enumerate(combinations):
        print(f"\n🔁 Trial {i+1}: Params = {combo}")
        embedding_dim, hidden_dim, dropout = combo

        # Initialize model with current hyperparameters
        model = LSTMClassifier(len(word2index), embedding_dim, hidden_dim, dropout, 2, 3)
        model.to(device)

        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-3)

        # Create dataloaders for training and validation
        train_dloader = make_dataloader(dataset["train"], word2index, 256, 16, device)
        val_dloader = make_dataloader(dataset["val"], word2index, 256, 16, device)

        train_accs, val_accs = [], []

        # Train the model for a few epochs
        for epoch in range(EPOCHS):
            print(f"===Epoch {epoch}===")
            print("Training...")
            train_loss, train_acc = train(train_dloader, model, criterion, optimizer)

            print("Validating...")
            val_loss, val_acc = test(val_dloader, model, criterion)

            train_accs.append(train_acc)
            val_accs.append(val_acc)

        # Calculate mean validation accuracy over epochs
        val_acc = np.mean(val_accs)

        # Update best combination if current one performs better
        if(val_acc > best_val_acc):
            best_val_acc = val_acc
            best_pair = combo
            print(f"best pair currently is {combo}")
            
    return best_pair  # Return the best performing hyperparameter combo