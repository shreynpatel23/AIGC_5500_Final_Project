# Import required libraries and modules
import pickle, os
import time
import torch, pandas as pd
import matplotlib.pyplot as plt
from utils import prepare_dataset, train, test
from dataloader import make_dataloader
from model import LSTMClassifier
from tunning import hyperparameter_tunning

# Path to the dataset
data_dir = os.path.join("data", "data.csv")

# Function to save model checkpoint
def save_cp(model):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    torch.save(model.state_dict(), "checkpoints/lstm_model.pt")

# Perform hyperparameter tuning to get best embedding_dim, hidden_dim, dropout
best_pair = hyperparameter_tunning(data_dir) 
embedding_dim, hidden_dim, dropout = best_pair

# Define number of training epochs and device (CPU/GPU)
EPOCHS = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare dataset and vocabulary index
dataset, word2index, _ = prepare_dataset(data_dir)

# Save the word2index mapping for reuse
with open("./data/word2index.pkl", "wb") as f:
    pickle.dump(word2index, f)

# Create dataloaders for training, validation, and test sets
train_dloader = make_dataloader(dataset["train"], word2index, 256, 16, device)
val_dloader = make_dataloader(dataset["val"], word2index, 256, 16, device)
test_dloader = make_dataloader(dataset["test"], word2index, 256, 16, device)

# Initialize LSTM model
model = LSTMClassifier(len(word2index), embedding_dim, hidden_dim, dropout, 2, 3)
model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-3)

# Lists to store performance metrics
train_accs, val_accs, test_accs = [], [], []
train_losses, val_losses, test_losses = [], [], []
epoch_times = []

# Variables for tracking early stopping
best_test_acc = 0
patience = 2
patience_counter = 0

# Training loop
for epoch in range(EPOCHS):
    print(f"===Epoch {epoch}===")
    epoch_start_time = time.time()

    # Training step
    print("Training...")
    train_loss, train_acc = train(train_dloader, model, criterion, optimizer)

    # Validation step
    print("Validating...")
    val_loss, val_acc = test(val_dloader, model, criterion)

    # Testing step
    print("Testing...")
    test_loss, test_acc = test(test_dloader, model, criterion)

    # Store results
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_accs.append(test_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    test_losses.append(test_loss)

    # Track epoch time
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"Epoch duration: {epoch_duration:.2f} seconds")
    epoch_times.append(epoch_duration)

    # Save best model based on test accuracy
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        patience_counter = 0
        save_cp(model)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Convert accuracy values to percentages
train_accs = [i * 100 for i in train_accs]
val_accs = [i * 100 for i in val_accs]
test_accs = [i * 100 for i in test_accs]

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), train_accs, label="Train Accuracy")
plt.plot(range(1, EPOCHS + 1), val_accs, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# Plot test accuracy per epoch
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), test_accs, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Test Accuracy vs Epoch")
plt.legend()
plt.show()

# Plot test loss per epoch
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Test Loss vs Epoch")
plt.legend()
plt.show()

# Save results to a CSV file for later analysis
data = {
    'Train Accuracy': train_accs,
    'Validation Accuracy': val_accs,
    'Test Accuracy': test_accs,
    'Train Loss': train_losses,
    'Validation Loss': val_losses,
    'Test Loss': test_losses
}
df = pd.DataFrame(data)
df.to_csv("lstm_results.csv")
