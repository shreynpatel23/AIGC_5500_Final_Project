# Import necessary libraries
import os
import torch
import time
import matplotlib.pyplot as plt, pandas as pd
from dataloader import read_data
from model import CustomDistilBertForSequenceClassification
from transformers import DistilBertTokenizerFast
from pathlib import Path
from tunning import hyperparameter_tunning
from utils import train, test
import shap

# Set up directory to store model and results
log_dir = Path("/content/results")
log_dir.mkdir(parents=True, exist_ok=True)
best_val_loss = float("inf")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to save best model
def save_model(model):
    checkpoint_path = log_dir / "best_model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Best model saved to {checkpoint_path}")

# Define dataset path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "data", "data.csv")

# Perform manual grid search for best (max_seq_length, dropout) combination
best_pair = hyperparameter_tunning(data_dir)
max_seq_length, dropout = best_pair

# Load train, validation, and test dataloaders using best max_seq_length
train_loader, val_loader, test_loader = read_data(data_dir, max_seq_length)

# Initialize DistilBERT model with best dropout
model = CustomDistilBertForSequenceClassification(dropout=dropout)
model.to(device)

# Freeze base DistilBERT layers (only fine-tune classification head)
for param in model.distilbert.parameters():
    param.requires_grad = False

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Initialize tracking variables
num_epochs = 10
train_accs, val_accs, test_accs = [], [], []
train_losses, val_losses, test_losses = [], [], []
epoch_times = []
best_test_acc = 0
patience = 2
patience_counter = 0

# Training loop
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    epoch_start_time = time.time()

    # Training step
    print("Training...")
    train_loss, train_acc = train(train_loader, model, criterion, optimizer)

    # Validation step
    print("Validating...")
    val_loss, val_acc = test(val_loader, model, criterion)

    # Test step
    print("Testing...")
    test_loss, test_acc = test(test_loader, model, criterion)

    # Save metrics for plots and logs
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_accs.append(test_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    test_losses.append(test_loss)

    # Track epoch duration
    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch duration: {epoch_duration:.2f} seconds")
    epoch_times.append(epoch_duration)

    # Save best model based on test accuracy
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        patience_counter = 0
        save_model(model)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# ================== Plotting Results ==================

# Convert accuracies to percentage for plots
train_accs = [i * 100 for i in train_accs]
val_accs = [i * 100 for i in val_accs]
test_accs = [i * 100 for i in test_accs]

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accs, label="Train Accuracy")
plt.plot(range(1, num_epochs + 1), val_accs, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# Plot test accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), test_accs, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Test Accuracy vs Epoch")
plt.legend()
plt.show()

# Plot test loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Test Loss vs Epoch")
plt.legend()
plt.show()

# Save all metrics to a CSV file
data = {
    'Train Accuracy': train_accs,
    'Validation Accuracy': val_accs,
    'Test Accuracy': test_accs,
    'Train Loss': train_losses,
    'Validation Loss': val_losses,
    'Test Loss': test_losses,
    'Epoch Time': epoch_times
}
df = pd.DataFrame(data)
df.to_csv("distilbert_results.csv")

# ================== SHAP Analysis ==================

# Reload best model for explainability
model = CustomDistilBertForSequenceClassification()
model.load_state_dict(torch.load("/content/best_model.pt", map_location=torch.device('cpu')))
model.to(device)
model.eval()

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Define prediction function for SHAP
def predict_proba(texts):
    inputs = tokenizer.batch_encode_plus(
        texts.tolist(),
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    ).to(device)

    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.nn.functional.softmax(logits, dim=1)

    return probs.cpu().numpy()

# Select a few samples for SHAP analysis
sample_reviews = df["text"][9:12]
label = df['label'][9:12]

# Create SHAP masker and explainer for text
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(predict_proba, masker)

# Generate and visualize SHAP values
shap_values = explainer(sample_reviews)
shap.plots.text(shap_values)
