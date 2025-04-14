import torch.nn as nn
from transformers import DistilBertModel

# Custom classifier built on top of DistilBERT for sentiment classification
class CustomDistilBertForSequenceClassification(nn.Module):
    def __init__(self, dropout=0.3, num_labels=3):
        super(CustomDistilBertForSequenceClassification, self).__init__()
        
        # Load pre-trained DistilBERT model
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Linear layer before final classifier (acts like a hidden dense layer)
        self.pre_classifier = nn.Linear(768, 768)  # 768 is the hidden size of DistilBERT
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Final classifier layer that maps to the number of output labels
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        # Get hidden states from DistilBERT
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the output corresponding to the [CLS] token (first token)
        hidden_state = distilbert_output[0]  # shape: (batch_size, seq_length, hidden_size)
        pooled_output = hidden_state[:, 0]   # shape: (batch_size, hidden_size)
        
        # Pass through a hidden linear layer + ReLU + Dropout
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        
        # Final classification layer to get logits
        logits = self.classifier(pooled_output)
        
        return logits
