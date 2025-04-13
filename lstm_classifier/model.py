import torch

class LSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout, n_layers, n_classes):
        super(LSTMClassifier, self).__init__()
        
        # Embedding layer: converts word indices to dense vectors
        self.embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer: processes sequences of embeddings
        # batch_first=True means input/output tensors are of shape (batch, seq, feature)
        self.rnn = torch.nn.LSTM(
            embedding_dim,     # input size (from embedding)
            hidden_dim,        # hidden state size
            n_layers,          # number of LSTM layers
            dropout=0.3,       # dropout between LSTM layers (not applied if n_layers == 1)
            batch_first=True
        )
        
        # Dropout layer: regularization to prevent overfitting
        self.dropout = torch.nn.Dropout(dropout)
        
        # Fully connected output layer: maps hidden state to output classes
        self.out = torch.nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # Embedding the input word indices (x["input_ids"]) to dense vectors
        embedded = self.embedding_layer(x["input_ids"])
        
        # Packing the embedded sequences to handle variable lengths efficiently
        pcked = torch.nn.utils.rnn.pack_padded_sequence(
            embedded,
            x["length"],           # actual lengths of sequences (no padding)
            enforce_sorted=False,  # allows input not to be sorted by length
            batch_first=True
        )

        # Passing packed sequence through LSTM
        output, _ = self.rnn(pcked)
        
        # Unpacking the sequence back to tensor form
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            output,
            padding_value=2,       # value to use for padding
            batch_first=True
        )

        # Selecting the output of the last valid time step for each sequence
        output = output[range(len(output)), x["length"] - 1]

        # Applying dropout for regularization
        output = self.dropout(output)

        # Final linear layer to get logits for each class
        return self.out(output)
