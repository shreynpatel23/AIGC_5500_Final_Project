import torch

class YelpReviews(torch.utils.data.Dataset):
    def __init__(self, data, word2index, max_sent_length, device):
        # Store the raw dataset, word-to-index mapping, max sentence length, and device (CPU or GPU)
        self.data = data
        self.word2index = word2index
        self.max_sent_length = max_sent_length
        self.device = device

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, index):
        # Get a single data sample
        sample = self.data[index]
        
        # Each sample is expected in the format: "text<TAB>label"
        text, label = sample.split("\t")
        label = int(label)  # convert label to integer
        
        # Convert the text to token indices
        input_ids, _text, text_length = self.text2indices(text)
        
        # Return the processed components as a dictionary
        return {
            "text": _text,  # original text with special tokens
            "input_ids": torch.tensor(input_ids).to(self.device),  # token ids as tensor
            "label": torch.tensor(label).to(self.device),  # label as tensor
            "length": text_length,  # actual sequence length (after adding <EOS>)
        }

    def text2indices(self, sentence):
        # Get special token IDs
        sos_id = self.word2index["<SOS>"]
        eos_id = self.word2index["<EOS>"]
        pad_id = self.word2index["<PAD>"]
        unk_id = self.word2index["<UNK>"]
        
        # Start with <SOS>
        input_ids, text = [sos_id], ["<SOS>"]
        
        # Convert each word in the sentence to its corresponding index
        for w in sentence.split():
            token_id = self.word2index.get(w, unk_id)  # use <UNK> if word not in vocab
            input_ids.append(token_id)
            if token_id == unk_id:
                text.append("<UNK>")
            else:
                text.append(w)

        # If the sentence is too long, truncate and add <EOS>
        if len(input_ids) >= self.max_sent_length:
            text = text[: self.max_sent_length]
            text.append("<EOS>")
            input_ids = input_ids[: self.max_sent_length]
            input_ids.append(eos_id)
            text_len = len(input_ids)

        else:
            # If the sentence is short, pad with <PAD> tokens
            to_add = self.max_sent_length - len(input_ids)
            text.append("<EOS>")
            input_ids.append(eos_id)
            text_len = len(input_ids)
            text.extend(["<PAD>"] * to_add)
            input_ids.extend([pad_id] * to_add)

        # Return input IDs, text (as string), and sequence length
        return input_ids, " ".join(text), text_len


def make_dataloader(dataset, word2index, max_sent_length, batch_size, device):
    # Create an instance of the custom YelpReviews dataset
    ds = YelpReviews(dataset, word2index, max_sent_length, device)
    
    # Return a PyTorch DataLoader for batching and shuffling
    return torch.utils.data.DataLoader(ds, batch_size, shuffle=True)
