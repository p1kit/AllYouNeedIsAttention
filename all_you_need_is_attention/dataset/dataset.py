import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, split):
        # Load data from file
        if split == "train":
            self.data = self.load_data("data/train.txt")
        elif split == "val":
            self.data = self.load_data("data/val.txt")
        elif split == "test":
            self.data = self.load_data("data/val.txt")
        else:
            raise ValueError("Invalid split name")

        # Create vocabulary
        self.vocab = self.create_vocab(self.data)

        # Convert data to numerical form
        self.data = self.convert_data(self.data, self.vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {"input": self.data[index]["input"], "target": self.data[index]["target"]}

    def load_data(self, filename):
        with open(filename, "r") as f:
            data = [line.strip().split("\t") for line in f]
        return data

    def create_vocab(self, data):
        # Collect unique words in the data
        words = set()
        for pair in data:
            words.update(pair[0].split() + pair[1].split())

        # Add special tokens to the vocabulary
        vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}

        # Add words to the vocabulary
        for i, word in enumerate(sorted(words)):
            vocab[word] = i + 3

        return vocab

    def convert_data(self, data, vocab):
        # Convert data to numerical form
        converted_data = []
        for pair in data:
            input_ids = [vocab["<sos>"]] + [vocab[word] for word in pair[0].split()] + [vocab["<eos>"]]
            target_ids = [vocab["<sos>"]] + [vocab[word] for word in pair[1].split()] + [vocab["<eos>"]]
            converted_data.append({"input": torch.tensor(input_ids), "target": torch.tensor(target_ids)})
        return converted_data
