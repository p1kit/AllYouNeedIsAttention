import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from all_you_need_is_attention.config.transformer_config import TransformerConfig
from all_you_need_is_attention.dataset.dataset import TranslationDataset
from all_you_need_is_attention.models.base_transformer import BaseTransformer

# Define hyper-parameters
config = TransformerConfig()
num_epochs = 500
batch_size = 32
learning_rate = 0.05

# Initialize model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaseTransformer.from_config(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Load training data
train_dataset = TranslationDataset("train")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load validation data
val_dataset = TranslationDataset("val")
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Train the model
for epoch in range(num_epochs):
    # Set model to training mode
    model.train()

    # Initialize variables for tracking loss and accuracy
    train_loss = 0.0
    train_acc = 0.0
    num_examples = 0

    # Iterate over batches of training data
    for batch in train_loader:
        # Get inputs and targets for the batch
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass through the model
        outputs = model.forward(inputs, targets[:, :-1])

        # Compute loss and accuracy
        loss = criterion(outputs.view(-1, config.vocab_size), targets[:, 1:].contiguous().view(-1))
        train_loss += loss.item() * inputs.size(0)
        num_examples += inputs.size(0)
        train_acc += (outputs.argmax(dim=-1) == targets[:, 1:].contiguous()).sum().item()

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

    # Compute training loss and accuracy
    train_loss /= num_examples
    train_acc /= num_examples

    # Evaluate the model on the validation set
    model.eval()

    with torch.no_grad():
        # Initialize variables for tracking loss and accuracy
        val_loss = 0.0
        val_acc = 0.0
        num_examples = 0

        # Iterate over batches of validation data
        for batch in val_loader:
            # Get inputs and targets for the batch
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            # Forward pass through the model
            outputs = model.forward(inputs, targets[:, :-1])

            # Compute loss and accuracy
            loss = criterion(outputs.view(-1, config.vocab_size), targets[:, 1:].contiguous().view(-1))
            val_loss += loss.item() * inputs.size(0)
            num_examples += inputs.size(0)
            print(inputs.size(0))
            val_acc += (outputs.argmax(dim=-1) == targets[:, 1:].contiguous()).sum().item()

        # Compute validation loss and accuracy
        val_loss /= num_examples
        val_acc /= num_examples

    # Print the training and validation loss and accuracy
    print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")

# Save the model weights
torch.save(model.state_dict(), "saved_weights.pt")
