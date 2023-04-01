import torch
from torch.utils.data import DataLoader

from all_you_need_is_attention.config.transformer_config import TransformerConfig
from all_you_need_is_attention.dataset.dataset import TranslationDataset
from all_you_need_is_attention.models.base_transformer import BaseTransformer
from machine_translation_task.training_script import criterion

# Define hyper-parameters
config = TransformerConfig()

# Initialize model and load saved weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaseTransformer.from_config(config).to(device)
model.load_state_dict(torch.load("saved_weights.pt", map_location=device))

# Load test data
test_dataset = TranslationDataset("test")
test_loader = DataLoader(test_dataset, batch_size=32)

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_loss = 0.0
    test_acc = 0.0
    num_examples = 0

    # Iterate over batches of test data
    for batch in test_loader:
        # Get inputs and targets for the batch
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        # Forward pass through the model
        outputs = model.forward(inputs, targets[:, :-1])

        # Compute loss and accuracy
        loss = criterion(outputs.view(-1, config.vocab_size), targets[:, 1:].contiguous().view(-1))
        test_loss += loss.item() * inputs.size(0)
        num_examples += inputs.size(0)
        test_acc += (outputs.argmax(dim=-1) == targets[:, 1:].contiguous()).sum().item()

    # Compute test loss and accuracy
    test_loss /= num_examples
    test_acc /= num_examples

# Print the test loss and accuracy
print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.4f}")
