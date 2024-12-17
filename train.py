import yaml
import os
import matplotlib.pyplot as plt
from dataloader import CustomDataset
from models import SimpleCNN, ResNet, VisionTransformer
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import torch.nn as nn

# Mapping activation function names to PyTorch classes
activation_mapping = {
    "ReLU": nn.ReLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
    "LeakyReLU": nn.LeakyReLU
}

activation_fn = activation_mapping[config['model'].get('activation_fn', 'ReLU')]

model = SimpleCNN(
    num_classes=config['model']['num_classes'],
    num_conv_layers=config['model']['num_conv_layers'],
    conv_padding=config['model']['conv_padding'],
    num_hidden_layers=config['model']['num_hidden_layers'],
    hidden_layer_sizes=config['model']['hidden_layer_sizes'],
    activation_fn=activation_fn
).to(config['training']['device'])


def check_save_path(path):
    """Check if save path exists or raise an error."""
    if not os.path.exists(path):
        print(f"Save path '{path}' does not exist. Creating it...")
        os.makedirs(path)
    else:
        raise FileExistsError(f"Save path '{path}' already exists. Please specify a new path.")

def plot_loss(train_losses, val_losses, save_path):
    """Plot and save the loss graph."""
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs. Epochs")
    plt.savefig(os.path.join(save_path, "loss_plot.png"))
    plt.close()

def train(config_path, save_path):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Check and prepare save path
    check_save_path(save_path)

    # Load datasets
    train_dataset = CustomDataset(config['data']['train_data'], config['data']['train_labels'])
    val_dataset = CustomDataset(config['data']['val_data'], config['data']['val_labels'])
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Initialize model, criterion, and optimizer
    model_cls = globals()[config['model']['name']]
    model = model_cls(num_classes=config['model']['num_classes']).to(config['training']['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training loop with early stopping
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = config['training']['patience']
    patience_counter = 0

    for epoch in range(config['training']['epochs']):
        # Training
        model.train()
        epoch_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config['training']['device']), targets.to(config['training']['device'])
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config['training']['device']), targets.to(config['training']['device'])
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)

        # Check for early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Print progress
        print(f"Epoch {epoch + 1}, Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

    # Save loss plot
    plot_loss(train_losses, val_losses, save_path)

if __name__ == "__main__":
    # Example usage
    train("config.yaml", save_path="output_dir")
