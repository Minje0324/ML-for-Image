import torch.nn as nn
import torch
import torch.nn.functional as F

# Define custom activation functions not in PyTorch by default
def Swish(x):
    return x * torch.sigmoid(x)

def Mish(x):
    return x * torch.tanh(F.softplus(x))

# Add a mapping for all activation functions
activation_mapping = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "PReLU": nn.PReLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
    "Softmax": lambda: nn.Softmax(dim=-1),
    "Softplus": nn.Softplus,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "Swish": Swish,
    "Mish": Mish
}

class SimpleCNN(nn.Module):
    def __init__(self, 
                 num_classes, 
                 num_conv_layers=2, 
                 conv_padding=False, 
                 num_hidden_layers=1, 
                 hidden_layer_sizes=[128], 
                 activation_fn="ReLU"):
        """
        Args:
            num_classes (int): Number of output classes.
            num_conv_layers (int): Number of convolutional layers.
            conv_padding (bool): Apply padding to convolutional layers (default: False).
            num_hidden_layers (int): Number of hidden layers in the fully connected part.
            hidden_layer_sizes (list of int): Sizes of each hidden layer.
            activation_fn (str): Activation function name.
        """
        super(SimpleCNN, self).__init__()

        if activation_fn not in activation_mapping:
            raise ValueError(f"Unsupported activation function '{activation_fn}'. Supported: {list(activation_mapping.keys())}")

        self.activation_fn = activation_mapping[activation_fn]

        self.conv_layers = nn.ModuleList()
        input_channels = 1  # Assuming input is grayscale (1 channel)

        # Dynamically add convolutional layers
        for i in range(num_conv_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=16 * (2**i),  # Increase channels with each layer
                        kernel_size=3,
                        stride=1,
                        padding=1 if conv_padding else 0,  # Apply padding if specified
                    ),
                    self.activation_fn(),
                    nn.MaxPool2d(2, 2)
                )
            )
            input_channels = 16 * (2**i)  # Update input channels for the next layer

        # Calculate the output size after the convolutional layers
        conv_output_size = input_channels * (32 // (2**num_conv_layers))**2

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        prev_size = conv_output_size
        for i in range(num_hidden_layers):
            self.fc_layers.append(nn.Linear(prev_size, hidden_layer_sizes[i]))
            self.fc_layers.append(self.activation_fn())
            prev_size = hidden_layer_sizes[i]
        self.fc_layers.append(nn.Linear(prev_size, num_classes))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)  # Flatten
        for layer in self.fc_layers:
            if callable(layer):  # For custom activation functions like Swish or Mish
                x = layer(x)
            else:
                x = layer(x)
        return x