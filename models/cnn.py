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
    "LeakyReLU": lambda: nn.LeakyReLU(negative_slope=0.01, inplace=False),
    "PReLU": lambda: nn.PReLU(),
    "FixedLeakyReLU": lambda: nn.LeakyReLU(negative_slope=0.01, inplace=False),
    "LearnableLeakyReLU": lambda: nn.PReLU(),
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
                 input_channels=1,
                 output_size=1, 
                 num_conv_layers=2, 
                 conv_padding=False, 
                 hidden_layers=[128], 
                 conv_activation_fn="ReLU", 
                 fc_activation_fns=None,
                 conv_filters=None):
        """
        Args:
            input_channels (int): Number of input channels (default: 1).
            output_size (int): Number of regression outputs (default: 1).
            num_conv_layers (int): Number of convolutional layers.
            conv_padding (bool): Apply padding to convolutional layers (default: False).
            hidden_layers (list of int): Sizes of each hidden layer.
            conv_activation_fn (str): Activation function for convolutional layers.
            fc_activation_fns (list of str): List of activation functions for fully connected layers.
            conv_filters (list of int): List of filter sizes for each convolutional layer (optional).
        """
        super(SimpleCNN, self).__init__()

        num_hidden_layers = len(hidden_layers)
        
        if fc_activation_fns is None:
            fc_activation_fns = ["ReLU"] * num_hidden_layers
        if len(fc_activation_fns) != num_hidden_layers:
            raise ValueError(
                "The number of activation functions in fc_activation_fns and the number of sizes in hidden_layers "
                "must both match the number of hidden layers."
            )
        if conv_activation_fn not in activation_mapping:
            raise ValueError(f"Unsupported activation function '{conv_activation_fn}' for conv layers. Supported: {list(activation_mapping.keys())}")
        for fn in fc_activation_fns:
            if fn not in activation_mapping:
                raise ValueError(f"Unsupported activation function '{fn}' for fc layers. Supported: {list(activation_mapping.keys())}")

        self.conv_activation_fn = activation_mapping[conv_activation_fn]
        self.fc_activation_fns = [activation_mapping[fn] for fn in fc_activation_fns]

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        if conv_filters is None:
            conv_filters = [8 * (2**i) for i in range(num_conv_layers)]
        elif len(conv_filters) != num_conv_layers:
            raise ValueError("Length of conv_filters must match num_conv_layers")

        for i in range(num_conv_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=conv_filters[i],
                        kernel_size=3,
                        stride=1,
                        padding=1 if conv_padding else 0
                    ),
                    self.conv_activation_fn(),
                    nn.MaxPool2d(2, 2)
                )
            )
            input_channels = conv_filters[i]

        # Fully connected layers
        fc_layers = []
        prev_size = None  # This will be determined dynamically in forward
        self.hidden_layers = hidden_layers
        self.fc_activation_fns = fc_activation_fns
        self.fc_output_size = output_size

        self.fc_layers = nn.ModuleList(fc_layers)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically
        prev_size = x.size(1)  # Dynamically calculate size after flattening

        # Define fully connected layers dynamically
        fc_layers = nn.ModuleList()
        for size, activation in zip(self.hidden_layers, self.fc_activation_fns):
            fc_layers.append(nn.Linear(prev_size, size))
            fc_layers.append(activation_mapping[activation]())
            prev_size = size
        fc_layers.append(nn.Linear(prev_size, self.fc_output_size))

        # Forward pass through FC layers
        for layer in fc_layers:
            x = layer(x)
        return x

