from torchvision.models import resnet18
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 1000, pretrained: bool = False):
        super(ResNet, self).__init__()
        self.model = resnet18(pretrained=pretrained)

        # Adjust first conv layer for non-RGB inputs
        if input_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=False,
            )

        # Replace classification head
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)