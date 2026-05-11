"""
model_baseline.py - Architecture 1: Simple Baseline CNN
========================================================
A shallow CNN with 2 convolution blocks and no regularization.
Serves as the performance floor for comparison.

Architecture:
    Conv2d(3→16, 3×3) → ReLU → MaxPool2d(2×2)
    Conv2d(16→32, 3×3) → ReLU → MaxPool2d(2×2)
    Flatten → Linear(32×6×6 → 128) → ReLU → Linear(128 → 10)
"""

import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Simple 2-block CNN for CIFAR-10 classification.

    Design choices:
    - Small filter counts (16, 32) to keep parameters low.
    - No batch normalization or dropout — intentionally simple.
    - MaxPool after each conv block to reduce spatial dimensions.
    - Standard Flatten → Dense classifier head.
    """

    def __init__(self, num_classes=10):
        super(BaselineCNN, self).__init__()

        # --- Block 1: Conv → ReLU → MaxPool ---
        # Input: 3×32×32 → Output: 16×15×15
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Block 2: Conv → ReLU → MaxPool ---
        # Input: 16×15×15 → Output: 32×6×6
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Classifier Head ---
        # Input: 32×6×6 = 1152 → Output: 10
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    import torch
    from torchsummary import summary

    model = BaselineCNN()
    print("=" * 60)
    print("ARCHITECTURE 1: Baseline CNN")
    print("=" * 60)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    summary(model, input_size=(3, 32, 32))

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable:,}")
