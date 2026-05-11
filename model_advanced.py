"""
model_advanced.py - Architecture 2: Advanced CNN with Regularization
====================================================================
A deeper CNN with 3 conv blocks, Batch Normalization, Dropout,
and Global Average Pooling. Designed to outperform the baseline.

Architecture:
    Block 1: Conv(3→32,3×3,pad=1) → BN → ReLU → Conv(32→32,3×3,pad=1) → BN → ReLU → MaxPool → Dropout(0.25)
    Block 2: Conv(32→64,3×3,pad=1) → BN → ReLU → Conv(64→64,3×3,pad=1) → BN → ReLU → MaxPool → Dropout(0.25)
    Block 3: Conv(64→128,3×3,pad=1) → BN → ReLU → GlobalAvgPool
    Classifier: Linear(128→256) → ReLU → Dropout(0.5) → Linear(256→10)
"""

import torch.nn as nn


class AdvancedCNN(nn.Module):
    """
    3-block CNN with modern regularization techniques for CIFAR-10.

    Key differences from Baseline:
    - 3 conv blocks instead of 2 (deeper feature extraction)
    - Double conv layers per block (richer feature learning)
    - Batch Normalization after every conv (stable training, acts as regularizer)
    - Dropout after conv blocks (0.25) and before output (0.5)
    - Global Average Pooling instead of Flatten (fewer parameters, spatial invariance)
    - Larger filter counts: 32 → 64 → 128
    """

    def __init__(self, num_classes=10):
        super(AdvancedCNN, self).__init__()

        # --- Block 1: Double Conv + BN + MaxPool + Dropout ---
        # Input: 3×32×32 → Output: 32×16×16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )

        # --- Block 2: Double Conv + BN + MaxPool + Dropout ---
        # Input: 32×16×16 → Output: 64×8×8
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )

        # --- Block 3: Conv + BN + Global Average Pooling ---
        # Input: 64×8×8 → Output: 128×1×1
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
        )

        # --- Classifier Head ---
        # Input: 128 → Output: 10
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    import torch
    from torchsummary import summary

    model = AdvancedCNN()
    print("=" * 60)
    print("ARCHITECTURE 2: Advanced CNN")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    summary(model, input_size=(3, 32, 32))

    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable:,}")
