"""
dataset.py - CIFAR-10 Data Loading, Preprocessing & Visualization
=================================================================
Loads the CIFAR-10 dataset with normalization and optional data augmentation.
Provides helper functions to visualize sample images with their labels.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# CIFAR-10 channel-wise mean and std (precomputed from training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_transforms(augment=False):
    """
    Build preprocessing transforms.

    Args:
        augment: If True, apply data augmentation (random crop + horizontal flip).

    Returns:
        A torchvision.transforms.Compose pipeline.
    """
    transform_list = []

    if augment:
        # Data augmentation: random crop with padding and horizontal flip
        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

    # Core preprocessing: convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return transforms.Compose(transform_list)


def get_data_loaders(batch_size=64, augment_train=False, num_workers=2):
    """
    Load CIFAR-10 training and test sets.

    Args:
        batch_size:     Batch size for both loaders.
        augment_train:  If True, apply augmentation to training data.
        num_workers:    Number of parallel data loading workers.

    Returns:
        (train_loader, test_loader) tuple of DataLoader objects.
    """
    train_transform = get_transforms(augment=augment_train)
    test_transform  = get_transforms(augment=False)

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"[Dataset] CIFAR-10 loaded successfully")
    print(f"  Training samples : {len(train_dataset):,}")
    print(f"  Test samples     : {len(test_dataset):,}")
    print(f"  Classes          : {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES)})")
    print(f"  Image size       : 32x32x3 (RGB)")
    print(f"  Batch size       : {batch_size}")
    print(f"  Augmentation     : {'Yes' if augment_train else 'No'}")

    return train_loader, test_loader


def show_sample_images(data_loader, num_images=16, save_path=None):
    """
    Display a grid of sample images from the dataset.

    Args:
        data_loader: A DataLoader to pull a batch from.
        num_images:  Number of images to display (will form a square grid).
        save_path:   If provided, save the figure to this path.
    """
    # Get one batch
    images, labels = next(iter(data_loader))

    # Denormalize for display
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    images_display = images[:num_images] * std + mean
    images_display = images_display.clamp(0, 1)

    # Plot grid
    cols = 4
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.5 * rows))
    fig.suptitle('CIFAR-10 Sample Images', fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            # Convert from CHW to HWP for matplotlib
            img = images_display[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(CLASS_NAMES[labels[i].item()], fontsize=11)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Dataset] Sample images saved to {save_path}")

    plt.close(fig)


if __name__ == '__main__':
    # Quick test: load data and show samples
    import os
    os.makedirs('results', exist_ok=True)

    train_loader, test_loader = get_data_loaders(batch_size=64)
    show_sample_images(train_loader, num_images=16, save_path='results/sample_images.png')
    print("\nDone! Check results/sample_images.png")
