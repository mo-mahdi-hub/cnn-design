"""
train.py - Training Engine
===========================
Trains CNN models on CIFAR-10 under identical conditions for fair comparison.
Logs per-epoch metrics and total training time.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch, return (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test/validation set, return (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, model_name,
                epochs=20, lr=0.001, device=None, save_dir='models'):
    """
    Full training loop for a model.

    Args:
        model:        The CNN model to train.
        train_loader: Training DataLoader.
        test_loader:  Test/validation DataLoader.
        model_name:   Name for saving weights and logging.
        epochs:       Number of training epochs.
        lr:           Learning rate for Adam optimizer.
        device:       torch.device (auto-detected if None).
        save_dir:     Directory to save model weights.

    Returns:
        dict with training history and metadata:
        {
            'model_name': str,
            'epochs': int,
            'train_loss': list, 'train_acc': list,
            'val_loss': list, 'val_acc': list,
            'training_time': float (seconds),
            'total_params': int,
            'best_val_acc': float,
        }
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training history
    history = {
        'model_name': model_name,
        'epochs': epochs,
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'training_time': 0.0,
        'total_params': sum(p.numel() for p in model.parameters()),
        'best_val_acc': 0.0,
    }

    # Count parameters
    total_params = history['total_params']
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    print(f"  Device           : {device}")
    print(f"  Parameters       : {total_params:,} (trainable: {trainable:,})")
    print(f"  Optimizer        : Adam (lr={lr})")
    print(f"  Loss function    : CrossEntropyLoss")
    print(f"  Epochs           : {epochs}")
    print(f"{'='*60}")

    os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, f'{model_name}_best.pt')
            torch.save(model.state_dict(), save_path)

        print(
            f"  Epoch {epoch:>2}/{epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
            f"Time: {epoch_time:.1f}s"
            f"{'  * best' if val_acc >= best_val_acc else ''}"
        )

    total_time = time.time() - start_time
    history['training_time'] = total_time
    history['best_val_acc'] = best_val_acc

    print(f"\n  Training complete in {total_time:.1f}s")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")

    return history


if __name__ == '__main__':
    # Quick sanity check with 2 epochs
    from dataset import get_data_loaders
    from model_baseline import BaselineCNN

    train_loader, test_loader = get_data_loaders(batch_size=64)
    model = BaselineCNN()

    history = train_model(
        model, train_loader, test_loader,
        model_name='baseline_test', epochs=2
    )
    print(f"\nSanity check passed! Final val acc: {history['val_acc'][-1]:.2f}%")
