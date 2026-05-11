"""
evaluate.py - Evaluation, Comparison & Visualization
=====================================================
Generates comparison tables, training curves, and prediction examples
for both CNN architectures.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import CLASS_NAMES, CIFAR10_MEAN, CIFAR10_STD


def print_comparison_table(history1, history2):
    """
    Print a formatted comparison table of both architectures.
    Returns the table as a string for saving.
    """
    h1, h2 = history1, history2

    table = f"""
{'='*80}
                     CNN ARCHITECTURE COMPARISON TABLE
{'='*80}
{'Metric':<35} {'Architecture 1':<20} {'Architecture 2':<20}
{'-'*80}
{'Model name':<35} {h1['model_name']:<20} {h2['model_name']:<20}
{'Total parameters':<35} {h1['total_params']:>14,}      {h2['total_params']:>14,}
{'Training epochs':<35} {h1['epochs']:>14}      {h2['epochs']:>14}
{'Final train accuracy (%)':<35} {h1['train_acc'][-1]:>13.2f}%      {h2['train_acc'][-1]:>13.2f}%
{'Final val/test accuracy (%)':<35} {h1['val_acc'][-1]:>13.2f}%      {h2['val_acc'][-1]:>13.2f}%
{'Best val/test accuracy (%)':<35} {h1['best_val_acc']:>13.2f}%      {h2['best_val_acc']:>13.2f}%
{'Final train loss':<35} {h1['train_loss'][-1]:>14.4f}      {h2['train_loss'][-1]:>14.4f}
{'Final val/test loss':<35} {h1['val_loss'][-1]:>14.4f}      {h2['val_loss'][-1]:>14.4f}
{'Training time (seconds)':<35} {h1['training_time']:>13.1f}s      {h2['training_time']:>13.1f}s
{'='*80}
"""
    print(table)
    return table


def plot_training_curves(history1, history2, save_dir='results'):
    """
    Plot side-by-side training curves (loss and accuracy) for both models.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs1 = range(1, history1['epochs'] + 1)
    epochs2 = range(1, history2['epochs'] + 1)

    # --- Loss Plot ---
    ax = axes[0]
    ax.plot(epochs1, history1['train_loss'], 'b-', label=f"{history1['model_name']} (train)", linewidth=2)
    ax.plot(epochs1, history1['val_loss'], 'b--', label=f"{history1['model_name']} (val)", linewidth=2)
    ax.plot(epochs2, history2['train_loss'], 'r-', label=f"{history2['model_name']} (train)", linewidth=2)
    ax.plot(epochs2, history2['val_loss'], 'r--', label=f"{history2['model_name']} (val)", linewidth=2)
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Accuracy Plot ---
    ax = axes[1]
    ax.plot(epochs1, history1['train_acc'], 'b-', label=f"{history1['model_name']} (train)", linewidth=2)
    ax.plot(epochs1, history1['val_acc'], 'b--', label=f"{history1['model_name']} (val)", linewidth=2)
    ax.plot(epochs2, history2['train_acc'], 'r-', label=f"{history2['model_name']} (train)", linewidth=2)
    ax.plot(epochs2, history2['val_acc'], 'r--', label=f"{history2['model_name']} (val)", linewidth=2)
    ax.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Evaluate] Training curves saved to {save_path}")


def show_predictions(model, test_loader, model_name, device,
                     num_examples=5, save_dir='results'):
    """
    Show prediction examples: correct and incorrect.
    Displays num_examples correct and up to num_examples incorrect predictions.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Denormalization constants
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    correct_examples = []
    wrong_examples = []

    with torch.no_grad():
        for images, labels in test_loader:
            images_dev, labels_dev = images.to(device), labels.to(device)
            outputs = model(images_dev)
            _, predicted = outputs.max(1)

            for i in range(images.size(0)):
                pred = predicted[i].item()
                true = labels[i].item()
                img = images[i].cpu()

                if pred == true and len(correct_examples) < num_examples:
                    correct_examples.append((img, true, pred))
                elif pred != true and len(wrong_examples) < num_examples:
                    wrong_examples.append((img, true, pred))

                if (len(correct_examples) >= num_examples and
                    len(wrong_examples) >= num_examples):
                    break
            if (len(correct_examples) >= num_examples and
                len(wrong_examples) >= num_examples):
                break

    # Combine examples
    all_examples = correct_examples + wrong_examples
    num_total = len(all_examples)

    fig, axes = plt.subplots(2, num_examples, figsize=(3 * num_examples, 7))
    fig.suptitle(f'Predictions: {model_name}', fontsize=14, fontweight='bold')

    for row, (examples, row_title) in enumerate([
        (correct_examples, 'Correct Predictions'),
        (wrong_examples, 'Wrong Predictions')
    ]):
        for col in range(num_examples):
            ax = axes[row, col] if num_examples > 1 else axes[row]
            if col < len(examples):
                img, true_label, pred_label = examples[col]
                # Denormalize
                img_display = img * std + mean
                img_display = img_display.clamp(0, 1).permute(1, 2, 0).numpy()

                ax.imshow(img_display)
                color = 'green' if true_label == pred_label else 'red'
                ax.set_title(
                    f'True: {CLASS_NAMES[true_label]}\nPred: {CLASS_NAMES[pred_label]}',
                    fontsize=9, color=color, fontweight='bold'
                )
            ax.axis('off')

        # Row label
        if num_examples > 1:
            axes[row, 0].set_ylabel(row_title, fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'predictions_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Evaluate] Prediction examples saved to {save_path}")


def generate_full_report(history1, history2, model1, model2,
                         test_loader, device, save_dir='results'):
    """
    Generate all evaluation outputs:
    1. Comparison table
    2. Training curves plot
    3. Prediction examples for both models
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Comparison table
    table = print_comparison_table(history1, history2)
    table_path = os.path.join(save_dir, 'comparison_table.txt')
    with open(table_path, 'w') as f:
        f.write(table)
    print(f"[Evaluate] Comparison table saved to {table_path}")

    # 2. Training curves
    plot_training_curves(history1, history2, save_dir)

    # 3. Prediction examples
    show_predictions(model1, test_loader, history1['model_name'], device,
                     num_examples=5, save_dir=save_dir)
    show_predictions(model2, test_loader, history2['model_name'], device,
                     num_examples=5, save_dir=save_dir)

    print(f"\n[Evaluate] Full report generated in {save_dir}/")
