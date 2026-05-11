"""
main.py - CNN Architecture Comparison Pipeline
================================================
ENGR 422 - Computer Vision | Lab Activity 5

Runs the full pipeline:
1. Load & visualize CIFAR-10 dataset
2. Train Architecture 1 (Baseline CNN)
3. Train Architecture 2 (Advanced CNN)
4. Evaluate, compare, and generate all results

Usage:
    python main.py                    # Full pipeline (train + evaluate)
    python main.py --epochs 30        # Custom number of epochs
    python main.py --skip-train       # Skip training, just evaluate saved models
    python main.py --batch-size 128   # Custom batch size
"""

import os
import argparse
import torch

from dataset import get_data_loaders, show_sample_images
from model_baseline import BaselineCNN
from model_advanced import AdvancedCNN
from train import train_model
from evaluate import generate_full_report, print_comparison_table


def parse_args():
    parser = argparse.ArgumentParser(
        description='CNN Architecture Comparison for CIFAR-10'
    )
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training and evaluate saved models')
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("=" * 60)
    print("  CNN Architecture Comparison for Image Classification")
    print("  ENGR 422 - Computer Vision | Lab Activity 5")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print("=" * 60)

    # ===== STEP 1: Load Dataset =====
    print("\n[Step 1] Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size, augment_train=False
    )
    show_sample_images(train_loader, num_images=16,
                       save_path='results/sample_images.png')

    # ===== STEP 2 & 3: Build Models =====
    model1 = BaselineCNN().to(device)
    model2 = AdvancedCNN().to(device)

    if not args.skip_train:
        # ===== STEP 2: Train Architecture 1 =====
        print("\n[Step 2] Training Architecture 1: Baseline CNN...")
        history1 = train_model(
            model1, train_loader, test_loader,
            model_name='Baseline_CNN',
            epochs=args.epochs, lr=args.lr, device=device
        )

        # ===== STEP 3: Train Architecture 2 =====
        print("\n[Step 3] Training Architecture 2: Advanced CNN...")
        history2 = train_model(
            model2, train_loader, test_loader,
            model_name='Advanced_CNN',
            epochs=args.epochs, lr=args.lr, device=device
        )

        # Save histories for later use
        torch.save({'history1': history1, 'history2': history2},
                    'models/training_histories.pt')

    else:
        # Load saved models and histories
        print("\n[Skip] Loading saved models and histories...")
        histories = torch.load('models/training_histories.pt',
                               weights_only=False)
        history1 = histories['history1']
        history2 = histories['history2']

        model1.load_state_dict(
            torch.load('models/Baseline_CNN_best.pt', weights_only=True)
        )
        model2.load_state_dict(
            torch.load('models/Advanced_CNN_best.pt', weights_only=True)
        )

    # ===== STEP 4: Evaluate & Compare =====
    print("\n[Step 4] Generating evaluation report...")

    # Load best weights for evaluation
    model1.load_state_dict(
        torch.load('models/Baseline_CNN_best.pt', weights_only=True)
    )
    model2.load_state_dict(
        torch.load('models/Advanced_CNN_best.pt', weights_only=True)
    )

    generate_full_report(
        history1, history2, model1, model2,
        test_loader, device, save_dir='results'
    )

    # ===== CONCLUSION =====
    print("\n" + "=" * 60)
    print("  CONCLUSION")
    print("=" * 60)

    better = "Architecture 2 (Advanced CNN)" if history2['best_val_acc'] > history1['best_val_acc'] else "Architecture 1 (Baseline CNN)"
    acc_diff = abs(history2['best_val_acc'] - history1['best_val_acc'])
    param_diff = history2['total_params'] - history1['total_params']
    time_diff = history2['training_time'] - history1['training_time']

    print(f"\n  {better} achieved higher accuracy.")
    print(f"  Accuracy difference: {acc_diff:.2f}%")
    print(f"  Parameter difference: {param_diff:+,} params")
    print(f"  Training time difference: {time_diff:+.1f}s")

    print(f"\n  The Advanced CNN uses Batch Normalization, Dropout, and")
    print(f"  Global Average Pooling which help prevent overfitting and")
    print(f"  improve generalization. The deeper architecture (3 blocks")
    print(f"  vs 2) allows learning more complex features.")

    if history2['best_val_acc'] > history1['best_val_acc']:
        print(f"\n  The more complex model performed better, demonstrating")
        print(f"  that regularization techniques effectively combat")
        print(f"  overfitting while the additional depth captures")
        print(f"  richer feature representations.")
    else:
        print(f"\n  Interestingly, the simpler model performed comparably,")
        print(f"  suggesting that for this task complexity, the baseline")
        print(f"  architecture may be sufficient.")

    print(f"\n  All results saved to results/ directory.")
    print("=" * 60)


if __name__ == '__main__':
    main()
