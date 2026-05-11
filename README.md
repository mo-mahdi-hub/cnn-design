# CNN Architecture Comparison for Image Classification

**ENGR 422 – Computer Vision | Lab Activity 5**  
**Student:** Mohammed Mahdi  

## Overview

This project designs, trains, and compares two different CNN architectures on the **CIFAR-10** dataset to understand how layer combinations affect accuracy, training time, and computational cost.

## Dataset

**CIFAR-10** — 60,000 32×32 color images across 10 classes:  
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Architectures

| Feature | Architecture 1 (Baseline) | Architecture 2 (Advanced) |
|---|---|---|
| Conv blocks | 2 | 3 (6 conv layers) |
| Filters | 16 → 32 | 32 → 64 → 128 |
| Batch Normalization | ✗ | ✓ |
| Dropout | ✗ | ✓ (0.25 / 0.5) |
| Pooling | MaxPool + Flatten | MaxPool + Global Avg Pool |

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (train + evaluate + compare)
python main.py

# Skip training and just evaluate (if models are already saved)
python main.py --skip-train
```

## Project Structure

```
├── dataset.py           # Data loading, preprocessing, visualization
├── model_baseline.py    # Architecture 1 — simple baseline CNN
├── model_advanced.py    # Architecture 2 — advanced CNN with BN/Dropout
├── train.py             # Training engine
├── evaluate.py          # Evaluation, comparison table, prediction examples
├── main.py              # Main entry point
├── requirements.txt     # Dependencies
└── results/             # Generated plots and comparison outputs
```

## Results

*Results will be added after training is complete.*
