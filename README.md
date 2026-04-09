# CIFAR-100 Image Classification Project

## Project Overview

This project implements an image classification system using ResNet-18/34/50 models on the CIFAR-100 dataset. It includes comprehensive training, evaluation, and visualization capabilities.

## Features

1. **Training Framework**
   - Support for multiple ResNet architectures (18, 34, 50)
   - Multiple loss functions (Cross Entropy, Focal Loss)
   - Multiple optimizers (Adam, SGD)
   - Learning rate scheduling (ReduceLROnPlateau, StepLR)
   - Early stopping mechanism

2. **Visualization**
   - Training loss, training accuracy, and test accuracy over epochs
   - Comparison of different learning rates (0.1, 0.01, 0.001, 0.0001)
   - Comparison of different batch sizes (8, 16, 32, 64, 128)
   - Comparison of different loss functions
   - Visualization of first 100 test results

3. **Experiment Management**
   - Automatic checkpoint saving
   - Metrics tracking and saving
   - Reproducible results with fixed random seeds

## Usage

### Training
```bash
python train.py --model resnet18 --batch_size 64 --lr 0.001 --epochs 50
```

### Custom Visualization
```bash
python train.py --custom_plots
```

### Test Sample Visualization
```bash
python train.py --visualize_test100
```

### Running Data Visualization
```bash
python train.py --plot_running_data
```

## Project Structure

```
CDS525_Group_Project/
├── config.py              # Configuration file
├── train.py               # Main training script
├── src/
│   ├── trainer.py         # Training logic
│   ├── models.py          # Model definitions
│   ├── data_loader.py     # Data loading and preprocessing
│   └── utils.py           # Utility functions
├── checkpoints/           # Model checkpoints
├── results/
│   ├── plots/             # Visualization plots
│   └── test_visualizations/  # Test sample visualizations
└── running data/          # Metrics files
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- numpy 1.24+
- matplotlib 3.7+
- scikit-learn 1.3+
- tqdm 4.65+

## Results

The project generates comprehensive visualization results including:

1. Single experiment curves for Cross Entropy and Focal Loss
2. Learning rate comparison plots (0.1 vs 0.01, 0.001 vs 0.0001)
3. Batch size comparison plots (8/16/32 vs 64/128)
4. Test sample visualizations (first 100 results)

