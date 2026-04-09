import argparse
import os
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from config import Config
from src.data_loader import get_dataloaders, get_test_loader
from src.models import get_model, print_model_summary
from src.trainer import train_full
from src.utils import (
    set_seed,
    save_metrics,
    load_checkpoint,
    plot_metrics_from_json_dir,
    visualize_first_n_test_samples,
)


def load_metrics(filepath):
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_single_experiment(metrics, title, save_path):
    """Plot training loss, training accuracy, and test accuracy over epochs for a single experiment"""
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1.plot(epochs, metrics['train_loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss', fontsize=12)
    ax1.set_title('Train Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, metrics['train_acc'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Train Accuracy (%)', fontsize=12)
    ax2.set_title('Train Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(epochs, metrics['test_acc'], 'r-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax3.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_learning_rate_comparison(metrics_dict, lr_list, title, save_path):
    """Plot training loss, training accuracy, and test accuracy over epochs for different learning rates"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, lr in enumerate(lr_list):
        metrics = metrics_dict[lr]
        epochs = range(1, len(metrics['train_loss']) + 1)
        ax1.plot(epochs, metrics['train_loss'], color=colors[idx], label=f'lr={lr}', linewidth=2)
        ax2.plot(epochs, metrics['train_acc'], color=colors[idx], label=f'lr={lr}', linewidth=2)
        ax3.plot(epochs, metrics['test_acc'], color=colors[idx], label=f'lr={lr}', linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss', fontsize=12)
    ax1.set_title('Train Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Train Accuracy (%)', fontsize=12)
    ax2.set_title('Train Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax3.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_batch_size_comparison(metrics_dict, bs_list, title, save_path):
    """Plot training loss, training accuracy, and test accuracy over epochs for different batch sizes"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, bs in enumerate(bs_list):
        metrics = metrics_dict[bs]
        epochs = range(1, len(metrics['train_loss']) + 1)
        ax1.plot(epochs, metrics['train_loss'], color=colors[idx], label=f'bs={bs}', linewidth=2)
        ax2.plot(epochs, metrics['train_acc'], color=colors[idx], label=f'bs={bs}', linewidth=2)
        ax3.plot(epochs, metrics['test_acc'], color=colors[idx], label=f'bs={bs}', linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss', fontsize=12)
    ax1.set_title('Train Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Train Accuracy (%)', fontsize=12)
    ax2.set_title('Train Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax3.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CIFAR-100 Image Classification Training')
    
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                        help=f'batch size (default: {Config.BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                        help=f'learning rate (default: {Config.LEARNING_RATE})')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                        help=f'number of epochs (default: {Config.NUM_EPOCHS})')
    parser.add_argument('--model', type=str, default=Config.MODEL_NAME,
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help=f'model architecture (default: {Config.MODEL_NAME})')
    parser.add_argument('--loss', type=str, default=Config.LOSS_FUNCTION,
                        choices=['cross_entropy'],
                        help=f'loss function (default: {Config.LOSS_FUNCTION})')
    parser.add_argument('--optimizer', type=str, default=Config.OPTIMIZER,
                        choices=['adam', 'sgd'],
                        help=f'optimizer (default: {Config.OPTIMIZER})')
    parser.add_argument('--pretrained', action='store_true', default=Config.PRETRAINED,
                        help=f'use pretrained weights (default: {Config.PRETRAINED})')
    parser.add_argument('--freeze_backbone', action='store_true', default=Config.FREEZE_BACKBONE,
                        help=f'freeze backbone layers (default: {Config.FREEZE_BACKBONE})')
    parser.add_argument('--seed', type=int, default=Config.RANDOM_SEED,
                        help=f'random seed (default: {Config.RANDOM_SEED})')
    parser.add_argument('--checkpoint_name', type=str, default='best_model.pth',
                        help='checkpoint filename')
    parser.add_argument('--metrics_name', type=str, default=None,
                        help='metrics filename (default: metrics_YYYYMMDD_HHMMSS.json)')

    parser.add_argument('--plot_running_data', action='store_true', default=False,
                        help='plot metrics json files from running data directory (no training)')
    parser.add_argument('--running_data_dir', type=str,
                        default=os.path.join(Config.PROJECT_ROOT, 'running data'),
                        help='directory that contains metrics_*.json files')
    parser.add_argument('--plots_dir', type=str,
                        default=os.path.join(Config.RESULTS_DIR, 'plots'),
                        help='output directory for plots')
    parser.add_argument('--plot_max_epochs', type=int, default=100,
                        help='max epochs to visualize from each json (default: 100)')

    parser.add_argument('--visualize_test100', action='store_true', default=False,
                        help='visualize first 100 test samples (ground truth only if no checkpoint exists)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='optional checkpoint filepath; overrides --checkpoint_name')
    
    parser.add_argument('--custom_plots', action='store_true', default=False,
                        help='generate custom plots as requested')
    
    return parser.parse_args()


def main():
    """Main function for training and visualization"""
    args = parse_args()
    
    set_seed(args.seed)

    did_anything = False

    if args.plot_running_data:
        plot_metrics_from_json_dir(
            json_dir=[args.running_data_dir, Config.PROJECT_ROOT],
            output_dir=args.plots_dir,
            max_epochs=args.plot_max_epochs,
        )
        print(f"Plots saved to: {args.plots_dir}")
        did_anything = True

    if args.visualize_test100:
        print("\nLoading data (test loader only)...")
        test_loader, classes = get_test_loader(batch_size=128)

        output_dir = os.path.join(Config.RESULTS_DIR, "test_visualizations")
        os.makedirs(output_dir, exist_ok=True)

        gt_path = os.path.join(output_dir, "test_first100_ground_truth.png")
        visualize_first_n_test_samples(
            test_loader=test_loader,
            classes=classes,
            output_path=gt_path,
            n=100,
            device=Config.DEVICE,
            model=None,
        )
        print(f"Saved: {gt_path}")

        ckpt_path = args.checkpoint_path
        if ckpt_path is None:
            ckpt_path = os.path.join(Config.CHECKPOINT_DIR, args.checkpoint_name)

        model = None
        if os.path.exists(ckpt_path):
            print("\nLoading model + checkpoint for predictions...")
            model = get_model(
                model_name=args.model,
                num_classes=Config.NUM_CLASSES,
                pretrained=args.pretrained,
                freeze_backbone=args.freeze_backbone
            ).to(Config.DEVICE)
            load_checkpoint(model, filename=os.path.basename(ckpt_path), save_dir=os.path.dirname(ckpt_path))

            pred_path = os.path.join(output_dir, "test_first100_pred_vs_true.png")
            visualize_first_n_test_samples(
                test_loader=test_loader,
                classes=classes,
                output_path=pred_path,
                n=100,
                device=Config.DEVICE,
                model=model,
            )
            print(f"Saved: {pred_path}")
        else:
            print(f"\nCheckpoint not found, skip prediction visualization: {ckpt_path}")

        did_anything = True

    if args.custom_plots:
        print("\n" + "=" * 60)
        print("Generating custom plots")
        print("=" * 60)
        
        metrics_dir = args.running_data_dir
        plots_dir = args.plots_dir
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. 生成Cross Entropy的训练图
        ce_path = os.path.join(metrics_dir, 'metrics_adam.json')
        if os.path.exists(ce_path):
            ce_metrics = load_metrics(ce_path)
            plot_single_experiment(
                ce_metrics,
                'Cross Entropy Loss - Training Metrics',
                os.path.join(plots_dir, 'single_cross_entropy.png')
            )
        
        # 2. 生成Focal Loss的训练图
        focal_path = os.path.join(metrics_dir, 'metrics_focal.json')
        if os.path.exists(focal_path):
            focal_metrics = load_metrics(focal_path)
            plot_single_experiment(
                focal_metrics,
                'Focal Loss - Training Metrics',
                os.path.join(plots_dir, 'single_focal_loss.png')
            )
        
        # 3. 生成学习率对比图
        lr_files = {
            0.1: os.path.join(metrics_dir, 'metrics_lr0.1.json'),
            0.01: os.path.join(metrics_dir, 'metrics_lr0.01.json'),
            0.001: os.path.join(metrics_dir, 'metrics_lr0.001.json'),
            0.0001: os.path.join(metrics_dir, 'metrics_lr0.0001.json'),
        }
        
        lr_metrics = {}
        for lr, path in lr_files.items():
            if os.path.exists(path):
                lr_metrics[lr] = load_metrics(path)
        
        if len(lr_metrics) >= 2:
            lr_list1 = [lr for lr in [0.1, 0.01] if lr in lr_metrics]
            if len(lr_list1) == 2:
                plot_learning_rate_comparison(
                    {lr: lr_metrics[lr] for lr in lr_list1},
                    lr_list1,
                    'Learning Rate Comparison (0.1, 0.01)',
                    os.path.join(plots_dir, 'lr_comparison_0.1_0.01.png')
                )
            
            lr_list2 = [lr for lr in [0.001, 0.0001] if lr in lr_metrics]
            if len(lr_list2) == 2:
                plot_learning_rate_comparison(
                    {lr: lr_metrics[lr] for lr in lr_list2},
                    lr_list2,
                    'Learning Rate Comparison (0.001, 0.0001)',
                    os.path.join(plots_dir, 'lr_comparison_0.001_0.0001.png')
                )
        
        # 4. 生成Batch Size对比图
        bs_files = {
            8: os.path.join(metrics_dir, 'metrics_bs8.json'),
            16: os.path.join(metrics_dir, 'metrics_bs16.json'),
            32: os.path.join(metrics_dir, 'metrics_bs32.json'),
            64: os.path.join(metrics_dir, 'metrics_bs64.json'),
            128: os.path.join(metrics_dir, 'metrics_bs128.json'),
        }
        
        bs_metrics = {}
        for bs, path in bs_files.items():
            if os.path.exists(path):
                bs_metrics[bs] = load_metrics(path)
        
        if len(bs_metrics) >= 2:
            bs_list1 = [bs for bs in [8, 16, 32] if bs in bs_metrics]
            if len(bs_list1) >= 2:
                plot_batch_size_comparison(
                    {bs: bs_metrics[bs] for bs in bs_list1},
                    bs_list1,
                    'Batch Size Comparison (8, 16, 32)',
                    os.path.join(plots_dir, 'bs_comparison_8_16_32.png')
                )
            
            bs_list2 = [bs for bs in [64, 128] if bs in bs_metrics]
            if len(bs_list2) >= 2:
                plot_batch_size_comparison(
                    {bs: bs_metrics[bs] for bs in bs_list2},
                    bs_list2,
                    'Batch Size Comparison (64, 128)',
                    os.path.join(plots_dir, 'bs_comparison_64_128.png')
                )
        
        print("\n" + "=" * 60)
        print("All custom plots generated!")
        print("=" * 60)
        did_anything = True

    if did_anything:
        return
    
    print("=" * 60)
    print("CIFAR-100 Image Classification - Training Script")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print("=" * 60)
    
    print("\nLoading data...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        batch_size=args.batch_size
    )
    print(f"Data loaded successfully.")
    print(f"Classes: {', '.join(classes[:10])}... (total {len(classes)} classes)")
    
    print("\nCreating model...")
    model = get_model(
        model_name=args.model,
        num_classes=Config.NUM_CLASSES,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(Config.DEVICE)
    print_model_summary(model)
    
    print("\nStarting training...")
    start_time = time.time()
    
    metrics = train_full(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        device=Config.DEVICE,
        loss_name=args.loss,
        optimizer_name=args.optimizer,
        lr=args.lr,
        checkpoint_filename=args.checkpoint_name
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    if args.metrics_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_filename = f"metrics_{timestamp}.json"
    else:
        metrics_filename = args.metrics_name
    
    metrics_filepath = save_metrics(metrics, metrics_filename)
    print(f"\nMetrics saved to: {metrics_filepath}")
    
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Final Train Acc: {metrics['train_acc'][-1]:.2f}%")
    print(f"Final Val Acc: {metrics['val_acc'][-1]:.2f}%")
    print(f"Final Test Acc: {metrics['test_acc'][-1]:.2f}%")
    print(f"Best Val Acc: {max(metrics['val_acc']):.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
