import argparse
import os
import time
from datetime import datetime
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


def parse_args():
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
    
    return parser.parse_args()


def main():
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
