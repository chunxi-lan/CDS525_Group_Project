import argparse
import time
from datetime import datetime
from config import Config
from src.data_loader import get_dataloaders
from src.models import get_model, print_model_summary
from src.trainer import train_full
from src.utils import set_seed, save_metrics


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
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    set_seed(args.seed)
    
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
