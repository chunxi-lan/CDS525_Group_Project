import os
import json
import random
import numpy as np
import torch
from config import Config


def set_seed(seed=Config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_metrics(metrics, filename, save_dir=Config.RESULTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    return filepath


def load_metrics(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc,
                    filename='best_model.pth', save_dir=Config.CHECKPOINT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'val_acc': val_acc
    }
    torch.save(checkpoint, filepath)
    return filepath


def load_checkpoint(model, optimizer=None, scheduler=None,
                    filename='best_model.pth', save_dir=Config.CHECKPOINT_DIR,
                    device=Config.DEVICE):
    filepath = os.path.join(save_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['val_acc']
