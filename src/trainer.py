import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import Config
from src.utils import save_checkpoint


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def test(model, test_loader, criterion, device):
    """Test model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def get_loss_function(loss_name=Config.LOSS_FUNCTION):
    """Get loss function"""
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


def get_optimizer(model, optimizer_name=Config.OPTIMIZER,
                  lr=Config.LEARNING_RATE,
                  weight_decay=Config.WEIGHT_DECAY,
                  momentum=Config.MOMENTUM):
    """Get optimizer"""
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_type='plateau',
                  patience=Config.SCHEDULER_PATIENCE,
                  factor=Config.SCHEDULER_FACTOR):
    """Get learning rate scheduler"""
    if scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=patience, factor=factor, verbose=True
        )
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        return None


def train_full(model, train_loader, val_loader, test_loader,
               num_epochs=Config.NUM_EPOCHS,
               device=Config.DEVICE,
               loss_name=Config.LOSS_FUNCTION,
               optimizer_name=Config.OPTIMIZER,
               lr=Config.LEARNING_RATE,
               early_stopping_patience=Config.EARLY_STOPPING_PATIENCE,
               checkpoint_filename='best_model.pth'):
    """Full training loop"""
    
    criterion = get_loss_function(loss_name)
    optimizer = get_optimizer(model, optimizer_name, lr)
    scheduler = get_scheduler(optimizer)

    best_val_acc = 0.0
    epochs_no_improve = 0

    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)

        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['test_loss'].append(test_loss)
        metrics['test_acc'].append(test_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_acc, checkpoint_filename)
            print(f"Best model saved with Val Acc: {best_val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.2f}%")
    return metrics
