import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from config import Config, CIFAR100_CLASSES


def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])

    return train_transform, test_transform


def get_dataloaders(batch_size=Config.BATCH_SIZE,
                     num_workers=Config.NUM_WORKERS,
                     pin_memory=Config.PIN_MEMORY,
                     data_dir=Config.DATA_DIR,
                     val_ratio=0.2):
    train_transform, test_transform = get_data_transforms()

    full_train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.RANDOM_SEED)
    )

    val_dataset.dataset.transform = test_transform

    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, CIFAR100_CLASSES


def get_test_loader(batch_size=Config.BATCH_SIZE,
                    num_workers=Config.NUM_WORKERS,
                    pin_memory=Config.PIN_MEMORY,
                    data_dir=Config.DATA_DIR):
    _, test_transform = get_data_transforms()

    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return test_loader, CIFAR100_CLASSES


if __name__ == '__main__':
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    print(f"Number of classes: {len(classes)}")
