import os
import torch


class Config:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 64
    NUM_WORKERS = 4
    PIN_MEMORY = True

    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9

    EARLY_STOPPING_PATIENCE = 10
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5

    MODEL_NAME = 'resnet18'
    NUM_CLASSES = 100
    PRETRAINED = True
    FREEZE_BACKBONE = False

    LOSS_FUNCTION = 'cross_entropy'
    OPTIMIZER = 'adam'

    RANDOM_SEED = 42


CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]
