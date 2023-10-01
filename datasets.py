from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

from cutout import Cutout

__all__ = ['cifar10_dataloaders', 'cifar100_dataloaders', 'svhn_dataloaders']


def cifar10_dataloaders(batch_size, cutout, data_dir='datasets/cifar10', use_val=True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    if cutout:
        length = 8
        print(f'Using cutout data augmentation of length: {length}')
        train_transform.transforms.append(Cutout(n_holes=1, length=length))

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if use_val:
        train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
        val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    else:
        train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
        val_loader = None
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


def svhn_dataloaders(batch_size, cutout, data_dir='datasets/svhn', use_val=True):
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    if cutout:
        length = 8
        print(f'Using cutout data augmentation of length: {length}')
        train_transform.transforms.append(Cutout(n_holes=1, length=length))

    if use_val:
        train_set = Subset(SVHN(data_dir, split='train', transform=train_transform, download=True), list(range(50000)))
        val_set = Subset(SVHN(data_dir, split='train', transform=test_transform, download=True), list(range(50000, 73257)))
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    else:
        train_set = SVHN(data_dir, split='train', transform=train_transform, download=True)
        val_loader = None
    test_set = SVHN(data_dir, split='test', transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


def cifar100_dataloaders(batch_size, cutout, data_dir='datasets/cifar100', use_val=True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    if cutout:
        length = 8
        print(f'Using cutout data augmentation of length: {length}')
        train_transform.transforms.append(Cutout(n_holes=1, length=length))

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if use_val:
        train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
        val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    else:
        train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
        val_loader = None
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
