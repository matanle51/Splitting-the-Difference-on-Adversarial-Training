import time

import numpy as np
import torch
import torch.nn.functional as F

from datasets import *
from dbat import dbat_loss
from normalize_utils import NormalizeByChannelMeanStd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_robust_model(model, device, train_loader, optimizer, epoch,
                       log_interval, step_size, epsilon, num_steps,
                       attack_type, n_epochs, num_classes):
    agg_loss = 0.0
    model.train()
    all_time = []
    gen_data = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss

        s_time = time.time()
        loss, gen_data_batch = dbat_loss(model=model,
                                         x_natural=data,
                                         y=target,
                                         optimizer=optimizer,
                                         step_size=step_size,
                                         epsilon=epsilon,
                                         perturb_steps=num_steps,
                                         attack_type=attack_type,
                                         epoch=epoch,
                                         num_classes=num_classes)
        gen_data.append(gen_data_batch)
        e_time = time.time()
        all_time.append(e_time - s_time)
        loss.backward()
        optimizer.step()

        agg_loss += loss.item()

        # print progress
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), agg_loss / (batch_idx + 1)))
    print(f'Timing stats: avg={np.mean(all_time)}; std={np.std(all_time)}')
    return gen_data


def eval_train(model, device, train_loader, proj=True):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, proj)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader, proj=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, proj)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def save_best_robust_model(epoch, model, natural_acc, optimizer, robust_acc, test_accuracy, test_loss, train_accuracy,
                           train_loss, model_dir):
    torch.save({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_robust_accuracy': robust_acc
    }, model_dir)
    print(f'Best model was found with: robust_acc={robust_acc}; natural_acc={natural_acc}')


def get_dataloaders(args):
    # prepare dataset
    if args.dataset == 'cifar10':
        print('Loading cifar10 data loaders')
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_loader, val_loader, test_loader = cifar10_dataloaders(batch_size=args.batch_size, cutout=args.cutout)
        num_classes = 10
    elif args.dataset == 'cifar100':
        print('Loading cifar100 data loaders')
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        train_loader, val_loader, test_loader = cifar100_dataloaders(batch_size=args.batch_size, cutout=args.cutout)
        num_classes = 100
    elif args.dataset == 'svhn':
        print('Loading svhn data loaders')
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4376821, 0.4437697, 0.47280442], std=[0.19803012, 0.20101562, 0.19703614])
        train_loader, val_loader, test_loader = svhn_dataloaders(batch_size=args.batch_size, cutout=args.cutout)
        num_classes = 10
    else:
        raise ValueError("Unknown Dataset")

    return train_loader, val_loader, test_loader, dataset_normalization, num_classes
