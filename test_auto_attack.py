import argparse

import torch
import torch.nn as nn
from autoattack import AutoAttack
from torchvision import transforms

from models.preactresnet import PreActResNet18
from models.wideresnet import WideResNet
from train_utils import get_dataloaders

parser = argparse.ArgumentParser(description='Test Auto Attack')
parser.add_argument('--epsilon', default=8/255, help='perturbation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='batch size(default: 128)')
parser.add_argument('--norm', type=str, default='Linf', metavar='N', help='Norm to use in Auto-Attack')
parser.add_argument('--version', type=str, default='standard', metavar='N', help='Auto Attack version to run')
parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'SVHN'])
parser.add_argument('--use-normalize', type=bool, default=True)
parser.add_argument('--model-type', type=str, default='WideResNet', choices=['WideResNet', 'PreActResNet18'])
parser.add_argument('--model-dir', type=str, default='model_cifar10', choices=['model_cifar10', 'model_cifar100', 'model_svhn'])
parser.add_argument('--checkpoint-name', type=str, default='checkpoint.pt')

args = parser.parse_args([])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}
transform_test = transforms.Compose([transforms.ToTensor()])

_, _, test_loader, dataset_normalization, num_classes = get_dataloaders(args)


def test_auto_attack():
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, version=args.version, device=device)

    model.eval()

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0).to(device)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0).to(device)

    print(f'x_test={len(x_test)}; y_test={len(y_test)}')
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=250)
    pgd_out = model(x_adv, 0)
    err_robust = (pgd_out.data.max(1)[1] != y_test.data).float().sum()
    print(f'err_robust={err_robust}')
    torch.save({'x_adv': x_adv}, f'auto_attack_res.pt')


if __name__ == '__main__':
    # Create model instance
    if args.model_type == 'PreActResNet18':
        model = nn.DataParallel(PreActResNet18(num_classes=num_classes, use_normalize=args.use_normalize, normalize_layer=dataset_normalization, eval_mode=True)).to(device)
    elif args.model_type == 'WideResNet':
        model = nn.DataParallel(WideResNet(num_classes=num_classes, use_normalize=args.use_normalize, normalize_layer=dataset_normalization, eval_mode=True)).to(device)
    else:
        raise NotImplementedError(f'Model type not supported: {args.model_type}')
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_name)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_auto_attack()
