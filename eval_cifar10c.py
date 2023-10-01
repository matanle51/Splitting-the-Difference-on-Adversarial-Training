import argparse
import json

import torch
from torchvision import transforms
from tqdm import tqdm

from cifar10c.cifar10_c import CIFAR10C
from models.wideresnet import WideResNet
from train_utils import eval_test
from normalize_utils import NormalizeByChannelMeanStd

parser = argparse.ArgumentParser(description='Evaluation code for CIFAR10-C')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='Batch size for testing (default: 128)')
parser.add_argument('--attacks', default=['brightness', 'defocus_blur', 'fog',
                                          'glass_blur', 'jpeg_compression', 'motion_blur', 'saturate', 'snow',
                                          'speckle_noise', 'contrast', 'elastic_transform', 'frost',
                                          'gaussian_noise', 'impulse_noise', 'pixelate', 'shot_noise',
                                          'spatter', 'zoom_blur'],
                    help='corruption attack type')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--data', type=str, default='cifar10c/cifar10-c-datasets', help='path to dir of data')
parser.add_argument('--model-dir', type=str, default='model_cifar10', help='path to model we wish to load')
parser.add_argument('--checkpoint', default='model.pt', type=str, help='path to pretrained model')
parser.add_argument('--num-classes', default=10, help='Number of original dataset classes')

args = parser.parse_args([])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Number of classes: {args.num_classes}')

def main():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    model = WideResNet(num_classes=10, use_normalize=True, normalize_layer=dataset_normalization, eval_mode=True).to(device)


    assert args.checkpoint != ''

    checkpoint = torch.load(f'{args.model_dir}/{args.checkpoint}', map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f'test_accuracy={checkpoint["test_accuracy"]}')
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    print('read checkpoint {}'.format(args.checkpoint))

    res_dict = {}
    for attack_type in tqdm(args.attacks):
        test_datasets = CIFAR10C(root=args.data, transform=transform_test, attack_type=attack_type)
        test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False)

        test_loss, test_accuracy = eval_test(model, device, test_loader)
        print("For attack type {}, Test Accuracy: {}".format(attack_type, test_accuracy))
        res_dict[attack_type] = test_accuracy

    with open(f'{args.model_dir}/cifar10c_results.json', 'w+') as fp:
        json.dump(res_dict, fp)


if __name__ == '__main__':
    main()
