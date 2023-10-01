import argparse
import os

import torch
import torch.optim as optim

from sched_lr import adjust_learning_rate
from models.wideresnet import WideResNet
from models.preactresnet import PreActResNet18
from swa_utils import moving_average, bn_update
from train_utils import train_robust_model, eval_train, eval_test, save_best_robust_model, get_dataloaders
from whitebox_attack import eval_adv_test_whitebox

parser = argparse.ArgumentParser(description='DBAT experiment')

#----------------------- Train parameters -----------------------#
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=7e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--schedule', default='cifar_swa', help='learning rate schedule')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset [cifar10, cifar100, svhn]')
parser.add_argument('--num-classes', default=-1, help='Number of original dataset classes')
parser.add_argument('--cutout', type=bool, default=True)
parser.add_argument('--use-normalize', type=bool, default=True)
#-----------------------------------------------------------------#

#----------------------- Attack parameters -----------------------#
parser.add_argument('--epsilon', default=8./255, help='perturbation')
parser.add_argument('--num-steps', default=10, help='perturb number of steps')
parser.add_argument('--test-num-steps', default=20, help='perturb number of steps for robust test')
parser.add_argument('--step-size', default=2./255, help='perturb step size')
parser.add_argument('--test-step-size', default=1./255, help='perturb step size for test')
parser.add_argument('--test-attack-freq', default=20, type=int, metavar='N', help='save frequency')
parser.add_argument('--attack_type', type=str, default='targeted', help='Which attack to use', choices=['targeted', 'targeted_ll', 'untargeted'])
#-----------------------------------------------------------------#

#------------------------ General Settings -----------------------#
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='model', help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', default=20, type=int, metavar='N', help='save frequency')
#-----------------------------------------------------------------#

#-------------------------- SWA setting --------------------------#
parser.add_argument('--swa', default=True, help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=55, metavar='N', help='SWA start epoch number (default: 55)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N', help='SWA model collection frequency/cycle length in epochs (default: 1)')
#-----------------------------------------------------------------#

args = parser.parse_args([])


torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, _, dataset_normalization, num_classes = get_dataloaders(args)

args.num_classes = num_classes
print(f'Number of classes: {args.num_classes}')

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)


def main():
    model = WideResNet(num_classes=args.num_classes, use_normalize=args.use_normalize, normalize_layer=dataset_normalization).to(device)

    if args.swa:
        print('---- Using SWA ----')
        swa_model = WideResNet(num_classes=args.num_classes, use_normalize=args.use_normalize, normalize_layer=dataset_normalization).to(device)
        swa_n = 0

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_robust_acc, swa_best_robust_acc = 0, 0

    start_epoch = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch, schedule=args.schedule, total_epochs=args.epochs)

        # adversarial training
        gen_data = train_robust_model(model, device, train_loader, optimizer, epoch,
                           args.log_interval, args.step_size, args.epsilon, args.num_steps, args.attack_type, args.epochs, args.num_classes)

        # evaluation on natural examples
        print('================================================================')
        train_loss, train_accuracy = eval_train(model, device, train_loader)
        test_loss, test_accuracy = eval_test(model, device, val_loader)
        print('================================================================')

        # SWA update
        if args.swa:
            swa_model, swa_n, swa_best_robust_acc = swa_update(swa_best_robust_acc, epoch, model, optimizer, swa_model, swa_n, gen_data)

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.model_dir, 'model-nn-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(args.model_dir, 'opt-nn-checkpoint_epoch{}.tar'.format(epoch)))

        if epoch % args.test_attack_freq == 0 or epoch >= args.swa_start:
            natural_acc, robust_acc = eval_adv_test_whitebox(model, device, val_loader, len(val_loader.dataset),
                                                             args.epsilon, step_size=args.test_step_size,
                                                             num_attack_steps=args.test_num_steps, loss_type='ce',
                                                             num_classes=args.num_classes)
            if robust_acc > best_robust_acc:
                best_robust_acc = robust_acc
                save_best_robust_model(epoch, model, natural_acc, optimizer, robust_acc,
                                       test_accuracy, test_loss, train_accuracy, train_loss,
                                       model_dir=f'{args.model_dir}/best_robust_checkpoint_cifar10.pt')


def swa_update(best_robust_acc, epoch, model, optimizer, swa_model, swa_n, gen_data):
    if args.swa and epoch >= args.swa_start and (epoch - args.swa_start) % args.swa_c_epochs == 0:
        # SWA
        moving_average(swa_model, model, 1.0 / (swa_n + 1))
        swa_n += 1
        bn_update(gen_data, swa_model, model, args.attack_type, args.epsilon, args.num_classes, args.num_steps,
                  args.step_size, epoch)

        print('============================ SWA RESULTS ============================')
        # evaluation on natural examples
        train_loss, train_accuracy = eval_train(swa_model, device, train_loader)
        test_loss, test_accuracy = eval_test(swa_model, device, val_loader)

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(swa_model.state_dict(),
                       os.path.join(args.model_dir, 'swa-model-nn-epoch{}.pt'.format(epoch)))

        # evaluation on adv examples
        if epoch % args.test_attack_freq == 0:
            natural_acc, robust_acc = eval_adv_test_whitebox(swa_model, device, val_loader,
                                                             len(val_loader.dataset),
                                                             args.epsilon, step_size=args.test_step_size,
                                                             num_attack_steps=args.test_num_steps, loss_type='ce',
                                                             num_classes=args.num_classes)
            if robust_acc > best_robust_acc:
                best_robust_acc = robust_acc
                save_best_robust_model(epoch, swa_model, natural_acc, optimizer, robust_acc,
                                       test_accuracy, test_loss, train_accuracy, train_loss,
                                       model_dir=f'{args.model_dir}/swa_best_robust_checkpoint.pt')
        print('========================== SWA RESULTS END ==========================')
    return swa_model, swa_n, best_robust_acc


if __name__ == '__main__':
    main()
