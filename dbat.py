import random

import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dbat_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=1/255,
              epsilon=8/255,
              perturb_steps=10,
              attack_type='targeted',
              epoch=None,
              num_classes=None,
              label_smoothing=0.5):
    model.eval()
    batch_size = len(x_natural)

    # generate adversarial example
    x_adv = generate_adv_example(attack_type, batch_size, epsilon, model, num_classes, perturb_steps, step_size, x_natural, y, epoch)

    # Switch model to train mode
    model.train()

    # get x_adv
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    # zero gradient
    optimizer.zero_grad()

    # calculate natural images loss
    comb_inputs = torch.cat((x_natural, x_adv), 0)
    comb_outputs = model(comb_inputs)

    y_adv = y.detach() + num_classes

    y = torch.cat((y, y_adv), 0).long().to(device)

    ce_loss = nn.CrossEntropyLoss(reduction='mean', label_smoothing=label_smoothing)
    loss = ce_loss(comb_outputs, y)

    return loss, [comb_inputs, y]


def generate_adv_example(attack_type, batch_size, epsilon, model, num_classes, perturb_steps, step_size,
                         x_natural, y, epoch):
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    if attack_type in ('targeted', 'targeted_ll'):
        # Select random target class for each sample
        if attack_type == 'targeted_ll':
            with torch.no_grad():
                outs = model(x_natural.detach())
                targeted_y = torch.min(outs, 1).indices.long()  # Take least likely label
                for idx, (lbl, t_lbl) in enumerate(zip(y, targeted_y)):
                    if t_lbl == lbl or t_lbl == lbl + num_classes:
                        cls_list = list(range(num_classes * 2))
                        cls_list.remove(lbl)
                        cls_list.remove(lbl + num_classes)
                        targeted_y[idx] = random.choice(cls_list)
        elif attack_type == 'targeted':
            targeted_y = []
            for lbl in y:
                if epoch < 20:
                    cls_list = list(range(num_classes))
                    cls_list.remove(lbl)
                else:
                    cls_list = list(range(num_classes * 2))
                    cls_list.remove(lbl)
                    cls_list.remove(lbl+num_classes)
                targeted_y.append(random.choice(cls_list))
            targeted_y = torch.tensor(targeted_y).long().to(device)
        elif attack_type == 'adv':
            targeted_y = y.clone().detach() + num_classes
        else:
            raise NotImplementedError(f'Target type={attack_type} not supported')

        # Run targeted-pgd
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(x_adv), targeted_y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() - step_size * torch.sign(
                grad.detach())  # minus since we want to got with the direction of the gradient
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif attack_type == 'untargeted':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(x_adv, proj=True), y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise NotImplemented(f'Attack not implemented: {attack_type}')
    return x_adv
