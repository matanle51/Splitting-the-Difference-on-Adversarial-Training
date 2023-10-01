import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def moving_average_no_fc(net1, net2, alpha=1):
    for np1, np2 in zip(net1.named_parameters(), net2.named_parameters()):
        name1, param1 = np1
        name2, param2 = np2
        if 'fc' in name1 and 'fc' in name2:
            print(f'For fully connected layer: {name1} we are taking the last model parameters; no average')
            param1.data += param2.data
        else:
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(gen_data, swa_model, model, attack_type, epsilon, num_classes, perturb_steps, step_size, epoch):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param swa_model: model being update
        :return: None
    """
    if not check_bn(swa_model):
        return
    swa_model.train()
    momenta = {}
    swa_model.apply(reset_bn)
    swa_model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, y in gen_data:
        input, y = input.to(device), y.to(device)
        b = input.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        # model.eval()
        # x_adv = generate_adv_example(attack_type, input.data.size(0), epsilon, model, num_classes, perturb_steps,
        #                              step_size, input, y, epoch)

        # # get x_adv
        # x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

        # comb_inputs = torch.cat((input, x_adv), 0)
        swa_model(input)
        n += b

    swa_model.apply(lambda module: _set_momenta(module, momenta))
