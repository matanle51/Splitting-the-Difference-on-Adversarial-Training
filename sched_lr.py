def adjust_learning_rate(optimizer, epoch, schedule, total_epochs):
    """decrease the learning rate"""
    decay_factor = 1
    if schedule == 'cifar':
        if epoch == int(0.75 * total_epochs):
            decay_factor = 0.1
        if epoch == int(0.9 * total_epochs):
            decay_factor = 0.1
        if epoch == total_epochs:
            decay_factor = 0.1
    elif schedule == 'cifar_long':
        if epoch == 100:
            decay_factor = 0.1
        if epoch == 150:
            decay_factor = 0.1
        if epoch == total_epochs:
            decay_factor = 0.1
    elif schedule == 'cifar_swa':
        if epoch == 50:
            decay_factor = 0.1
        if epoch == 150:
            decay_factor = 0.1
    elif schedule == 'svhn':
        if epoch == 50:
            decay_factor = 0.1
        if epoch == 75:
            decay_factor = 0.1
    else:
        raise ValueError('Unkown LR schedule %s' % schedule)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_factor
        print(f'Update learning rate to: {param_group["lr"]}')
