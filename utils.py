import torch
import torch.nn.functional as F

def encode(input_string, c_to_i):
    return [c_to_i[c] for c in input_string]


def decode(input_encoding, i_to_c):
    return ''.join([i_to_c[i] for i in input_encoding])


def get_loss(loss_name):
    if loss_name == 'cross_entropy':
        return F.cross_entropy
    else:
        raise AssertionError(f"{loss_name} loss not supported")


def get_optimizer(parameters, optimizer_cfg):
    optimizer_name_lwr = optimizer_cfg.name.lower()
    if optimizer_name_lwr == 'adam':
        return torch.optim.Adam(parameters, **optimizer_cfg.params)
    elif optimizer_name_lwr == 'adamw':
        return torch.optim.AdamW(parameters, **optimizer_cfg.params)
    elif optimizer_name_lwr == 'sgd':
        return torch.optim.SGD(parameters, **optimizer_cfg.params)
    else:
        raise AssertionError(f"{optimizer_cfg.name} optimizer not supported")


def get_scheduler(optimizer, cfg):
    scheduler_name_lwr = cfg.scheduler.name.lower()
    if scheduler_name_lwr == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                   max_lr=cfg.optimizer.params.lr,
                                                   total_steps=cfg.num_epochs,
                                                   epochs=cfg.num_epochs,
                                                   **cfg.scheduler.params)
    else:
        raise AssertionError(f"{cfg.scheduler.name} scheduler not supported")


