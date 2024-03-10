import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
import os

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


def visualize_updates(run, model, scheduler, epoch):
    g_fig, g_ax = plt.subplots(figsize=(20, 5))
    d_fig, d_ax = plt.subplots(figsize=(20, 5))

    legends = []
    for i, p in enumerate(model.parameters()):
        if p.ndim == 2 and p.requires_grad:
            # Log data-update to data values
            update_to_data = (scheduler.get_last_lr()[0] * p.grad.std() / p.data.std()).log10().item()
            run.log({f"update-to-data/layer_{i}": update_to_data}, step=epoch)

            # Log gradient distributions
            grad_y, grad_x = torch.histogram(p.grad.to('cpu'), density=True)
            g_ax.plot(grad_x[:-1], grad_y)

            # Log data distributions
            wgt_y, wgt_x = torch.histogram(p.data.to('cpu'), density=True)
            d_ax.plot(wgt_x[:-1], wgt_y)
            legends.append(f"layer_{i}")
    g_ax.legend(legends)
    d_ax.legend(legends)

    run.log({"gradient_values": wandb.Image(g_fig)}, step=epoch)
    run.log({"data_values": wandb.Image(d_fig)}, step=epoch)

    plt.close(fig=g_fig)
    plt.close(fig=d_fig)


def log_artifact(run, artifact, file_paths=None):
    if file_paths:
        if isinstance(file_paths, list):
            for fp in file_paths:
                if os.path.exists(fp):
                    artifact.add_file(fp)
        elif isinstance(file_paths, str):
            artifact.add_file(file_paths)
        else:
            raise AssertionError(f"file_paths argument {file_paths} not recognized. Must be instance of: string or list")

    run.log_artifact(artifact)
