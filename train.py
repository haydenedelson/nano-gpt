import os
import hydra
import torch
import wandb

from tqdm import tqdm

import utils
from model import GPT

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_batch(data, batch_size, block_size, device):
    # Randomly sample offsets in the data to construct mini batches with
    idxs = torch.randint(len(data) - block_size, (batch_size,))
    # Construct batch tensors
    x = torch.stack([data[i: i+block_size] for i in idxs]).to(device)
    y = torch.stack([data[i+1: i+block_size+1] for i in idxs]).to(device)
    return x, y


def run_epoch(model, data, loss_fn, cfg, optimizer=None):
    x_batch, y_batch = get_batch(data, cfg.batch_size, cfg.block_size, DEVICE)

    if optimizer:
        optimizer.zero_grad()

    logits = model(x_batch)

    # Compute loss
    logits = logits.view(-1, logits.size(-1))
    y_batch = y_batch.view(-1)
    loss = loss_fn(logits, y_batch)

    if optimizer:
        loss.backward()
        optimizer.step()

    return loss.item()


@hydra.main(version_base=None, config_name='config', config_path='config')
def main(cfg):
    torch.manual_seed(cfg.seed)
    run = wandb.init(project=cfg.project_name, reinit=True, save_code=True, job_type='model-training')

    # Load data
    with open(os.path.join(hydra.utils.get_original_cwd(), cfg.data.file_path), 'r', encoding='utf-8') as in_file:
        text = in_file.read()
    chars = sorted(sorted(list(set(text))))
    vocab_size = len(chars)

    # Create encoding/decoding mappings
    c_to_i = {ch: i for i, ch in enumerate(chars)}
    i_to_c = {i: ch for i, ch in enumerate(chars)}

    # Encode dataset
    data = torch.tensor(utils.encode(text, c_to_i), dtype=torch.long)

    # Split into train/test
    n_val = int(len(data) * cfg.data.val_pct)
    train_data = data[:-n_val]
    val_data = data[-n_val:]

    # Get model
    model = GPT(vocab_size, **cfg.model.params)
    model = model.to(DEVICE)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"# of trainable parameters: {trainable_params}")
    wandb.watch(model, log='gradients', log_freq=50, log_graph=True)

    # Check device
    print("Device:", DEVICE)

    # Get optimizer
    optimizer = utils.get_optimizer(model.parameters(), cfg.optimizer)
    print("Optimizer:", optimizer)

    # Get scheduler
    scheduler = utils.get_scheduler(optimizer, cfg)
    print("Scheduler:", scheduler)

    loss_fn = utils.get_loss(cfg.loss.name)
    print("Loss:", loss_fn)

    try:
        for epoch in tqdm(range(cfg.num_epochs)):
            # Run validation step first per epoch
            torch.set_grad_enabled(False)
            model.eval()
            val_loss = run_epoch(model, val_data, loss_fn, cfg)
            wandb.log({f"val/{cfg.loss.name}": val_loss}, step=epoch)

            if (epoch == 0) or (epoch % cfg.logging.generate_interval) == 0 or (epoch == cfg.num_epochs - 1):
                context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
                generated_text = utils.decode(model.generate(context, max_new_tokens=cfg.logging.num_prediction_tokens)[0].tolist(), i_to_c)
                print(generated_text)

            # Run training step
            torch.set_grad_enabled(True)
            model.train()
            train_loss = run_epoch(model, train_data, loss_fn, cfg, optimizer=optimizer)
            scheduler.step()
            wandb.log({f"train/{cfg.loss.name}": train_loss}, step=epoch)

            wandb.log({'learning_rate': optimizer.param_groups[0]['lr'],
                       'momentum': optimizer.param_groups[0]['momentum'] if 'momentum' in optimizer.param_groups[0] else 0.0},
                       step=epoch)
            
            # Log gradient updates
            with torch.no_grad():
                utils.visualize_updates(run, model, scheduler, epoch)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")

    wandb.finish()


if __name__ == "__main__":
    main()

    

