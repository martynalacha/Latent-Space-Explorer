import os
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from model import VAE
from preprocessing import get_celeba_dataloaders


def vae_loss(x, x_hat, mu, log_var, beta=1.0):
    """
    VAE Loss = Reconstruction Loss + beta * KL Divergence.
    Fixed scaling issue by using sum reduction over all dimensions, 
    then averaging over the batch size.
    """
    batch_size = x.size(0)
    
    # Sum over all pixels: (B, 3, 64, 64) -> Scalar
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    
    # Sum over all latent dimensions: (B, latent_dim) -> Scalar
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Average per batch to keep learning rate stable
    recon_loss /= batch_size
    kl_loss /= batch_size
    
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def get_kl_weight(epoch, args):
    if args.anneal_epochs <= 0:
        return args.beta
    progress = min(epoch / args.anneal_epochs, 1.0)
    return args.beta * progress


def train_one_epoch(model, loader, optimizer, device, kl_weight, writer, global_step):
    model.train()
    total_loss_sum = recon_loss_sum = kl_loss_sum = 0.0
    n_batches = 0

    # Dataloader must now return images AND target attributes
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.float().to(device) # Condition vector 'c'

        x_hat, mu, log_var = model(images, labels)
        loss, recon, kl = vae_loss(images, x_hat, mu, log_var, kl_weight)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_sum += loss.item()
        recon_loss_sum += recon.item()
        kl_loss_sum    += kl.item()
        n_batches      += 1
        global_step    += 1

        writer.add_scalar('batch/train_loss', loss.item(),  global_step)
        writer.add_scalar('batch/recon_loss', recon.item(), global_step)
        writer.add_scalar('batch/kl_loss',    kl.item(),    global_step)
        writer.add_scalar('batch/kl_weight',  kl_weight,    global_step)

        if (batch_idx + 1) % 200 == 0:
            print(f"  [batch {batch_idx+1}/{len(loader)}] "
                  f"loss={loss.item():.4f}  recon={recon.item():.4f}  "
                  f"kl={kl.item():.4f}  kl_weight={kl_weight:.3f}")

    return {
        'total': total_loss_sum / n_batches,
        'recon': recon_loss_sum / n_batches,
        'kl':    kl_loss_sum    / n_batches,
    }, global_step


@torch.no_grad()
def validate(model, loader, device, kl_weight):
    model.eval()
    total_loss_sum = recon_loss_sum = kl_loss_sum = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)
        
        x_hat, mu, log_var = model(images, labels)
        loss, recon, kl = vae_loss(images, x_hat, mu, log_var, kl_weight)
        
        total_loss_sum += loss.item()
        recon_loss_sum += recon.item()
        kl_loss_sum    += kl.item()
        n_batches      += 1

    return {
        'total': total_loss_sum / n_batches,
        'recon': recon_loss_sum / n_batches,
        'kl':    kl_loss_sum    / n_batches,
    }


def train(args):
    # Optimize GPU operations if input sizes are fixed (64x64)
    torch.backends.cudnn.benchmark = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading dataloaders...")
    train_loader, valid_loader, _ = get_celeba_dataloaders(
        batch_size=args.batch_size, 
        img_size=64,
        num_workers=args.num_workers 
    )
    print(f"  Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)}")

    model = VAE(latent_dim=args.latent_dim, cond_dim=args.cond_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model CVAE | latent_dim={args.latent_dim} | cond_dim={args.cond_dim} | params: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=args.runs_dir)

    # Dummy inputs for TensorBoard graph
    dummy_img = torch.zeros(1, 3, 64, 64).to(device)
    dummy_cond = torch.zeros(1, args.cond_dim).to(device)
    writer.add_graph(model, (dummy_img, dummy_cond))

    history = {'train': [], 'valid': []}
    best_valid_loss = float('inf')
    global_step = 0

    # Fetch fixed batch for visualization
    fixed_images, fixed_labels = next(iter(valid_loader))
    fixed_images = fixed_images[:8].to(device)
    fixed_labels = fixed_labels[:8].float().to(device)

    for epoch in range(1, args.epochs + 1):
        kl_weight = get_kl_weight(epoch, args)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}   kl_weight={kl_weight:.4f}   lr={optimizer.param_groups[0]['lr']:.2e}")
        print('='*60)

        train_metrics, global_step = train_one_epoch(
            model, train_loader, optimizer, device, kl_weight, writer, global_step
        )
        print(f"  TRAIN -> total={train_metrics['total']:.4f}  "
              f"recon={train_metrics['recon']:.4f}  kl={train_metrics['kl']:.4f}")

        valid_metrics = validate(model, valid_loader, device, kl_weight)
        print(f"  VALID -> total={valid_metrics['total']:.4f}  "
              f"recon={valid_metrics['recon']:.4f}  kl={valid_metrics['kl']:.4f}")

        scheduler.step(valid_metrics['total'])

        writer.add_scalars('epoch/total_loss', {'train': train_metrics['total'], 'valid': valid_metrics['total']}, epoch)
        writer.add_scalars('epoch/recon_loss', {'train': train_metrics['recon'], 'valid': valid_metrics['recon']}, epoch)
        writer.add_scalars('epoch/kl_loss',    {'train': train_metrics['kl'],    'valid': valid_metrics['kl']},    epoch)

        model.eval()
        with torch.no_grad():
            x_hat, _, _ = model(fixed_images, fixed_labels)
        grid = make_grid(torch.cat([fixed_images.cpu(), x_hat.cpu()], dim=0), nrow=8, normalize=True)
        writer.add_image('reconstructions/orig_vs_recon', grid, epoch)

        with torch.no_grad():
            # Generate random samples using fixed labels
            z_random = torch.randn(8, args.latent_dim).to(device)
            samples = model.decode(z_random, fixed_labels).cpu()
        writer.add_image('samples/from_prior', make_grid(samples, nrow=8, normalize=True), epoch)

        history['train'].append(train_metrics)
        history['valid'].append(valid_metrics)

        if valid_metrics['total'] < best_valid_loss:
            best_valid_loss = valid_metrics['total']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'valid_loss': best_valid_loss, 'args': vars(args)},
                       checkpoint_dir / "best_model.pt")
            print(f"  ★ New best model (valid_loss={best_valid_loss:.4f})")

    writer.close()
    with open(checkpoint_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Conditional VAE on CelebA")
    parser.add_argument('--epochs',         type=int,   default=20)
    parser.add_argument('--batch_size',     type=int,   default=64)
    parser.add_argument('--latent_dim',     type=int,   default=128)
    parser.add_argument('--cond_dim',       type=int,   default=40, help='Number of condition attributes')
    parser.add_argument('--cond_embed_dim', type=int,   default=128, help='Dimension of condition embedding')
    parser.add_argument('--lr',             type=float, default=3e-4)
    parser.add_argument('--beta',           type=float, default=0.1, help='KL divergence weight')
    parser.add_argument('--anneal_epochs',  type=int,   default=5)
    parser.add_argument('--num_workers',    type=int,   default=4, help='Dataloader workers for speed')
    parser.add_argument('--save_every',     type=int,   default=5)
    parser.add_argument('--checkpoint_dir', type=str,   default='checkpoints')
    parser.add_argument('--runs_dir',       type=str,   default='runs')
    parser.add_argument('--hist_every',     type=int,   default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)