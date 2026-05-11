import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Convolutional conditional encoder.
    Maps image 3x64x64 + condition vector to (mu, log_var).
    """
    def __init__(self, latent_dim: int = 128, cond_dim: int = 40, cond_embed_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)
        )

        self.conv = nn.Sequential(
            # 3 x 64 x 64  ->  32 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 x 32 x 32  ->  64 x 16 x 16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 x 16 x 16  ->  128 x 8 x 8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 x 8 x 8  ->  256 x 4 x 4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten_dim = 256 * 4 * 4

        # Inputs to FC layers are now flattened image features + condition attributes
        self.fc_mu      = nn.Linear(self.flatten_dim + cond_embed_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim + cond_embed_dim, latent_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        h = self.conv(x)                    # (B, 256, 4, 4)
        h = h.view(h.size(0), -1)           # (B, 4096)

        c_embed = self.cond_proj(c)          # (B, cond_embed_dim)
        
        # Concatenate features with condition vector along feature dimension
        hc = torch.cat([h, c_embed], dim=1)       # (B, 4096 + cond_dim)
        
        mu      = self.fc_mu(hc)            # (B, latent_dim)
        log_var = self.fc_log_var(hc)       # (B, latent_dim)
        return mu, log_var


class Decoder(nn.Module):
    """
    Convolutional conditional decoder.
    Maps latent vector z + condition vector to image 3x64x64.
    """
    def __init__(self, latent_dim: int = 128, cond_dim: int = 40, cond_embed_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2)
        )

        # Input is latent vector + condition attributes
        self.fc = nn.Linear(latent_dim + cond_embed_dim, 256 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor):
        # Concatenate latent vector with condition vector
        c_embed = self.cond_proj(c)         # (B, cond_embed_dim)
        zc = torch.cat([z, c_embed], dim=1)       # (B, latent_dim + cond_dim)
        
        h = self.fc(zc)                     # (B, 4096)
        h = h.view(h.size(0), 256, 4, 4)    # (B, 256, 4, 4)
        x_hat = self.deconv(h)              # (B, 3, 64, 64)
        return x_hat


class VAE(nn.Module):
    """
    Conditional Variational Autoencoder.
    """
    def __init__(self, latent_dim: int = 128, cond_dim: int = 40):
        super().__init__()
        self.encoder = Encoder(latent_dim, cond_dim)
        self.decoder = Decoder(latent_dim, cond_dim)

    def reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, log_var = self.encoder(x, c)
        z = self.reparametrize(mu, log_var)
        x_hat = self.decoder(z, c)
        return x_hat, mu, log_var

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        return self.encoder(x, c)

    def decode(self, z: torch.Tensor, c: torch.Tensor):
        return self.decoder(z, c)