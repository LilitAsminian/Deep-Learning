import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_size=(3, 64, 64), latent_size=64):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        c, h, w = input_size

        # Encoder with convolutional layers and batch normalization
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),  # (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),  # Flatten to (128 * 8 * 8)
        )

        # Fully connected layers for latent space
        encoded_size = 128 * (h // 8) * (w // 8)

        
        self.mu_layer = nn.Linear(encoded_size, latent_size)
        self.logvar_layer = nn.Sequential(
            nn.Linear(encoded_size, latent_size),
            nn.ReLU()  # Ensure logvar remains positive
        )


        # Decoder with deconvolutional layers and batch normalization
        self.decoder_input = nn.Linear(latent_size, encoded_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, h // 8, w // 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, c, kernel_size=4, stride=2, padding=1),  # (3, 64, 64)
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decode
        decoded = self.decoder(self.decoder_input(z))

        return decoded, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std