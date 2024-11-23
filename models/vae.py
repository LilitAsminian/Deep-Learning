import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_size=(3, 64, 64), latent_size=32):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        c, h, w = input_size
        flattened_size = c * h * w

        # Encoder with dropout
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Latent space
        self.mu_layer = nn.Linear(256, latent_size)
        self.logvar_layer = nn.Linear(256, latent_size)

        # Decoder with dropout
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, flattened_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Encode
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Decode
        decoded = self.decoder(z)
        reconstructed = decoded.view(batch_size, *self.input_size)

        return reconstructed, mu, logvar



    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample latent vector z.
        """
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Random normal distribution
        return mu + eps * std  # Reparametrize

    def call(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through 
        encoder, reparameterization, and decoder.
    
        Inputs:
        - x: Batch of input images of shape (N, 3, H, W)
        
        Returns:
        - x_hat: Reconstructed input data of shape (N, 3, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z)
        - logvar: Matrix representing estimated variance in log-space (N, Z)
        """
        # Flatten input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten to (N, C*H*W)
        
        # Encode
        hidden = self.encoder(x)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decoder(z)
        x_hat = reconstruction.view(batch_size, *self.input_size)  # Reshape to original image size
        
        return x_hat, mu, logvar

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
