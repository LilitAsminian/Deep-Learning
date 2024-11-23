import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, input_size=(3, 64, 64), num_classes=10, latent_size=15, hidden_size=256):
        super(CVAE, self).__init__()
        self.input_size = input_size  # (C, H, W)
        self.latent_size = latent_size  # Z
        self.num_classes = num_classes  # C (number of classes)
        flattened_image_size = input_size[0] * input_size[1] * input_size[2]  # C*H*W

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_image_size + num_classes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Latent space
        self.mu_layer = nn.Linear(hidden_size, latent_size)
        self.logvar_layer = nn.Linear(hidden_size, latent_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + num_classes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, flattened_image_size),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample latent vector z.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def call(self, x, c):
        """
        Forward pass through the Conditional VAE.
        
        Inputs:
        - x: Input data of shape (N, C, H, W)
        - c: One-hot encoded class vector of shape (N, num_classes)
        
        Returns:
        - x_hat: Reconstructed data of shape (N, C, H, W)
        - mu: Latent space mean of shape (N, latent_size)
        - logvar: Latent space log-variance of shape (N, latent_size)
        """
        # Flatten input and concatenate with class vector
        batch_size = x.size(0)
        x_flattened = x.view(batch_size, -1)
        encoder_input = torch.cat([x_flattened, c], dim=1)
        
        # Encode
        hidden = self.encoder(encoder_input)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Concatenate latent variable with class vector for decoder input
        decoder_input = torch.cat([z, c], dim=1)
        
        # Decode
        reconstruction = self.decoder(decoder_input)
        x_hat = reconstruction.view(batch_size, *self.input_size)
        
        return x_hat, mu, logvar

    def forward(self, x, labels):
        """
        A standard forward pass using the `call` method.
        """
        # One-hot encode labels
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()
        return self.call(x, one_hot_labels)