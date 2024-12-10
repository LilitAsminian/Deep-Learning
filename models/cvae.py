import torch
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, input_size=(3, 64, 64), num_classes=5, latent_size=32, label_embedding_size=64):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.latent_size = latent_size
        self.label_embedding_size = label_embedding_size

        c, h, w = input_size
        total_channels = c + label_embedding_size

        self.label_embedding = nn.Embedding(num_classes, label_embedding_size)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(total_channels, 32, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, total_channels, h, w) 
            encoded_output = self.encoder(dummy_input)
            self.flatten_size = encoded_output.view(1, -1).size(1)

        self.fc_mu = nn.Linear(self.flatten_size, latent_size)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_size)

        # Decoder
        self.fc_decoder = nn.Linear(latent_size + label_embedding_size, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, h // 8, w // 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, c, kernel_size=4, stride=2, padding=1), 
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        labels_embed = self.label_embedding(labels).unsqueeze(-1).unsqueeze(-1)
        labels_embed = labels_embed.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, labels_embed], dim=1)

        # Encode
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # Decode
        z = torch.cat([z, self.label_embedding(labels)], dim=1)
        x = self.fc_decoder(z)
        x = self.decoder(x)
        return x, mu, logvar
