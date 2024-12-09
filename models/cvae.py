import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, input_size, num_classes, latent_size):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.latent_size = latent_size

        # Label embedding for conditioning
        self.label_embedding = nn.Embedding(num_classes, 64)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_size[0] + 1, 64, kernel_size=4, stride=2, padding=1),  # +1 for label
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Fully connected layers for latent variables
        self.fc_mu = nn.Linear(512 * (input_size[1] // 16) * (input_size[2] // 16), latent_size)
        self.fc_logvar = nn.Linear(512 * (input_size[1] // 16) * (input_size[2] // 16), latent_size)

        # Decoder
        self.fc_decoder = nn.Linear(latent_size + 64, 512 * (input_size[1] // 16) * (input_size[2] // 16))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, input_size[0], kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar.clamp(-20, 20))  # Clamp logvar
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x, labels):
        # Embed labels and concatenate with input
        embedded_labels = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)  # (B, 64, 1, 1)
        embedded_labels = embedded_labels.expand(-1, -1, x.size(2), x.size(3))  # Match spatial size
        x = torch.cat([x, embedded_labels], dim=1)

        # Encode
        enc_out = self.encoder(x)
        enc_out = enc_out.view(enc_out.size(0), -1)  # Flatten
        mu = self.fc_mu(enc_out)
        logvar = self.fc_logvar(enc_out)
        z = self.reparameterize(mu, logvar)

        # Concatenate latent vector with label embedding
        z = torch.cat([z, self.label_embedding(labels)], dim=1)

        # Decode
        dec_input = self.fc_decoder(z)
        dec_input = dec_input.view(dec_input.size(0), 512, self.input_size[1] // 16, self.input_size[2] // 16)
        recon_x = self.decoder(dec_input)

        return recon_x, mu, logvar


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)




