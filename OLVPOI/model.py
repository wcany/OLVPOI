import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAEClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0.0 <= dropout < 1.0.")
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, features):
        latent = F.normalize(self.encoder(features), dim=1)
        reconstructed = self.decoder(latent)
        logits = self.classifier(latent).squeeze(1)
        return latent, reconstructed, logits
