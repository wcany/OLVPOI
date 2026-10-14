import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from tqdm import tqdm

from model import ContrastiveAEClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def contrastive_loss(latent, adjacency, temperature, eps=1e-8):
    latent = F.normalize(latent, dim=1)
    similarity = torch.matmul(latent, latent.T) / temperature
    similarity.fill_diagonal_(-1e9)
    exponential_similarity = torch.exp(similarity)
    positive_sum = (exponential_similarity * adjacency).sum(dim=1)
    denominator = exponential_similarity.sum(dim=1)
    per_sample_loss = -torch.log((positive_sum + eps) / (denominator + eps))
    valid = adjacency.sum(dim=1) > 0
    if not torch.any(valid):
        raise RuntimeError("No valid positive pairs remain after KNN pruning.")
    return per_sample_loss[valid].mean()


def compute_binary_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "ROC_AUC": float(roc_auc_score(labels, probabilities)),
        "Accuracy": float(accuracy_score(labels, predictions)),
        "F1": float(
            f1_score(labels, predictions, average="binary", pos_label=1, zero_division=0)
        ),
        "Sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "Specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
    }


def train(
    config,
    training_features,
    training_labels,
    training_adjacency,
    device,
    seed,
    progress_description,
):
    """Reset the model seed, initialize once, and train the model."""
    set_seed(seed)
    features = torch.tensor(training_features, dtype=torch.float32, device=device)
    labels = torch.tensor(training_labels, dtype=torch.float32, device=device)
    adjacency = torch.tensor(
        training_adjacency.toarray(), dtype=torch.float32, device=device
    )
    model = ContrastiveAEClassifier(
        input_dim=training_features.shape[1],
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
    ).to(device)
    if config.optimizer != "Adam":
        raise ValueError("The current configuration requires Adam.")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.minimum_learning_rate,
    )
    bce = nn.BCEWithLogitsLoss()
    iterator = range(config.epochs)
    if config.show_progress:
        iterator = tqdm(iterator, desc=progress_description, leave=False)

    model.train()
    for _ in iterator:
        latent, reconstructed, logits = model(features)
        reconstruction_loss = F.mse_loss(reconstructed, features)
        contrastive_loss_value = contrastive_loss(
            latent, adjacency, config.contrastive_temperature
        )
        classification_loss = bce(logits, labels)
        total_loss = (
            classification_loss
            + config.alpha * reconstruction_loss
            + config.beta * contrastive_loss_value
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())
    return model


def predict(model, features, device):
    tensor = torch.tensor(features, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        _, _, logits = model(tensor)
    probabilities = torch.sigmoid(logits).cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain NaN or Inf.")
    return probabilities
