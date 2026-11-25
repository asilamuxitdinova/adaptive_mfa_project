"""
Keystroke-dynamics verifiers: is this typed sentence from the same person as that one?

Two verifiers are compared:
  - Baseline (no training): summary statistics of each sentence's timings, compared by distance.
  - TypeNet-style LSTM (trained): a sequence model mapping a sentence to a 128-number
    embedding, trained so the same person's sentences are close together.
    Based on Acien et al., "TypeNet: Deep Learning Keystroke Biometrics" (IEEE T-BIOM, 2021),
    which introduced this approach on the same Aalto dataset. TypeNet uses triplet loss; the
    default here is supervised contrastive loss (Khosla et al., NeurIPS 2020), because batch-hard
    triplet loss can collapse towards identical embeddings (on the first real-data run its loss
    stuck at the margin and all scores were squeezed into a tiny range).

Scores follow src.metrics: higher = more likely the same person.
"""
import numpy as np

from src.keystroke_data import KeystrokeSet


# ---------------------------------------------------------------- baseline (no training)

def baseline_features(data: KeystrokeSet) -> np.ndarray:
    """Per sentence: mean, std and median of each timing feature over its real keys, plus length."""
    n, max_len, _ = data.x.shape
    mask = np.arange(max_len)[None, :] < data.lengths[:, None]
    timings = np.where(mask[..., None], data.x[..., 1:], np.nan)      # drop keycode, hide padding
    with np.errstate(all="ignore"):
        feats = np.concatenate([np.nanmean(timings, 1), np.nanstd(timings, 1),
                                np.nanmedian(timings, 1), (data.lengths / max_len)[:, None]], axis=1)
    return np.nan_to_num(feats).astype(np.float32)


def baseline_scores(features: np.ndarray, rows_a, rows_b) -> np.ndarray:
    """Negative scaled Manhattan distance between sentence statistics (closer = higher score)."""
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return -np.abs((features[rows_a] - features[rows_b]) / scale).sum(axis=1)


# ---------------------------------------------------------------- TypeNet-style model

def build_model(n_features: int, hidden: int = 128, embedding: int = 128, layers: int = 2, dropout: float = 0.5):
    import torch
    from torch import nn

    class TypeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Linear(hidden, embedding)
            self.config = dict(n_features=n_features, hidden=hidden, embedding=embedding, layers=layers, dropout=dropout)

        def forward(self, x, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h, _) = self.lstm(packed)
            return nn.functional.normalize(self.head(h[-1]), dim=1)   # unit length: cosine similarity

    return TypeNet()


def batch_hard_triplet_loss(embeddings, labels, margin: float = 0.2):
    """For each sentence: its farthest same-person sentence vs its closest other-person sentence."""
    import torch

    dist = torch.cdist(embeddings, embeddings)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    hardest_pos = (dist * (same & ~eye)).max(dim=1).values
    hardest_neg = dist.masked_fill(same, float("inf")).min(dim=1).values
    return torch.relu(hardest_pos - hardest_neg + margin).mean()


def supervised_contrastive_loss(embeddings, labels, temperature: float = 0.1):
    """
    Each sentence should be more similar to its own participant's other sentences than to
    every other sentence in the batch (softmax over all similarities). Using all pairs, not
    just the hardest one, makes training stable and resistant to collapse.
    """
    import torch

    n = len(labels)
    eye = torch.eye(n, dtype=torch.bool, device=labels.device)
    sim = (embeddings @ embeddings.T / temperature).masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positives = (labels[:, None] == labels[None, :]) & ~eye
    per_anchor = log_prob.masked_fill(~positives, 0.0).sum(dim=1) / positives.sum(dim=1).clamp(min=1)
    return -per_anchor[positives.any(dim=1)].mean()


def embedding_spread(embeddings: np.ndarray) -> float:
    """Average per-dimension standard deviation; close to 0 means all embeddings are the same (collapse)."""
    return float(embeddings.std(axis=0).mean()) if len(embeddings) else 0.0


def embed(model, data: KeystrokeSet, device="cpu", batch_size: int = 2048) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(data.x), batch_size):
            x = torch.from_numpy(data.x[i:i + batch_size]).to(device)
            n = torch.from_numpy(data.lengths[i:i + batch_size])
            out.append(model(x, n).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.config["embedding"]), dtype=np.float32)


def embedding_scores(embeddings: np.ndarray, rows_a, rows_b) -> np.ndarray:
    """Cosine similarity of unit-length embeddings."""
    return (embeddings[rows_a] * embeddings[rows_b]).sum(axis=1)
