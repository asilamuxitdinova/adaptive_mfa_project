"""
Face verifiers: are these two photos of the same person?

Two verifiers are compared:
  - Baseline (no learning): the aligned face crop shrunk to 32x32 grey pixels, compared by
    cosine similarity. Shows how far plain pixels get.
  - Pretrained FaceNet: MTCNN finds and aligns the face, then an Inception-ResNet-v1 trained on
    VGGFace2 (Schroff et al., "FaceNet", CVPR 2015; Cao et al., "VGGFace2", FG 2018; PyTorch
    port: facenet-pytorch) maps it to a 512-number embedding. Not trained here: evaluation only.

If MTCNN finds no face (dark, blurred or side-on photos), the central part of the image is
used instead and the photo is counted as a detection failure. This matters for Stage 3, where
bad light makes detection fail.

Scores follow src.metrics: higher = more likely the same person.
"""
from dataclasses import dataclass

import numpy as np

IMAGE_SIZE = 160          # FaceNet input size
MARGIN = 14               # pixels around the detected face (facenet-pytorch's LFW example)
BASELINE_SIZE = 32


@dataclass
class FaceEmbeddings:
    embeddings: np.ndarray      # (n, 512) unit length
    pixels: np.ndarray          # (n, 1024) baseline features, unit length
    detected: np.ndarray        # (n,) bool, False = no face found, central crop used


def center_crop(img, size=IMAGE_SIZE, fraction=0.6):
    """Fallback when no face is detected: the central square, standardized like MTCNN output."""
    w, h = img.size
    side = int(min(w, h) * fraction)
    left, top = (w - side) // 2, (h - side) // 2
    crop = img.crop((left, top, left + side, top + side)).resize((size, size))
    arr = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1)
    return (arr - 127.5) / 128.0                       # facenet-pytorch's fixed_image_standardization


def pixel_features(faces: np.ndarray) -> np.ndarray:
    """(n, 3, H, W) standardized crops -> (n, 32*32) grey, mean-removed, unit-length vectors."""
    import torch
    grey = torch.from_numpy(faces).mean(dim=1, keepdim=True)
    small = torch.nn.functional.adaptive_avg_pool2d(grey, BASELINE_SIZE).flatten(1).numpy()
    small = small - small.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(small, axis=1, keepdims=True)
    return small / np.maximum(norm, 1e-8)


class FaceEmbedder:
    """MTCNN alignment + FaceNet embedding. `pretrained=None` gives random weights (tests only)."""

    def __init__(self, device="cpu", pretrained="vggface2"):
        from facenet_pytorch import MTCNN, InceptionResnetV1
        self.device = device
        self.mtcnn = MTCNN(image_size=IMAGE_SIZE, margin=MARGIN, post_process=True, device=device)
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(device)

    def align(self, images):
        """PIL images -> (n, 3, 160, 160) standardized face crops and a detected-face mask."""
        faces, detected = [None] * len(images), np.zeros(len(images), dtype=bool)
        by_size = {}
        for i, img in enumerate(images):
            by_size.setdefault(img.size, []).append(i)        # MTCNN batches need equal sizes
        for idx in by_size.values():
            found = self.mtcnn([images[i] for i in idx])
            for i, face in zip(idx, found):
                if face is not None:
                    faces[i], detected[i] = face.cpu().numpy(), True
                else:
                    faces[i] = center_crop(images[i])
        return np.stack(faces).astype(np.float32), detected

    def embed_files(self, paths, batch_size=64, progress=True) -> FaceEmbeddings:
        import torch
        from PIL import Image

        embs, pixels, detected = [], [], []
        for start in range(0, len(paths), batch_size):
            images = [Image.open(p).convert("RGB") for p in paths[start:start + batch_size]]
            faces, found = self.align(images)
            with torch.no_grad():
                embs.append(self.model(torch.from_numpy(faces).to(self.device)).cpu().numpy())
            pixels.append(pixel_features(faces))
            detected.append(found)
            if progress and (start // batch_size) % 20 == 0:
                print(f"  embedded {min(start + batch_size, len(paths)):,} / {len(paths):,} images", flush=True)
        return FaceEmbeddings(np.concatenate(embs), np.concatenate(pixels), np.concatenate(detected))


def cosine_scores(vectors: np.ndarray, rows_a, rows_b) -> np.ndarray:
    """Cosine similarity of unit-length vectors."""
    return (vectors[rows_a] * vectors[rows_b]).sum(axis=1)


def kfold_accuracy(scores: np.ndarray, labels: np.ndarray, folds: np.ndarray):
    """
    The standard LFW protocol: for each fold, pick the threshold with the best accuracy on the
    other 9 folds, then measure accuracy on this fold. Returns (mean, std) over folds.
    """
    accs = []
    for f in np.unique(folds):
        train, test = folds != f, folds == f
        accs.append(((scores[test] >= best_threshold(scores[train], labels[train])) == labels[test]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Threshold t (accept when score >= t) with the highest accuracy. It is placed halfway between
    the lowest accepted score and the score just below it, so it generalizes to unseen pairs.
    """
    order = np.argsort(scores, kind="stable")
    s, y = scores[order], labels[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])          # first index of each distinct score
    neg_below = np.r_[0, np.cumsum(y == 0)][starts]               # rejected impostors
    pos_above = (y == 1).sum() - np.r_[0, np.cumsum(y == 1)][starts]   # accepted genuine pairs
    k = int(np.argmax(neg_below + pos_above))
    return float(s[starts[k]] if k == 0 else (s[starts[k]] + s[starts[k - 1]]) / 2)
