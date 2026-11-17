"""
Biometric verification metrics shared by every factor (face, voice, keystroke, ...).

A verifier compares two samples and returns a similarity score: higher means
"more likely the same person". Each comparison has a label:
    1 = genuine pair (same person), 0 = impostor pair (different people).

At a decision threshold t, a pair is accepted when score >= t:
    FAR (False Acceptance Rate) = impostor pairs accepted / all impostor pairs
    FRR (False Rejection Rate)  = genuine pairs rejected  / all genuine pairs
    EER (Equal Error Rate)      = the error rate at the threshold where FAR == FRR

Trust in the paper's framework is scaled by accuracy, i.e. (1 - EER).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class VerificationResult:
    eer: float          # equal error rate, 0..1
    threshold: float    # score threshold where FAR ~= FRR
    far: float          # FAR at that threshold
    frr: float          # FRR at that threshold
    n_genuine: int
    n_impostor: int

    @property
    def accuracy(self) -> float:
        """Fraction of decisions that are correct at the EER point: 1 - EER."""
        return 1.0 - self.eer

    def __str__(self) -> str:
        return (f"EER {self.eer:.2%} (FAR {self.far:.2%}, FRR {self.frr:.2%}) "
                f"at threshold {self.threshold:.4f}  [{self.n_genuine} genuine, {self.n_impostor} impostor pairs]")


def _validate(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError("scores and labels must be 1-D arrays of the same length")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 (impostor) or 1 (genuine)")
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise ValueError("need at least one genuine and one impostor pair")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return scores, labels


def far_frr(scores, labels, threshold):
    """FAR and FRR when accepting pairs with score >= threshold."""
    scores, labels = _validate(scores, labels)
    accepted = scores >= threshold
    far = accepted[labels == 0].mean()
    frr = (~accepted[labels == 1]).mean()
    return float(far), float(frr)


def far_frr_curve(scores, labels):
    """FAR and FRR at every distinct threshold, sorted from lowest to highest threshold."""
    scores, labels = _validate(scores, labels)
    thresholds = np.unique(scores)
    genuine = np.sort(scores[labels == 1])
    impostor = np.sort(scores[labels == 0])
    # accepted = score >= t, so rejected = count of scores strictly below t
    frr = np.searchsorted(genuine, thresholds, side="left") / len(genuine)
    far = 1.0 - np.searchsorted(impostor, thresholds, side="left") / len(impostor)
    return thresholds, far, frr


def compute_eer(scores, labels) -> VerificationResult:
    """Equal error rate and the threshold where FAR and FRR cross."""
    scores, labels = _validate(scores, labels)
    thresholds, far, frr = far_frr_curve(scores, labels)
    # FAR falls and FRR rises as the threshold increases; take the crossing point
    i = int(np.argmin(np.abs(far - frr)))
    return VerificationResult(
        eer=float((far[i] + frr[i]) / 2),
        threshold=float(thresholds[i]),
        far=float(far[i]),
        frr=float(frr[i]),
        n_genuine=int(labels.sum()),
        n_impostor=int((labels == 0).sum()),
    )
