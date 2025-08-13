import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from collections import Counter

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


def get_best_clustering(
    X: np.ndarray, thresholds: list[float], metric="silhouette"
) -> np.ndarray:
    """Get the best clustering of X by searching for optimal threshold that maximizes the metric.

    This function defaults with AgglomerativeClustering clustering algorithm.

    Args:
        X: the feature matrix.
        thresholds: the choices of distance thresholds.
        metric: which metric to optimize. Currently supports: "silhouette" and "davies".

    Returns:
        A numpy array of clusterId where the order matches with the
        inputs (index 0 provides Id for row 0 of X)
    """
    best_score = -np.inf if metric == "silhouette" else np.inf
    best_labels = np.array([])

    for t in thresholds:
        clustering = AgglomerativeClustering(
            n_clusters=None, metric="euclidean", distance_threshold=t
        )
        labels = clustering.fit_predict(X)
        # need at least 2 clusters to score
        if len(set(labels)) < 2:
            continue

        if metric == "silhouette":
            score = silhouette_score(X, labels)
            # silhouette: higher -> better
            if score > best_score:
                best_score = score
                best_labels = labels

        elif metric == "davies":
            score = davies_bouldin_score(X, labels)
            # DB index: lower -> better
            if score < best_score:
                best_score = score
                best_labels = labels

    return best_labels


def purity_score(y_true, y_pred) -> float:
    """Compute purity score.

    Args:
        y_true: an iterable of true labels.
        y_pred: an iterable of predicted labels.

    Returns:
        The purity score of the prediction.

    Note:
        - For each predicted label we keep track of the count for each true label.
        - Then we take the majority of the true label for each predicted label
        - We sum those majorities then divide with total data
    """
    counts = {}
    for t, p in zip(y_true, y_pred):
        counts.setdefault(int(p), Counter())
        counts[int(p)][int(t)] += 1
    return sum(max(c.values()) for c in counts.values()) / len(y_true)


def compute_cosine_logits(model: nn.Module, emb: Tensor, s: float) -> Tensor:
    """Compute cosine-similarity-based logits between input embeddings and classifier weights.

    Normalizes both the embeddings and the classifier weights to unit length so the
    dot product equals the cosine similarity, then scales the result by `s` to control
    the logit magnitude for use in softmax classification.

    Args:
        model : A model with `head.classifier.weight` as the class weight matrix.
        emb : Input embedding vectors from the model backbone.
        s : Scaling factor applied to the cosine similarity values.

    Returns
        Scaled cosine similarity logits of shape (batch_size, num_classes).
    """
    W = model.head.classifier.weight  # (num_classes, embed_dim)
    emb_n = F.normalize(emb, dim=1)  # normalize embeddings to unit length
    W_n = F.normalize(W, dim=1)  # normalize weights to unit length
    logits = s * (emb_n @ W_n.t())  # cosine similarity × scale
    return logits


class CenterLoss(nn.Module):
    """Center Loss for improving feature discrimination.

    This loss penalizes the Euclidean distance between the normalized feature
    vectors of samples and their corresponding (learned) class centers, encouraging features
    of the same class to cluster together on the unit hypersphere.

    Args:
        num_classes: Number of distinct classes.
        feat_dim: Dimensionality of the feature vectors.
        alpha: Scaling factor controlling the contribution of the center loss.

    Notes
    -----
    - Centers are L2-normalized so that Euclidean distance approximates angular distance.
    - Initialized with orthogonal vectors for a good starting spread.
    - idea from: https://doi.org/10.1007/978-3-319-46478-7_31
    """

    def __init__(self, num_classes: int, feat_dim: int, alpha: float = 1.0):
        super().__init__()
        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim)
        )  # make as trainable matrix
        nn.init.orthogonal_(
            self.centers
        )  # init with maximum difference (based on cosine similarity)
        self.alpha = alpha

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Both feats and centers on the sphere -> Euclidean ~ angular
        feats = F.normalize(feats, dim=1)
        centers = F.normalize(self.centers, dim=1)

        # get the centroids for each sample in the batch
        c = centers.index_select(0, labels)  # (B, D)

        # the loss is the scaled (self.alpha) mean squared distance between each sample and the centroid
        return self.alpha * ((feats - c).pow(2).sum(dim=1)).mean()
