"""Extension of MCQ model that also motivates intra-class compactness.

The compactness is enforced by adding CenterLoss where

The clustering is done on the Hellinger transformation of the generated probabilities.
"""

import argparse, os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from sklearn.cluster import KMeans

from model.model_architecture import (
    MCQClusteringNet2,
)

from PIL import Image

# ---------------------------- Config for labels ----------------------------
# We want: A,B,C,D,E,F,a,b,d,e,f  (11 classes) with c merged into C.
CANONICAL = ["A", "B", "C", "D", "E", "F", "a", "b", "d", "e", "f"]
MERGE_TO_CANON = {**{c: c for c in CANONICAL}, "c": "C"}
NUM_CLASSES = len(CANONICAL)  # 11


# ---------------------------- Dataset utils ----------------------------
def label_to_char_map(ds) -> dict[int, str]:
    """Build a label to character mapping for an EMNIST dataset.

    EMNIST stores a `mapping` array that links each class index to a Unicode
    code point for the corresponding character (digit, uppercase letter, or lowercase letter).

    This function reads the last column of `ds.mapping` (if 2D) or the entire
    array (if 1D) and converts each code point to its actual character.

    Args:
        ds (torchvision.datasets.EMNIST):
            An EMNIST dataset object (any split: ByClass, ByMerge, Letters, etc.).

    Returns:
        Dictionary where keys are integer class indices (0, 1, 2, ...)
        and values are the corresponding characters.
    """
    inv = {v: k for k, v in ds.class_to_idx.items()}
    return {i: str(inv[i]) for i in range(len(inv))}


def filter_and_remap(ds):
    """Keep only characters in CANONICAL plus lowercase 'c' (so we can merge 'c' and 'C'),
    then remap labels to 0..len(CANONICAL)-1 in CANONICAL order.

    This function MUTATES `ds.data` and `ds.targets` in place and returns `ds`.

    Args:
        ds (torchvision.datasets.EMNIST): Any EMNIST split.

    Returns:
        torchvision.datasets.EMNIST: The same dataset object with filtered samples and remapped targets.
    """
    # Build label to char mapping from dataset
    id2ch = label_to_char_map(ds)

    # Build the keep set and a char to new_index lookup
    keep_chars = set(CANONICAL) | {"c"}
    ch2new = {ch: i for i, ch in enumerate(CANONICAL)}

    # Compute mask of samples to keep
    targets_cpu = ds.targets.cpu()
    keep_mask = torch.tensor(
        [id2ch[int(t)] in keep_chars for t in targets_cpu],
        dtype=torch.bool,
        device=targets_cpu.device,
    )

    # Filter images
    ds.data = ds.data[keep_mask]

    # Remap targets with merging (e.g., 'c' -> 'C') then to 0..K-1 via ch2new
    new_targets = []
    for t in targets_cpu[keep_mask].tolist():
        ch = id2ch[int(t)]
        ch_can = MERGE_TO_CANON.get(ch, ch)  # merge if needed
        new_targets.append(ch2new[ch_can])

    ds.targets = torch.tensor(new_targets, dtype=torch.long, device=targets_cpu.device)

    # Sanity check
    num_classes = len(CANONICAL)
    assert ds.targets.min().item() == 0 and ds.targets.max().item() == num_classes - 1

    return ds


# =========== Other helpers ===============
class CenterLoss(nn.Module):
    """Center Loss for improving feature discrimination.

    This loss penalizes the Euclidean distance between the normalized feature
    vectors of samples and their corresponding class centers, encouraging features
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


def compute_cosine_logits(model: nn.Module, emb: Tensor, s: float) -> Tensor:
    """Compute cosine-similarity-based logits between input embeddings and classifier weights.

    Normalizes both the embeddings and the classifier weights to unit length so the
    dot product equals the cosine similarity, then scales the result by `s` to control
    the logit magnitude for use in softmax classification.

    Parameters
    ----------
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


# ---------------------------- Metrics ----------------------------
def purity_score(y_true, y_pred):
    counts = {}
    for t, p in zip(y_true, y_pred):
        counts.setdefault(int(p), Counter())
        counts[int(p)][int(t)] += 1
    return sum(max(c.values()) for c in counts.values()) / len(y_true)


def cluster_kmeans_probs(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    n_clusters: int = 11,
    tau: float = 1.0,
):
    """Perform KMeans clustering on model output probabilities (with Hellinger transformation).

    This function:
      1. Runs the given model on a data loader to collect class probability distributions
         (softmax over logits, with optional temperature scaling).
      2. Applies a Hellinger mapping (`sqrt(probs)`) to map probability vectors to the
         Hellinger space, improving Euclidean separability for clustering.
      3. Runs KMeans with a fixed number of clusters.

    Args:
        ----------
        model: A PyTorch model that, when called, returns (embedding, logits) for an input batch.
        dl: torch.utils.data.DataLoader
            DataLoader providing `(input_tensor, label)` pairs.
        device: Device on which to run model inference (`"cpu"` or `"cuda"`).
        n_clusters: Number of clusters to form in KMeans.
        tau: Temperature parameter for softmax scaling. Values < 1 sharpen the distribution,
            values > 1 soften it.

    Returns
    -------
    y_true : np.ndarray of shape (N,)
        Ground-truth labels collected from the dataset.
    y_pred : np.ndarray of shape (N,)
        Predicted cluster assignments from KMeans.

    Notes
    -----
    - The Hellinger mapping (`sqrt(probs)`) transforms probability distributions so that
      Euclidean distance better approximates their similarity, improving KMeans performance.
    - Assumes the model's forward pass returns a tuple `(embedding, logits)`.
    - The temperature scaling is applied before softmax: `softmax(logits / tau)`.
    """
    model.eval()
    Ps, y_true = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            # logits: (Batch, Class)
            emb, _ = model(xb)
            logits = compute_cosine_logits(model, emb, s=30)
            # dim = 1: apply softmax across class dimension (so probabilities across class sums to 1)
            probs = torch.softmax(logits / tau, dim=1).cpu().numpy()
            Ps.append(probs)
            y_true.append(yb.numpy())

    P = np.vstack(Ps)  # (N, C)
    Phi = np.sqrt(np.clip(P, 1e-12, 1.0))  # Hellinger map

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    y_pred = km.fit_predict(Phi)

    return np.concatenate(y_true), y_pred


# ---------------------------- Main script ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="dataset")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # derotate because EMNIST raw dataset is rotated
    rotation_fix = transforms.Lambda(
        lambda img: img.rotate(-90, expand=True).transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )
    )

    # Transforms
    train_tf = transforms.Compose(
        [
            rotation_fix,
            transforms.Grayscale(1),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomAffine(
                30, translate=(0.1, 0.15), scale=(0.7, 1.3), shear=30
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            transforms.RandomApply(
                [transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.3
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.06), value="random"),
            transforms.Normalize(0.5, 0.5),
        ]
    )
    val_tf = transforms.Compose(
        [
            rotation_fix,
            transforms.Grayscale(1),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    # Data: use byclass to get uppercase+lowercase
    tr = datasets.EMNIST(
        args.data_root, split="byclass", train=True, download=True, transform=train_tf
    )
    vl = datasets.EMNIST(
        args.data_root, split="byclass", train=False, download=True, transform=val_tf
    )

    # get only A-F and a-f (merge 'C' and 'c')
    tr = filter_and_remap(tr)
    vl = filter_and_remap(vl)

    tr_ld = DataLoader(
        tr, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    vl_ld = DataLoader(
        vl, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # load model architecture
    device = torch.device(args.device)
    model = MCQClusteringNet2(out_classes=NUM_CLASSES).to(device)

    # infer feature dimension once
    with torch.no_grad():
        x0, _ = next(iter(tr_ld))
        emb0, _ = model(x0[:2].to(device))
        feat_dim = emb0.shape[1]

    center_loss = CenterLoss(NUM_CLASSES, feat_dim, alpha=1.0).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Two optimizers: one for model, one for centers (small LR, no WD)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    optimizer_centers = optim.SGD(
        center_loss.parameters(), lr=args.lr * 0.1, momentum=0.9, weight_decay=0.0
    )

    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr * 5, epochs=args.epochs, steps_per_epoch=len(tr_ld)
    )

    lambda_c = 0.2

    # training
    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        # ---- train
        model.train()
        for x, y in tqdm(tr_ld, desc=f"Train {ep}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            optimizer_centers.zero_grad()

            emb, _ = model(x)
            logits = compute_cosine_logits(model, emb, s=30)

            ce = criterion(logits, y)
            cl = center_loss(emb, y)
            loss = ce + lambda_c * cl

            loss.backward()

            optimizer.step()
            optimizer_centers.step()
            scheduler.step()

        # ---- supervised accuracy (quick sanity)
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in tqdm(vl_ld, desc="Val", leave=False):
                x = x.to(device)
                emb, _ = model(x)
                logits = compute_cosine_logits(model, emb, s=30)

                preds = logits.argmax(1).cpu()
                correct += (preds == y).sum().item()
        acc = correct / len(vl_ld.dataset)
        best_acc = max(best_acc, acc)

        y_true, y_pred = cluster_kmeans_probs(
            model, vl_ld, device, n_clusters=NUM_CLASSES, tau=2.0
        )
        purity = purity_score(y_true, y_pred)
        produced_k = len(np.unique(y_pred))
        real_k = len(np.unique(y_true))

        print(
            f"[A-F/a-f (C/c merged)] Epoch {ep}/{args.epochs} | "
            f"acc={acc:.4f} | purity={purity:.4f} | "
            f"produced_k={produced_k} | real_k={real_k} | "
        )

    # Save final weight
    os.makedirs("weights", exist_ok=True)
    output_path = "weights/mcq_model_2.pth"
    torch.save(model.state_dict(), output_path)
    print(f"Saved final to {output_path}")


if __name__ == "__main__":
    main()
