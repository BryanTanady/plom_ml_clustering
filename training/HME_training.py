"""This training solely optimizes multi-label classification task.

The labels are defined as the 229 symbols and are encoded as 1 if presents and
0 if not.
"""

import os
import json
import argparse
import random
from contextlib import nullcontext

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR

from torch.amp.autocast_mode import autocast  # type: ignore
from torch.amp.grad_scaler import GradScaler

from sklearn.cluster import KMeans
from pathlib import Path


from model.model_architecture import HMESymbolicNet
from utils.utils import purity_score


ROOT = Path(__file__).parent.parent
GENERATED_DATA_PATH = ROOT / "dataset" / "hme" / "generated_data"

TRAIN_ROOT = GENERATED_DATA_PATH / "image"
MULTI_HOT_LABEL = GENERATED_DATA_PATH / "multihot_label.json"
TEST_DATA = GENERATED_DATA_PATH / "test.json"

MODEL_OUTPUT_PATH = ROOT / "weights" / "hme_model.pth"
SAVE_PATH = ROOT / "weights" / "hme_model_resumed.pth"

os.makedirs(MODEL_OUTPUT_PATH.parent, exist_ok=True)

with open(GENERATED_DATA_PATH / "vocab.json", "r") as f:
    vocab = json.load(f)
    VOCAB_LENGTH = len(vocab)


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    """Allow behavior to be "more deterministic" for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
class TrainHMEDataset(Dataset):
    """Training Dataset for HME

    Args:
      img_dir: directory of training images
      label_json: mapping filename -> multi-hot of symbols.
    """

    def __init__(self, img_dir, label_json, transform=None):
        self.img_dir = img_dir
        with open(label_json, "r") as f:
            raw = json.load(
                f
            )  # raw maps filename to list of multi-hot encoding of symbols

        # see if it really maps to a list with length = VOCAB_LENGTH
        first = next(iter(raw.values()))
        if not isinstance(first, list) or len(first) != VOCAB_LENGTH:
            raise ValueError(
                f"Unexpected format or size of multi hot encoding: {first}"
            )

        self.label_map = raw

        # ensure we only train on labelled data
        self.files = sorted([fn for fn in os.listdir(img_dir) if fn in self.label_map])

        if not self.files:
            raise ValueError(f"No labeled images found in {img_dir}.")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Get an item of index idx in the Dataset.

        Returns:
            - tensor representation of the grayscale image (with train_tf applied).
            - Multi-hot encoding of the symbols in the image
            - filename.
        """
        fn = self.files[idx]
        path = os.path.join(self.img_dir, fn)
        img = Image.open(path).convert("L")  # 'L': load as grayscale
        tensor = self.transform(img) if self.transform else transforms.ToTensor()(img)
        vec = torch.tensor(self.label_map[fn], dtype=torch.float32)
        return tensor, vec, fn


class TestHMEDataset(Dataset):
    """Testing dataset for HME.

    Args:
        img_dir: directory of test images
        test_label_json: mapping filename -> an integer representing its class
            (determined from the exact equation in the image).

    """

    def __init__(self, img_dir, test_label_json, transform=None):
        self.img_dir = img_dir
        with open(test_label_json, "r") as f:
            self.label_map = json.load(f)

        self.files = sorted([fn for fn in os.listdir(img_dir) if fn in self.label_map])
        if not self.files:
            raise ValueError(f"No labeled images found in {img_dir}.")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Get an item of index idx in the Dataset.

        Returns:
            - tensor representation of the grayscale image (with test_tf applied).
            - Multi-hot encoding of the symbols in the image
            - filename.
        """
        fn = self.files[idx]
        path = os.path.join(self.img_dir, fn)
        img = Image.open(path).convert("L")
        tensor = self.transform(img) if self.transform else transforms.ToTensor()(img)
        cls = int(self.label_map[fn])
        return tensor, cls, fn


# -----------------------------------------------------------------------------
# Training / Eval
# -----------------------------------------------------------------------------
def extract_embeddings(model: nn.Module, loader: DataLoader, device: torch.device):
    """Use given model to extract embeddings for all batches in loader.

    Returns:
        - A tensor of embeddings.
        - A 1D numpy array of labels for each sample in all batches.
    """
    model.eval()
    all_embs, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            emb, _ = model(imgs)
            all_embs.append(emb.detach().cpu())
            all_labels.extend([int(x) for x in lbls])
    embs = torch.cat(
        all_embs, dim=0
    )  # combine all per-batch embeddings to [N, Emb_dim] tensor
    return embs, np.array(all_labels)


def kmeans_purity(embs: torch.Tensor, labels: np.ndarray, random_state: int = 42):
    """Compute purity by clustering embds with kmeans.

    Args:
        embs: the embeddings for
        labels: the precomputed class label for each sample, where
            same equations are given same labels. The value doesn't really
            matter.
        random_state: just for better reproducibility.

    -------------
    Note: It may feel like "cheating" as we are giving away
    the exact n_clusters here, but the point is to keep track
    whether there is better separability as traininig progresses.
    """
    n_clusters = int(len(np.unique(labels)))
    feats = F.normalize(embs, dim=1).cpu().numpy()
    km = KMeans(
        n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state
    )
    preds = km.fit_predict(feats)
    purity = purity_score(labels, preds)
    return purity


def compute_class_stats(label_json: str, num_classes: int):
    """Compute the weights for each symbol based on their frequency in the dataset."""

    # load a mapping of filename -> list (multi hot of symbols)
    with open(label_json, "r") as f:
        data = json.load(f)

    # Compute the overall frequency of all symbols
    N = len(data)
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for v in data.values():
        counts += torch.tensor(v, dtype=torch.float32)

    # BCE pos_weight for positives of rare classes
    pos = counts.clamp_min(1.0)  # clamp to avoid div by zero
    neg = (N - counts).clamp_min(1.0)
    pos_weight = neg / pos  # [C]

    # Cui et al (2019) give weights to rarer samples: better for long tailed dataset.
    beta = 0.999
    eff = (1.0 - beta) / (1.0 - torch.pow(beta, counts.clamp_min(1.0)))
    cls_weight = 1.0 / eff
    cls_weight = cls_weight / cls_weight.mean()  # normalize around 1

    # Prior logit for logit adjustment
    pi = (counts / max(N, 1)).clamp(1e-6, 1 - 1e-6)
    prior_logit = torch.log(pi / (1 - pi))

    return pos_weight, cls_weight, prior_logit


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Train HMESymbolicNet on multi-label symbol vectors and evaluate clustering purity."
    )
    # Data / paths
    parser.add_argument("--img-height", type=int, default=128)
    parser.add_argument("--img-width", type=int, default=256)

    # Model
    parser.add_argument("--num-classes", type=int, default=VOCAB_LENGTH)
    parser.add_argument("--emb-dim", type=int, default=256)

    # Train
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)

    # Eval / save
    parser.add_argument("--test-interval", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint .pth to resume from (model.state_dict) or to eval-only",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, just evaluate checkpoint specified by --resume",
    )

    args = parser.parse_args()

    # init seed (for reproducibility)
    seed_everything(args.seed)

    # init device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    print(f"Using device: {device}")

    # Get train and test paths
    train_dir = os.path.join(TRAIN_ROOT, "train")
    test_dir = os.path.join(TRAIN_ROOT, "test")

    # Transforms
    train_tf = transforms.Compose(
        [
            transforms.Resize((args.img_height, args.img_width)),
            transforms.RandomAffine(
                20, translate=(0.2, 0.3), scale=(0.5, 1.5), shear=10
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5
            ),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.RandomApply([transforms.RandomAdjustSharpness(2)], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.01, 0.03), value="random"),
            transforms.Normalize(0.5, 0.5),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize((args.img_height, args.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    # Data
    pin_mem = is_cuda
    train_ds = TrainHMEDataset(train_dir, MULTI_HOT_LABEL, train_tf)
    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    test_ds = TestHMEDataset(test_dir, TEST_DATA, test_tf)
    test_ld = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    pos_weight, cls_weight, prior_logit = compute_class_stats(
        MULTI_HOT_LABEL, args.num_classes
    )
    pos_weight = pos_weight.to(device)
    cls_weight = cls_weight.to(device)
    prior_logit = prior_logit.to(device)

    TAU_LA = 0.5

    # Model
    model = HMESymbolicNet(out_classes=args.num_classes, emb_dim=args.emb_dim).to(
        device
    )

    # Optionally resume
    if args.resume:
        if os.path.isfile(args.resume):
            state = torch.load(args.resume, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded checkpoint from {args.resume}")
        else:
            raise FileNotFoundError(f"--resume path not found: {args.resume}")

    if args.eval_only:
        embs, labels = extract_embeddings(model, test_ld, device)
        purity = kmeans_purity(embs, labels)
        print(f"[EVAL-ONLY] KMeans purity on embeddings: {purity:.4f}")
        return

    # what to optimize
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # optimize under what criteria
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    # How does learning rate change
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr * 5,
        steps_per_epoch=len(train_ld),
        epochs=args.epochs,
    )

    # Automatic mixed precision: uses float16 where safe (little impact to stability)
    # Used for faster training
    scaler = GradScaler(enabled=is_cuda)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        amp_ctx = autocast(device_type="cuda") if is_cuda else nullcontext()

        for imgs, vecs, _ in tqdm(train_ld, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs, vecs = imgs.to(device, non_blocking=True), vecs.to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                _, logits = model(imgs)

                # bias logits by priors to help rare classes
                logits = logits + TAU_LA * prior_logit

                # per-class BCE, then weight by cls importance; final mean
                bce_per_class = criterion(logits, vecs)  # [B, C]
                loss = (bce_per_class * cls_weight).mean()

            scaler.scale(loss).backward()
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += float(loss.detach().cpu())

        avg_loss = total_loss / max(1, len(train_ld))
        print(f"Epoch {epoch}  Train Loss: {avg_loss:.4f}")

        # Evaluate
        if epoch % args.test_interval == 0:
            with torch.no_grad():
                embs, labels = extract_embeddings(model, test_ld, device)
                purity = kmeans_purity(embs, labels)
                print(f"[Eval] KMeans purity on embeddings: {purity:.4f}")

        # Save
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved model checkpoint to {SAVE_PATH}")


if __name__ == "__main__":
    main()
