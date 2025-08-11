"""Script to download EMNIST MCQ dataset used for MCQ Clustering model training."""

from torchvision.datasets import EMNIST
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]

DATA_DIR = Path(PROJECT_ROOT / "dataset/mcq")
DATA_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Downloading EMNIST dataset at {DATA_DIR}")
    EMNIST(
        root=str(DATA_DIR),  # where to store
        split="letters",  # A–Z (labels 1–26)
        train=True,  # training split
        download=True,  # fetch if not already present
    )

    EMNIST(
        root=str(DATA_DIR), split="letters", train=False, download=True  # test split
    )
    print("EMNIST letters dataset downloaded.")
