from model.model_architecture import MCQClusteringNet

#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, resnet34, ResNet34_Weights, ResNet18_Weights
from tqdm import tqdm


# ─── Helpers ────────────────────────────────────────────────────────
def filter_imagefolder(ds, allowed):
    new_mapping = {c: i for i, c in enumerate(allowed)}
    filtered = []
    for path, _ in ds.samples:
        cls = os.path.basename(os.path.dirname(path))
        if cls in allowed:
            filtered.append((path, new_mapping[cls]))
    ds.samples = filtered
    ds.targets = [lbl for _, lbl in filtered]
    ds.classes = allowed
    ds.class_to_idx = new_mapping
    return ds


class CombinedDataset(ConcatDataset):
    """Merge upper+lower, offsetting lower-labels by len(upper)."""

    def __init__(self, up_ds, lo_ds):
        super().__init__([up_ds, lo_ds])
        self.num_up = len(up_ds.classes)
        self.classes = up_ds.classes + lo_ds.classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        if idx < self.cumulative_sizes[0]:
            return self.datasets[0][idx]
        img, lbl = self.datasets[1][idx - self.cumulative_sizes[0]]
        return img, lbl + self.num_up


# ─── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="dataset")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--pre_epochs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    # Define class sets
    up_allowed = [chr(c) for c in range(ord("A"), ord("F") + 1)]
    lo_allowed = [chr(c) for c in range(ord("a"), ord("f") + 1)]
    lo_allowed.remove("c")


    train_tf = transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomAffine(
                45, translate=(0.2, 0.2), scale=(0.7, 1.3), shear=30
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5
            ),
            transforms.RandomChoice(
                [
                    transforms.GaussianBlur(kernel_size=3),
                    transforms.RandomAdjustSharpness(sharpness_factor=2),
                ],
                p=[0.3, 0.3],
            ),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomErasing(p=0.7, scale=(0.03, 0.15), value="random"),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # ─── Phase 1: EMNIST pre-training ────────────────────────────────
    emnist_train = datasets.EMNIST(
        root=args.data_root,
        split="letters",
        train=True,
        download=True,
        transform=train_tf,
        target_transform=lambda t: t - 1,
    )
    emnist_val = datasets.EMNIST(
        root=args.data_root,
        split="letters",
        train=False,
        download=True,
        transform=val_tf,
        target_transform=lambda t: t - 1,
    )
    emnist_tr_ld = DataLoader(
        emnist_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    emnist_vl_ld = DataLoader(
        emnist_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # ─── Phase 2: Custom A–F/a–f dataset ─────────────────────────────
    def load_and_filter(case, split):
        root = os.path.join(args.data_root, case, split)
        ds = datasets.ImageFolder(
            root, transform=(train_tf if split == "train" else val_tf)
        )
        return filter_imagefolder(ds, up_allowed if case == "upper" else lo_allowed)

    up_tr, up_val, up_te = [
        load_and_filter("upper", s) for s in ("train", "validation", "test")
    ]
    lo_tr, lo_val, lo_te = [
        load_and_filter("lower", s) for s in ("train", "validation", "test")
    ]
    train_ds = CombinedDataset(up_tr, lo_tr)
    val_ds = CombinedDataset(up_val, lo_val)
    test_ds = CombinedDataset(up_te, lo_te)
    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    test_ld = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    device = torch.device(args.device)

    # ====== Load model architecture ====
    model = MCQClusteringNet()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # -- 1) Pre-train head on EMNIST (26 classes)
    # model.fc = nn.Linear(model.fc.in_features, 26).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr * 10,
        epochs=args.pre_epochs,
        steps_per_epoch=len(emnist_tr_ld),
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )
    for ep in range(1, args.pre_epochs + 1):
        model.train()
        for x, y in tqdm(
            emnist_tr_ld, desc=f"EMNIST Train {ep}/{args.pre_epochs}", leave=False
        ):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
        model.eval()
        correct = 0
        for x, y in tqdm(emnist_vl_ld, desc="EMNIST Val", leave=False):
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
        acc = correct / len(emnist_vl_ld.dataset)
        print(f"[EMNIST] Epoch {ep}/{args.pre_epochs} · acc={acc:.4f}")

    # -- 2) Fine-tune head on custom (11 classes) with layer freezing
    # Freeze all backbone layers
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False
    # Replace and train only the final head
    model.fc = nn.Linear(in_features, 11).to(device)
    # Use a lower learning rate for the head
    optimizer = optim.Adam(
        [{"params": model.fc.parameters(), "lr": args.lr * 0.1}], weight_decay=1e-6
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_ld),
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )
    best_val = 0.0
    no_imp = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        # Unfreeze backbone after warm-up epoch 2
        if ep == 3:
            for name, param in model.named_parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=args.lr * 10,
                epochs=args.epochs - 2,
                steps_per_epoch=len(train_ld),
                pct_start=0.3,
                anneal_strategy="cos",
                div_factor=25.0,
                final_div_factor=1e4,
            )
        for x, y in tqdm(train_ld, desc=f"Fine Train {ep}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
        model.eval()
        correct = 0
        for x, y in tqdm(val_ld, desc="Fine Val", leave=False):
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
        acc = correct / len(val_ld.dataset)
        print(f"[Fine] Epoch {ep}/{args.epochs} · val_acc={acc:.4f}")
        if acc > best_val:
            best_val = acc
            no_imp = 0
            torch.save(model.state_dict(), "model.pth")
            print(f"  ▶ Saved best (acc={best_val:.4f})")
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print(f"Stopping early after {no_imp} no-improve")
                break

    # ─── Test ─────────────────────────────────────────────────────
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    correct = 0
    for x, y in tqdm(test_ld, desc="Test", leave=False):
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
    print(f"Test acc: {correct/len(test_ld.dataset):.4f}")


if __name__ == "__main__":
    main()
