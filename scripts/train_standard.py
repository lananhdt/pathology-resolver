import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import (
    read_split_csv,
    HistologyDataset,
)
from src.model import (
    StandardPathologyClassifier,
)
from src.utils import (
    load_config,
    set_seed,
)


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    train=True,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets, _ in tqdm(
        loader,
        leave=False,
    ):
        images = images.to(device)
        targets = targets.to(device)

        if train:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(
                logits,
                targets,
            )

            if train:
                loss.backward()
                optimizer.step()

        total_loss += (
            loss.item() * images.size(0)
        )

        correct += (
            logits.argmax(dim=1)
            == targets
        ).sum().item()

        total += images.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/seed_42.yaml",
    )

    parser.add_argument(
        "--output",
        default="checkpoints/standard_42.pt",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_data = HistologyDataset(
        read_split_csv(
            cfg["data"]["train_split"]
        ),
        image_size=cfg["data"][
            "image_size"
        ],
        train=True,
    )

    val_data = HistologyDataset(
        read_split_csv(
            cfg["data"]["val_split"]
        ),
        image_size=cfg["data"][
            "image_size"
        ],
        train=False,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=cfg["training"][
            "batch_size"
        ],
        shuffle=True,
        num_workers=cfg["data"][
            "num_workers"
        ],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=cfg["training"][
            "batch_size"
        ],
        shuffle=False,
        num_workers=cfg["data"][
            "num_workers"
        ],
        pin_memory=True,
    )

    model = StandardPathologyClassifier(
        num_classes=cfg["model"][
            "num_classes"
        ],
        pretrained=cfg["model"][
            "pretrained"
        ],
        feature_dim=cfg["model"][
            "feature_dim"
        ],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"][
            "weight_decay"
        ],
    )

    best_loss = float("inf")

    for epoch in range(
        cfg["training"]["epochs"]
    ):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            train=False,
        )

        print({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        })

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]

            Path(args.output).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": val_metrics,
                },
                args.output,
            )


if __name__ == "__main__":
    main()
