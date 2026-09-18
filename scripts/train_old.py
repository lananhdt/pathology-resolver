import argparse
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import (
    collect_samples,
    split_samples,
    HistologyDataset,
)
from src.model import PathologyResolver
from src.losses import total_loss
from src.metrics import classification_accuracy
from src.utils import (
    load_config,
    set_seed,
    save_checkpoint,
)


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    cfg,
    train=True,
):
    model.train(train)

    total = 0.0
    total_acc = 0.0
    n_batches = 0

    iterator = tqdm(
        loader,
        desc="train" if train else "val"
    )

    for images, targets, _ in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with autocast(
            enabled=cfg["training"]["amp"]
            and device.type == "cuda"
        ):
            outputs = model(
                images,
                return_aux=True
            )

            logits = outputs["logits"]
            evidence = outputs["resolved_evidence"]

            augmented_images = torch.flip(
                images,
                dims=[-1]
            )

            augmented_outputs = model(
                augmented_images,
                return_aux=True
            )

            augmented_evidence = (
                augmented_outputs["resolved_evidence"]
            )

            mask = (
                evidence <
                torch.quantile(
                    evidence.detach(),
                    cfg["loss"]["mask_ratio"],
                    dim=0,
                    keepdim=True,
                )
            ).float()

            masked_logits = model.forward_masked(
                images,
                mask
            )

            loss, loss_values = total_loss(
                original_logits=logits,
                targets=targets,
                original_evidence=evidence,
                augmented_evidence=augmented_evidence,
                masked_logits=masked_logits,
                faithfulness_weight=cfg["loss"][
                    "faithfulness_weight"
                ],
                consistency_weight=cfg["loss"][
                    "consistency_weight"
                ],
                sparsity_weight=cfg["loss"][
                    "sparsity_weight"
                ],
            )

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["training"]["grad_clip"]
            )

            scaler.step(optimizer)
            scaler.update()

        total += loss.item()
        total_acc += classification_accuracy(
            logits.detach(),
            targets
        )
        n_batches += 1

        iterator.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{total_acc / n_batches:.4f}",
        })

    return {
        "loss": total / max(n_batches, 1),
        "accuracy": total_acc / max(n_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/base.yaml"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    samples = collect_samples(
        cfg["data"]["train_root"],
        cfg["data"]["max_samples_per_class"]
    )

    train_samples, val_samples = split_samples(
        samples,
        cfg["data"]["val_ratio"],
        cfg["seed"]
    )

    train_dataset = HistologyDataset(
        train_samples,
        cfg["data"]["image_size"],
        train=True,
    )

    val_dataset = HistologyDataset(
        val_samples,
        cfg["data"]["image_size"],
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model = PathologyResolver(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        feature_dim=cfg["model"]["feature_dim"],
        dropout=cfg["model"]["dropout"],
        constrained=True,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scaler = GradScaler(
        enabled=cfg["training"]["amp"]
        and device.type == "cuda"
    )

    checkpoint_dir = Path(
        cfg["output"]["checkpoint_dir"]
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_val_loss = float("inf")

    for epoch in range(cfg["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}")

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            cfg,
            train=True,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            scaler,
            device,
            cfg,
            train=False,
        )

        print({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        })

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch,
            val_metrics,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
            )


if __name__ == "__main__":
    main()

