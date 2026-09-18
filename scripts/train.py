import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data import read_split_csv
from src.data import (
    collect_samples,
    split_samples,
    HistologyDataset,
)
from src.losses import (
    make_top_deletion_mask,
    total_loss,
)
from src.metrics import classification_accuracy
from src.model import PathologyResolver
from src.utils import (
    load_config,
    save_checkpoint,
    set_seed,
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
    if train:
        model.train()
    else:
        model.eval()

    total_loss_value = 0.0
    total_accuracy = 0.0
    num_batches = 0

    progress = tqdm(
        loader,
        desc="train" if train else "val",
    )

    for images, targets, _ in progress:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        if train:
            optimizer.zero_grad(
                set_to_none=True
            )

        use_amp = (
            cfg["training"]["amp"]
            and device.type == "cuda"
        )

        with torch.set_grad_enabled(train):
            with autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                outputs = model(
                    images,
                    return_aux=True,
                )

                logits = outputs["logits"]
                evidence = outputs[
                    "resolved_evidence"
                ]

                augmented_images = torch.flip(
                    images,
                    dims=[-1],
                )

                augmented_outputs = model(
                    augmented_images,
                    return_aux=True,
                )

                augmented_evidence = (
                    augmented_outputs[
                        "resolved_evidence"
                    ]
                )

                deletion_mask = (
                    make_top_deletion_mask(
                        evidence,
                        ratio=cfg["loss"][
                            "mask_ratio"
                        ],
                    )
                )

                masked_logits = (
                    model.forward_masked(
                        images,
                        deletion_mask,
                    )
                )

                loss, loss_values = total_loss(
                    original_logits=logits,
                    masked_logits=masked_logits,
                    targets=targets,
                    original_evidence=evidence,
                    augmented_evidence=(
                        augmented_evidence
                    ),
                    faithfulness_weight=(
                        cfg["loss"][
                            "faithfulness_weight"
                        ]
                    ),
                    consistency_weight=(
                        cfg["loss"][
                            "consistency_weight"
                        ]
                    ),
                    sparsity_weight=(
                        cfg["loss"][
                            "sparsity_weight"
                        ]
                    ),
                )

            if train:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg["training"][
                        "grad_clip"
                    ],
                )

                scaler.step(optimizer)
                scaler.update()

        batch_accuracy = classification_accuracy(
            logits.detach(),
            targets,
        )

        total_loss_value += loss.item()
        total_accuracy += batch_accuracy
        num_batches += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{total_accuracy / num_batches:.4f}",
        )

    return {
        "loss": (
            total_loss_value
            / max(num_batches, 1)
        ),
        "accuracy": (
            total_accuracy
            / max(num_batches, 1)
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    train_samples = read_split_csv(
        cfg["data"]["train_split"]
    )

    val_samples = read_split_csv(
        cfg["data"]["val_split"]
    )

    train_dataset = HistologyDataset(
        train_samples,
        image_size=cfg["data"][
            "image_size"
        ],
        train=True,
    )

    val_dataset = HistologyDataset(
        val_samples,
        image_size=cfg["data"][
            "image_size"
        ],
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"][
            "batch_size"
        ],
        shuffle=True,
        num_workers=cfg["data"][
            "num_workers"
        ],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"][
            "batch_size"
        ],
        shuffle=False,
        num_workers=cfg["data"][
            "num_workers"
        ],
        pin_memory=True,
    )

    model = PathologyResolver(
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
        constrained=True,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"][
            "weight_decay"
        ],
    )

    use_amp = (
        cfg["training"]["amp"]
        and device.type == "cuda"
    )

    scaler = GradScaler(
        device="cuda",
        enabled=use_amp,
    )

    checkpoint_dir = Path(
        cfg["output"]["checkpoint_dir"]
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    for epoch in range(
        cfg["training"]["epochs"]
    ):
        print(
            f"\nEpoch {epoch + 1}/"
            f"{cfg['training']['epochs']}"
        )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg=cfg,
            train=True,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg=cfg,
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
            best_val_loss = val_metrics[
                "loss"
            ]

            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
            )


if __name__ == "__main__":
    main()

