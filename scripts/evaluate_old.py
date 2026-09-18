import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import (
    collect_samples,
    HistologyDataset,
)
from src.model import PathologyResolver
from src.metrics import (
    classification_accuracy,
    confidence_drop,
    effective_support,
    attribution_entropy,
)
from src.utils import load_config, load_checkpoint


def evaluate(args):
    cfg = load_config(args.config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    samples = collect_samples(
        args.data_root,
        max_samples_per_class=None
    )

    dataset = HistologyDataset(
        samples,
        image_size=cfg["data"]["image_size"],
        train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model = PathologyResolver(
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
        feature_dim=cfg["model"]["feature_dim"],
        dropout=cfg["model"]["dropout"],
        constrained=True,
    ).to(device)

    load_checkpoint(
        args.checkpoint,
        model,
        device=device
    )

    model.eval()

    total_acc = 0.0
    total_drop = 0.0
    total_support = 0.0
    total_entropy = 0.0
    count = 0

    for images, targets, _ in tqdm(loader):
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(
                images,
                return_aux=True
            )

            logits = outputs["logits"]
            evidence = outputs["resolved_evidence"]

            threshold = torch.quantile(
                evidence,
                0.8,
                dim=0,
                keepdim=True
            )

            top_mask = (
                evidence < threshold
            ).float()

            masked_logits = model.forward_masked(
                images,
                top_mask
            )

        batch_size = images.size(0)

        total_acc += classification_accuracy(
            logits,
            targets
        ) * batch_size

        total_drop += confidence_drop(
            logits,
            masked_logits,
            targets
        ) * batch_size

        total_support += effective_support(
            evidence
        ) * batch_size

        total_entropy += attribution_entropy(
            evidence
        ) * batch_size

        count += batch_size

    print({
        "accuracy": total_acc / count,
        "confidence_drop": total_drop / count,
        "effective_support": total_support / count,
        "attribution_entropy": total_entropy / count,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/base.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.pt"
    )
    parser.add_argument(
        "--data_root",
        required=True
    )
    args = parser.parse_args()
    evaluate(args)

