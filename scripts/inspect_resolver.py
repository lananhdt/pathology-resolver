import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
from torch.utils.data import DataLoader

from src.data import (
    HistologyDataset,
    read_split_csv,
)
from src.model import PathologyResolver
from src.utils import (
    load_checkpoint,
    load_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = HistologyDataset(
        read_split_csv(args.split),
        image_size=224,
        train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
    )

    model = PathologyResolver(
        num_classes=9,
        pretrained=False,
        feature_dim=512,
        dropout=0.2,
        constrained=True,
    ).to(device)

    load_checkpoint(
        args.checkpoint,
        model,
        device=device,
    )

    model.eval()

    weights = []
    disagreements = []

    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)

            output = model(
                images,
                return_aux=True,
            )

            weights.append(
                output["branch_weights"].cpu()
            )

            disagreements.append(
                output["disagreement"].cpu()
            )

    weights = torch.cat(weights)
    disagreements = torch.cat(
        disagreements
    )

    print(
        "class branch weight:",
        weights[..., 0].mean().item(),
    )

    print(
        "morph branch weight:",
        weights[..., 1].mean().item(),
    )

    print(
        "branch weight std:",
        weights.std().item(),
    )

    print(
        "disagreement mean:",
        disagreements.mean().item(),
    )

    print(
        "disagreement std:",
        disagreements.std().item(),
    )


if __name__ == "__main__":
    main()
