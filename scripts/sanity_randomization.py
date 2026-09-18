import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import (
    read_split_csv,
    HistologyDataset,
)
from src.model import PathologyResolver
from src.utils import (
    load_checkpoint,
    load_config,
)


def flatten_normalize(x):
    x = x.flatten(1)

    x = x - x.mean(
        dim=1,
        keepdim=True,
    )

    x = x / (
        x.norm(
            dim=1,
            keepdim=True,
        )
        + 1e-8
    )

    return x


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.pt",
    )

    parser.add_argument(
        "--split",
        required=True,
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = HistologyDataset(
        read_split_csv(args.split),
        image_size=cfg["data"][
            "image_size"
        ],
        train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=cfg["data"][
            "num_workers"
        ],
    )

    model = PathologyResolver(
        num_classes=cfg["model"][
            "num_classes"
        ],
        pretrained=False,
        feature_dim=cfg["model"][
            "feature_dim"
        ],
        dropout=cfg["model"]["dropout"],
        constrained=True,
    ).to(device)

    load_checkpoint(
        args.checkpoint,
        model,
        device=device,
    )

    model.eval()

    trained_maps = []
    randomized_maps = []

    with torch.no_grad():
        for images, _, _ in tqdm(loader):
            images = images.to(device)

            trained_output = model(
                images,
                return_aux=True,
            )

            trained_maps.append(
                trained_output[
                    "resolved_evidence"
                ].cpu()
            )

    randomized_model = PathologyResolver(
        num_classes=cfg["model"][
            "num_classes"
        ],
        pretrained=False,
        feature_dim=cfg["model"][
            "feature_dim"
        ],
        dropout=cfg["model"]["dropout"],
        constrained=True,
    ).to(device)

    randomized_model.eval()

    with torch.no_grad():
        for images, _, _ in tqdm(
            loader,
            desc="randomized",
        ):
            images = images.to(device)

            randomized_output = (
                randomized_model(
                    images,
                    return_aux=True,
                )
            )

            randomized_maps.append(
                randomized_output[
                    "resolved_evidence"
                ].cpu()
            )

    trained_maps = torch.cat(
        trained_maps,
        dim=0,
    )

    randomized_maps = torch.cat(
        randomized_maps,
        dim=0,
    )

    trained_maps = flatten_normalize(
        trained_maps
    )

    randomized_maps = flatten_normalize(
        randomized_maps
    )

    similarities = (
        trained_maps
        * randomized_maps
    ).sum(dim=1)

    absolute_difference = (
        trained_maps
        - randomized_maps
    ).abs().mean(dim=1)

    print({
        "mean_cosine_similarity": (
            similarities.mean().item()
        ),
        "median_cosine_similarity": (
            similarities.median().item()
        ),
        "p25_cosine_similarity": (
            torch.quantile(
                similarities,
                0.25,
            ).item()
        ),
        "p75_cosine_similarity": (
            torch.quantile(
                similarities,
                0.75,
            ).item()
        ),
        "std_cosine_similarity": (
            similarities.std().item()
        ),
        "mean_absolute_difference": (
            absolute_difference.mean().item()
        ),
        "median_absolute_difference": (
            absolute_difference.median().item()
        ),
    })


if __name__ == "__main__":
    main()
