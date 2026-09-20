import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import (
    HistologyDataset,
    read_split_csv,
)
from src.model import PathologyResolver
from src.utils import (
    load_checkpoint,
    load_config,
)
from baselines.gradcam_baseline import (
    make_gradcam,
)
from baselines.ig_baseline import (
    make_ig,
)


def get_map(
    model,
    images,
    targets,
    method,
    ig_steps,
):
    if method == "gradcam":
        with torch.enable_grad():
            result = make_gradcam(
                model,
                images,
                targets,
            )

        return result.detach()

    if method == "ig":
        with torch.enable_grad():
            result = make_ig(
                model,
                images,
                targets,
                steps=ig_steps,
            )

        return result.detach()

    raise ValueError(method)


def normalize_map(value):
    value = value.clamp(min=0.0)

    return value / (
        value.sum(
            dim=1,
            keepdim=True,
        )
        + 1e-8
    )


def collect_maps(
    model,
    loader,
    device,
    method,
    ig_steps,
):
    maps = []

    for images, targets, _ in tqdm(
        loader,
        desc=method,
    ):
        images = images.to(device)
        targets = targets.to(device)

        value = get_map(
            model,
            images,
            targets,
            method,
            ig_steps,
        )

        maps.append(
            normalize_map(value).cpu()
        )

    return torch.cat(
        maps,
        dim=0,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/seed_42.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--split",
        required=True,
    )

    parser.add_argument(
        "--method",
        choices=[
            "gradcam",
            "ig",
        ],
        required=True,
    )

    parser.add_argument(
        "--ig_steps",
        type=int,
        default=16,
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
        batch_size=16,
        shuffle=False,
        num_workers=cfg["data"][
            "num_workers"
        ],
    )

    trained_model = PathologyResolver(
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
        trained_model,
        device=device,
    )

    trained_model.eval()

    random_model = PathologyResolver(
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

    random_model.eval()

    trained_maps = collect_maps(
        trained_model,
        loader,
        device,
        args.method,
        args.ig_steps,
    )

    random_maps = collect_maps(
        random_model,
        loader,
        device,
        args.method,
        args.ig_steps,
    )

    trained_maps = F.normalize(
        trained_maps,
        dim=1,
    )

    random_maps = F.normalize(
        random_maps,
        dim=1,
    )

    similarities = (
        trained_maps
        * random_maps
    ).sum(dim=1)

    differences = (
        trained_maps
        - random_maps
    ).abs().mean(dim=1)

    print({
        "method": args.method,
        "mean_cosine_similarity": (
            similarities.mean().item()
        ),
        "median_cosine_similarity": (
            similarities.median().item()
        ),
        "mean_absolute_difference": (
            differences.mean().item()
        ),
        "median_absolute_difference": (
            differences.median().item()
        ),
    })


if __name__ == "__main__":
    main()
