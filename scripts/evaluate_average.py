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
from src.losses import (
    make_random_deletion_mask,
    make_top_deletion_mask,
)
from src.metrics import (
    classification_accuracy,
    confidence_drop,
)
from src.model import PathologyResolver
from src.utils import (
    load_checkpoint,
    load_config,
)


def evaluate(args):
    cfg = load_config(args.config)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    samples = read_split_csv(
        args.split
    )

    dataset = HistologyDataset(
        samples,
        image_size=cfg["data"][
            "image_size"
        ],
        train=False,
    )

    loader = DataLoader(
        dataset,
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

    resolver_accuracy = 0.0
    average_accuracy = 0.0
    resolver_top_drop = 0.0
    average_top_drop = 0.0
    resolver_random_drop = 0.0
    average_random_drop = 0.0
    total = 0

    ratio = args.ratio

    for images, targets, _ in tqdm(loader):
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            resolver_out = model(
                images,
                return_aux=True,
            )

            resolver_logits = (
                resolver_out["logits"]
            )

            resolver_evidence = (
                resolver_out[
                    "resolved_evidence"
                ]
            )

            average_logits, average_evidence = (
                model.forward_average(images)
            )

            resolver_top_mask = (
                make_top_deletion_mask(
                    resolver_evidence,
                    ratio=ratio,
                )
            )

            resolver_random_drops = []

            for _ in range(10):
                resolver_random_mask = (
                    make_random_deletion_mask(
                        resolver_evidence,
                        ratio=ratio,
                    )
                )

                resolver_random_logits = (
                    model.forward_masked_with_evidence(
                        images,
                        resolver_random_mask,
                        resolver_evidence,
                    )
                )

                resolver_random_drops.append(
                    confidence_drop(
                        resolver_logits,
                        resolver_random_logits,
                        targets,
                    )
                )

            resolver_random_drop_batch = (
                sum(resolver_random_drops)
                / len(resolver_random_drops)
            )

            average_top_mask = (
                make_top_deletion_mask(
                    average_evidence,
                    ratio=ratio,
                )
            )

            average_random_drops = []

            for _ in range(10):
                average_random_mask = (
                    make_random_deletion_mask(
                        average_evidence,
                        ratio=ratio,
                    )
                )

                average_random_logits = (
                    model.forward_masked_with_evidence(
                        images,
                        average_random_mask,
                        average_evidence,
                    )
                )

                average_random_drops.append(
                    confidence_drop(
                        average_logits,
                        average_random_logits,
                        targets,
                    )
                )

            average_random_drop_batch = (
                sum(average_random_drops)
                / len(average_random_drops)
            )

            resolver_top_logits = (
                model.forward_masked_with_evidence(
                    images,
                    resolver_top_mask,
                    resolver_evidence,
                )
            )

            resolver_random_logits = (
                model.forward_masked_with_evidence(
                    images,
                    resolver_random_mask,
                    resolver_evidence,
                )
            )

            average_top_logits = (
                model.forward_masked_with_evidence(
                    images,
                    average_top_mask,
                    average_evidence,
                )
            )

            average_random_logits = (
                model.forward_masked_with_evidence(
                    images,
                    average_random_mask,
                    average_evidence,
                )
            )

        batch_size = images.size(0)
        total += batch_size

        resolver_accuracy += (
            classification_accuracy(
                resolver_logits,
                targets,
            )
            * batch_size
        )

        average_accuracy += (
            classification_accuracy(
                average_logits,
                targets,
            )
            * batch_size
        )

        resolver_top_drop += (
            confidence_drop(
                resolver_logits,
                resolver_top_logits,
                targets,
            )
            * batch_size
        )

        resolver_random_drop += (
            resolver_random_drop_batch
            * batch_size
        )

        average_top_drop += (
            confidence_drop(
                average_logits,
                average_top_logits,
                targets,
            )
            * batch_size
        )

        average_random_drop += (
            average_random_drop_batch
            * batch_size
        )

    resolver_top = (
        resolver_top_drop / total
    )

    resolver_random = (
        resolver_random_drop / total
    )

    average_top = (
        average_top_drop / total
    )

    average_random = (
        average_random_drop / total
    )

    print({
        "resolver_accuracy": (
            resolver_accuracy / total
        ),
        "average_accuracy": (
            average_accuracy / total
        ),
        "resolver_top_drop": resolver_top,
        "resolver_random_drop": (
            resolver_random
        ),
        "resolver_gap": (
            resolver_top - resolver_random
        ),
        "average_top_drop": average_top,
        "average_random_drop": (
            average_random
        ),
        "average_gap": (
            average_top - average_random
        ),
    })


if __name__ == "__main__":
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

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()
    evaluate(args)
