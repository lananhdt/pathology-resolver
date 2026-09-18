import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data import read_split_csv
from src.data import (
    collect_samples,
    HistologyDataset,
)
from src.losses import (
    make_random_deletion_mask,
    make_top_deletion_mask,
)
from src.metrics import (
    evidence_entropy,
    classification_accuracy,
    confidence_drop,
    effective_support,
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

    total = 0
    accuracy_sum = 0.0
    top_drop_sum = 0.0
    random_drop_sum = 0.0
    support_sum = 0.0
    entropy_sum = 0.0

    for images, targets, _ in tqdm(loader):
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        with torch.no_grad():
            outputs = model(
                images,
                return_aux=True,
            )

            logits = outputs["logits"]
            evidence = outputs[
                "resolved_evidence"
            ]

            top_mask = (
                make_top_deletion_mask(
                    evidence,
                    ratio=cfg["loss"][
                        "mask_ratio"
                    ],
                )
            )

            random_mask = (
                make_random_deletion_mask(
                    evidence,
                    ratio=cfg["loss"][
                        "mask_ratio"
                    ],
                )
            )

            top_logits = model.forward_masked(
                images,
                top_mask,
            )

            random_logits = (
                model.forward_masked(
                    images,
                    random_mask,
                )
            )

        batch_size = images.size(0)
        total += batch_size

        accuracy_sum += (
            classification_accuracy(
                logits,
                targets,
            )
            * batch_size
        )

        top_drop_sum += (
            confidence_drop(
                logits,
                top_logits,
                targets,
            )
            * batch_size
        )

        random_drop_sum += (
            confidence_drop(
                logits,
                random_logits,
                targets,
            )
            * batch_size
        )

        support_sum += (
            effective_support(evidence)
            * batch_size
        )

        entropy_sum += (
            evidence_entropy(evidence)
            * batch_size
        )

    print({
        "num_samples": total,
        "accuracy": accuracy_sum / total,
        "top_deletion_confidence_drop": (
            top_drop_sum / total
        ),
        "random_deletion_confidence_drop": (
            random_drop_sum / total
        ),
        "top_minus_random_drop": (
            top_drop_sum / total
            - random_drop_sum / total
        ),
        "effective_support": (
            support_sum / total
        ),
        "evidence_entropy": (
            entropy_sum / total
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
        "--max_samples",
        type=int,
        default=None,
    )

    args = parser.parse_args()
    evaluate(args)

