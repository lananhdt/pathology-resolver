import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

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
    evidence_entropy,
    effective_support,
    topk_mass,
)
from src.model import PathologyResolver
from src.utils import (
    load_checkpoint,
    load_config,
)


def evaluate_ratio(
    model,
    loader,
    ratio,
    device,
):
    accuracy_sum = 0.0
    top_drop_sum = 0.0
    random_drop_sum = 0.0
    support_sum = 0.0
    entropy_sum = 0.0
    top_mass_sum = 0.0
    total = 0

    for images, targets, _ in tqdm(
        loader,
        desc=f"ratio={ratio:.2f}",
        leave=False,
    ):
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
                    ratio=ratio,
                )
            )

            random_mask = (
                make_random_deletion_mask(
                    evidence,
                    ratio=ratio,
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

        top_mass_sum += (
            topk_mass(
                evidence,
                ratio=ratio,
            )
            * batch_size
        )

    top_drop = top_drop_sum / total
    random_drop = random_drop_sum / total

    return {
        "ratio": ratio,
        "num_deleted": None,
        "accuracy": accuracy_sum / total,
        "top_deletion_drop": top_drop,
        "random_deletion_drop": random_drop,
        "top_minus_random": (
            top_drop - random_drop
        ),
        "effective_support": (
            support_sum / total
        ),
        "evidence_entropy": (
            entropy_sum / total
        ),
        "top_mass": (
            top_mass_sum / total
        ),
    }


def trapezoid_auc(x_values, y_values):
    x = torch.tensor(
        x_values,
        dtype=torch.float64,
    )

    y = torch.tensor(
        y_values,
        dtype=torch.float64,
    )

    return torch.trapezoid(y, x).item()


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

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )

    args = parser.parse_args()

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

    batch_size = (
        args.batch_size
        or cfg["training"]["batch_size"]
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
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

    ratios = [
        1 / 49,
        2 / 49,
        5 / 49,
        10 / 49,
        15 / 49,
        25 / 49,
    ]

    results = []

    for ratio in ratios:
        result = evaluate_ratio(
            model=model,
            loader=loader,
            ratio=ratio,
            device=device,
        )

        result["num_deleted"] = max(
            1,
            int(49 * ratio),
        )

        results.append(result)

        print(result)

    x_values = [
        item["ratio"]
        for item in results
    ]

    top_values = [
        item["top_deletion_drop"]
        for item in results
    ]

    random_values = [
        item["random_deletion_drop"]
        for item in results
    ]

    gaps = [
        item["top_minus_random"]
        for item in results
    ]

    output_dir = Path(
        cfg["output"]["result_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "deletion_curve.json"
    )

    payload = {
        "results": results,
        "top_drop_auc": trapezoid_auc(
            x_values,
            top_values,
        ),
        "random_drop_auc": trapezoid_auc(
            x_values,
            random_values,
        ),
        "gap_auc": trapezoid_auc(
            x_values,
            gaps,
        ),
    }

    with open(output_path, "w") as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    print("\nAUC summary:")
    print({
        "top_drop_auc": payload[
            "top_drop_auc"
        ],
        "random_drop_auc": payload[
            "random_drop_auc"
        ],
        "gap_auc": payload[
            "gap_auc"
        ],
    })

    print(
        f"\nSaved results to {output_path}"
    )


if __name__ == "__main__":
    main()
