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
    HistologyDataset,
    read_split_csv,
)
from src.losses import (
    make_top_keep_mask,
)
from src.metrics import (
    classification_accuracy,
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


def normalize_evidence(evidence):
    evidence = evidence.clamp(
        min=0.0
    )

    return evidence / (
        evidence.sum(
            dim=1,
            keepdim=True,
        )
        + 1e-8
    )


def target_confidence(
    logits,
    targets,
):
    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    return probabilities.gather(
        1,
        targets.unsqueeze(1),
    ).squeeze(1).mean().item()


def get_evidence(
    model,
    images,
    targets,
    method,
    ig_steps,
):
    if method == "pathresolve":
        with torch.no_grad():
            output = model(
                images,
                return_aux=True,
            )

        return output[
            "resolved_evidence"
        ]

    if method == "average":
        with torch.no_grad():
            _, evidence = (
                model.forward_average(
                    images
                )
            )

        return evidence

    if method == "gradcam":
        with torch.enable_grad():
            evidence = make_gradcam(
                model,
                images,
                targets,
            )

        return evidence.detach()

    if method == "ig":
        with torch.enable_grad():
            evidence = make_ig(
                model,
                images,
                targets,
                steps=ig_steps,
            )

        return evidence.detach()

    raise ValueError(method)


def evaluate_method(
    model,
    loader,
    method,
    device,
    ratio,
    ig_steps,
):
    accuracy_sum = 0.0
    confidence_sum = 0.0
    total = 0

    for images, targets, _ in tqdm(
        loader,
        desc=method,
    ):
        images = images.to(device)
        targets = targets.to(device)

        if method == "pathresolve":
            with torch.no_grad():
                original_output = model(
                    images,
                    return_aux=True,
                )

            original_logits = (
                original_output["logits"]
            )

        elif method == "average":
            with torch.no_grad():
                original_logits, _ = (
                    model.forward_average(
                        images
                    )
                )

        else:
            with torch.no_grad():
                original_logits = model(
                    images
                )

        evidence = get_evidence(
            model,
            images,
            targets,
            method,
            ig_steps,
        )

        evidence = normalize_evidence(
            evidence
        )

        keep_mask = make_top_keep_mask(
            evidence,
            ratio=ratio,
        )

        with torch.no_grad():
            kept_logits = (
                model.forward_with_external_keep(
                    images,
                    evidence,
                    keep_mask,
                )
            )

        batch_size = images.size(0)

        accuracy_sum += (
            classification_accuracy(
                kept_logits,
                targets,
            )
            * batch_size
        )

        confidence_sum += (
            target_confidence(
                kept_logits,
                targets,
            )
            * batch_size
        )

        total += batch_size

    return {
        "method": method,
        "ratio": ratio,
        "top_keep_accuracy": (
            accuracy_sum / total
        ),
        "top_keep_confidence": (
            confidence_sum / total
        ),
    }


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
        "--ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--ig_steps",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--output_name",
        default="sufficiency.json",
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

    results = {}

    for method in [
        "pathresolve",
        "average",
        "gradcam",
        "ig",
    ]:
        result = evaluate_method(
            model,
            loader,
            method,
            device,
            args.ratio,
            args.ig_steps,
        )

        results[method] = result
        print(result)

    output_dir = Path(
        cfg["output"]["result_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / args.output_name
    )

    with open(output_path, "w") as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        f"Saved to {output_path}"
    )


if __name__ == "__main__":
    main()
