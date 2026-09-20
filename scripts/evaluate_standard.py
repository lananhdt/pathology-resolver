import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import (
    read_split_csv,
    HistologyDataset,
)
from src.model import (
    StandardPathologyClassifier,
)
from src.utils import load_config


def load_standard_checkpoint(
    path,
    model,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    return checkpoint


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
        "--output_name",
        default=None,
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
        batch_size=cfg["training"][
            "batch_size"
        ],
        shuffle=False,
        num_workers=cfg["data"][
            "num_workers"
        ],
        pin_memory=True,
    )

    model = StandardPathologyClassifier(
        num_classes=cfg["model"][
            "num_classes"
        ],
        pretrained=False,
        feature_dim=cfg["model"][
            "feature_dim"
        ],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    load_standard_checkpoint(
        args.checkpoint,
        model,
        device,
    )

    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, targets, _ in tqdm(
            loader,
            desc="standard",
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            predictions = logits.argmax(
                dim=1
            )

            all_targets.extend(
                targets.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

    accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            all_targets,
            all_predictions,
        )
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
    )

    report = classification_report(
        all_targets,
        all_predictions,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_targets,
        all_predictions,
    )

    result = {
        "accuracy": accuracy,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "num_samples": len(all_targets),
    }

    print(json.dumps(
        result,
        indent=2,
    ))

    if args.output_name:
        output_dir = "outputs"
        output_path = (
            f"{output_dir}/"
            f"{args.output_name}"
        )

        with open(
            output_path,
            "w",
        ) as file:
            json.dump(
                result,
                file,
                indent=2,
            )

        print(
            f"Saved to {output_path}"
        )


if __name__ == "__main__":
    main()
