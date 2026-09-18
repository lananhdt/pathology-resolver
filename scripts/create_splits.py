import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import random
from pathlib import Path

from src.data import CLASS_NAMES


def collect_class_files(
    root,
    max_samples_per_class=None,
):
    root = Path(root)
    rows = []

    for label, class_name in enumerate(
        CLASS_NAMES
    ):
        class_dir = root / class_name

        files = sorted(
            list(class_dir.glob("*.tif"))
            + list(class_dir.glob("*.tiff"))
            + list(class_dir.glob("*.png"))
            + list(class_dir.glob("*.jpg"))
        )

        rng = random.Random(42 + label)
        rng.shuffle(files)

        if max_samples_per_class:
            files = files[
                :max_samples_per_class
            ]

        for path in files:
            rows.append({
                "path": str(path),
                "label": label,
                "class_name": class_name,
            })

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        default="data/splits",
    )

    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
    )

    args = parser.parse_args()

    rows = collect_class_files(
        args.root,
        args.max_samples_per_class,
    )

    train_rows = []
    val_rows = []
    test_rows = []

    by_label = {}

    for row in rows:
        by_label.setdefault(
            row["label"],
            [],
        ).append(row)

    for label, class_rows in by_label.items():
        rng = random.Random(1000 + label)
        rng.shuffle(class_rows)

        n = len(class_rows)
        n_test = int(n * args.test_ratio)
        n_val = int(n * args.val_ratio)

        test_rows.extend(
            class_rows[:n_test]
        )

        val_rows.extend(
            class_rows[
                n_test:n_test + n_val
            ]
        )

        train_rows.extend(
            class_rows[
                n_test + n_val:
            ]
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, split_rows in [
        ("train.csv", train_rows),
        ("val.csv", val_rows),
        ("test.csv", test_rows),
    ]:
        output_path = output_dir / name

        with open(
            output_path,
            "w",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "path",
                    "label",
                    "class_name",
                ],
            )

            writer.writeheader()
            writer.writerows(split_rows)

        print(
            name,
            len(split_rows),
            output_path,
        )


if __name__ == "__main__":
    main()
