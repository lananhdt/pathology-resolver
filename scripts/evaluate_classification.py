import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.data import read_split_csv, HistologyDataset
from src.model import PathologyResolver
from src.utils import load_checkpoint, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/seed_42.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--method", 
        choices=["pathresolve", "average", "uniform", "standard"], 
        default="pathresolve"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HistologyDataset(
        read_split_csv(args.split),
        image_size=cfg["data"]["image_size"],
        train=False,
    )

    loader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    model = PathologyResolver(
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
        feature_dim=cfg["model"]["feature_dim"],
        dropout=cfg["model"]["dropout"],
        constrained=True,
    ).to(device)

    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    all_targets = []
    all_predictions = []

    for images, targets, _ in tqdm(loader, desc=f"Evaluating {args.method}"):
        images = images.to(device, non_blocking=True)
        
        with torch.no_grad():
            if args.method == "pathresolve":
                outputs = model(images, return_aux=True)
                logits = outputs["logits"]
            elif args.method == "average":
                logits, _ = model.forward_average(images)
            elif args.method == "uniform":
                logits, _ = model.forward_uniform(images)
            elif args.method == "standard":
                logits, _ = model.forward_standard(images)
                
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(logits.argmax(dim=1).cpu().tolist())

    print("\n" + "="*50)
    print(f"RESULTS FOR: {args.method.upper()} (Config: {args.config})")
    print("="*50)
    
    print(f"Accuracy:          {accuracy_score(all_targets, all_predictions):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(all_targets, all_predictions):.4f}")
    print(f"Macro-F1:          {f1_score(all_targets, all_predictions, average='macro'):.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(all_targets, all_predictions, digits=4))
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(all_targets, all_predictions))
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
