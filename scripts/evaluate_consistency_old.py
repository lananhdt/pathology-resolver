import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from src.data import read_split_csv, HistologyDataset
from src.model import PathologyResolver
from src.utils import load_checkpoint, load_config
from src.consistency import map_cosine_similarity, rank_overlap
from baselines.gradcam_baseline import make_gradcam
from baselines.ig_baseline import make_ig

def evaluate_method_consistency(model, loader, method_name, device, ratio=0.2, ig_steps=16):
    color_transform = transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04
    )
    
    cosine_sum = 0.0
    overlap_sum = 0.0
    total_batches = 0
    
    for images, targets, _ in tqdm(loader, desc=f"Consistency {method_name}"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        augmented_images = color_transform(images)
        
        def get_evidence(imgs):
            if method_name == "pathresolve":
                with torch.no_grad():
                    out = model(imgs, return_aux=True)
                    return out["resolved_evidence"]
            elif method_name == "average":
                with torch.no_grad():
                    _, ev = model.forward_average(imgs)
                    return ev
            elif method_name == "gradcam":
                with torch.enable_grad():
                    ev = make_gradcam(model, imgs, targets)
                return ev.detach()
            elif method_name == "ig":
                with torch.enable_grad():
                    ev = make_ig(model, imgs, targets, steps=ig_steps)
                return ev.detach()
            else:
                raise ValueError(f"Unknown method: {method_name}")

        ev_original = get_evidence(images)
        ev_augmented = get_evidence(augmented_images)
        
        cosine_sum += map_cosine_similarity(ev_original, ev_augmented)
        overlap_sum += rank_overlap(ev_original, ev_augmented, ratio=ratio)
        total_batches += 1
        
    return {
        "method": method_name,
        "cosine_consistency": cosine_sum / total_batches,
        "top_k_overlap": overlap_sum / total_batches
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/seed_42.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--ig_steps", type=int, default=16)
    parser.add_argument("--output_name", default="consistency_results.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HistologyDataset(
        read_split_csv(args.split),
        image_size=cfg["data"]["image_size"],
        train=False,
    )

    loader = DataLoader(
        dataset, batch_size=16, shuffle=False,
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

    methods = ["pathresolve", "average", "gradcam", "ig"]
    results = {}

    for method in methods:
        res = evaluate_method_consistency(
            model=model, loader=loader, method_name=method,
            device=device, ratio=args.ratio, ig_steps=args.ig_steps
        )
        results[method] = res
        print(f"\n{method.upper()} Consistency:")
        print(f" - Cosine:  {res['cosine_consistency']:.4f}")
        print(f" - Overlap: {res['top_k_overlap']:.4f}")

    output_dir = Path(cfg["output"]["result_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    with open(output_path, "w") as file:
        json.dump(results, file, indent=2)

    print(f"\nSaved consistency results to {output_path}")

if __name__ == "__main__":
    main()
