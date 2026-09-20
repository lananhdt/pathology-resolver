import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

def robust_normalize_map(value):
    value = value.astype("float32")
    low = value.min()
    high = value.max()
    if high - low < 1e-8:
        return value * 0.0
    value = (value - low) / (high - low)
    return value

def normalize_map(value):
    value = value.clamp(min=0.0)
    return value / (value.sum(dim=1, keepdim=True) + 1e-8)

def evidence_to_image_map(evidence):
    evidence = evidence.view(-1, 7, 7)
    evidence = F.interpolate(
        evidence.unsqueeze(1),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    return evidence[:, 0]

def save_visualization(image_tensor, sample_maps, path, output_path):
    image = image_tensor.cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    image = image * std + mean
    image = image.clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).numpy()

    names = list(sample_maps.keys())

    figure, axes = plt.subplots(
        2,
        len(names) + 1,
        figsize=(4 * (len(names) + 1), 8),
    )

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original\n" + Path(path).name, fontsize=10)
    axes[0, 0].axis("off")

    axes[1, 0].imshow(image)
    axes[1, 0].set_title("Original\n" + Path(path).name, fontsize=10)
    axes[1, 0].axis("off")

    for index, name in enumerate(names, start=1):
        heatmap = sample_maps[name]
        
        axes[0, index].imshow(
            heatmap,
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
        )
        axes[0, index].set_title(f"{name} map")
        axes[0, index].axis("off")
        
        axes[1, index].imshow(image)
        axes[1, index].imshow(
            heatmap,
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
            alpha=0.45,
        )
        axes[1, index].set_title(f"{name} overlay")
        axes[1, index].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/seed_42.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output_dir", default="outputs/visuals")
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--ig_steps", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HistologyDataset(
        read_split_csv(args.split),
        image_size=224,
        train=False,
    )

    loader = DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=2,
    )

    model = PathologyResolver(
        num_classes=9, pretrained=False, feature_dim=512, dropout=0.2, constrained=True,
    ).to(device)

    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for images, targets, paths in loader:
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            output = model.forward_two_pass(images)
            pathresolve_map = output["resolved_evidence"]

            _, average_map = model.forward_average(images)

        with torch.enable_grad():
            gradcam_map = make_gradcam(model, images, targets)
            ig_map = make_ig(model, images, targets, steps=args.ig_steps)

        pathresolve_map = normalize_map(pathresolve_map)
        average_map = normalize_map(average_map)
        gradcam_map = normalize_map(gradcam_map.detach())
        ig_map = normalize_map(ig_map.detach())

        maps = {
            "PathResolve": evidence_to_image_map(pathresolve_map),
            "Average": evidence_to_image_map(average_map),
            "Grad-CAM": evidence_to_image_map(gradcam_map),
            "IG": evidence_to_image_map(ig_map),
        }

        for index, path in enumerate(paths):
            if saved >= args.num_images:
                break

            sample_maps = {
                name: robust_normalize_map(value[index].cpu().numpy())
                for name, value in maps.items()
            }

            output_path = output_dir / f"sample_{saved:04d}.png"
            save_visualization(images[index], sample_maps, path, output_path)
            saved += 1

        if saved >= args.num_images:
            break

    print(f"Saved {saved} visualizations to {output_dir}")

if __name__ == "__main__":
    main()
