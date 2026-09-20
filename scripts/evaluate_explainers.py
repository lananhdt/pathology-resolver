import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import read_split_csv, HistologyDataset
from src.losses import make_random_deletion_mask, make_top_deletion_mask
from src.metrics import confidence_drop
from src.model import PathologyResolver
from src.utils import load_checkpoint, load_config
from baselines.gradcam_baseline import make_gradcam
from baselines.ig_baseline import make_ig
from src.counterfactual import choose_counterfactual_class, contrastive_evidence

def target_confidence(logits, targets):
    probabilities = torch.softmax(logits, dim=1)
    return probabilities.gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()


def normalize_evidence(evidence):
    evidence = evidence.clamp(min=0.0)
    return evidence / (evidence.sum(dim=1, keepdim=True) + 1e-8)


def random_drop_mean(model, images, original_logits, evidence, targets, ratio, num_trials=10):
    drop_values = []
    conf_values = []
    
    for _ in range(num_trials):
        random_mask = make_random_deletion_mask(evidence, ratio=ratio)
        random_logits = model.forward_with_external_mask(images, evidence, random_mask)
        
        drop_values.append(confidence_drop(original_logits, random_logits, targets))
        conf_values.append(target_confidence(random_logits, targets))
        
    return sum(drop_values) / len(drop_values), sum(conf_values) / len(conf_values)


def evaluate_evidence(model, loader, evidence_name, device, ratio, ig_steps=16):
    top_drop_sum = 0.0
    random_drop_sum = 0.0
    original_conf_sum = 0.0
    top_conf_sum = 0.0
    random_conf_sum = 0.0
    total = 0

    for images, targets, _ in tqdm(loader, desc=evidence_name):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if evidence_name == "pathresolve":
            with torch.no_grad():
                outputs = model.forward_two_pass(imgs)
                evidence = outputs["resolved_evidence"]
                original_logits = outputs["logits"]
          
        elif method_name == "contrastive":
                with torch.no_grad():
                    tokens = model.extract_features(imgs)
                    first_pass_logits = model(imgs, return_aux=False, targets=None)
                    pred_class = first_pass_logits.argmax(dim=1)
                    counter_class = choose_counterfactual_class(first_pass_logits)
                    target_ev = model.resolve_evidence(tokens, targets=pred_class)["resolved_evidence"]
                    counter_ev = model.resolve_evidence(tokens, targets=counter_class)["resolved_evidence"]
                    return contrastive_evidence(target_ev, counter_ev)
                    
        elif evidence_name == "average":
            with torch.no_grad():
                original_logits, evidence = model.forward_average(images)
                
        elif evidence_name == "gradcam":
            with torch.enable_grad():
                evidence = make_gradcam(model, images, targets)
            evidence = evidence.detach()
            with torch.no_grad():
                original_logits = model.forward_with_external_evidence(images, evidence)
                
        elif evidence_name == "ig":
            with torch.enable_grad():
                evidence = make_ig(model, images, targets, steps=ig_steps)
            evidence = evidence.detach()
            with torch.no_grad():
                original_logits = model.forward_with_external_evidence(images, evidence)
                
        else:
            raise ValueError(f"Unknown evidence: {evidence_name}")

        with torch.no_grad():
            evidence = normalize_evidence(evidence)
            top_mask = make_top_deletion_mask(evidence, ratio=ratio)
            top_logits = model.forward_with_external_mask(images, evidence, top_mask)
            
            random_drop, random_conf = random_drop_mean(
                model=model, images=images, original_logits=original_logits,
                evidence=evidence, targets=targets, ratio=ratio, num_trials=10
            )
            top_drop = confidence_drop(original_logits, top_logits, targets)

        batch_size = images.size(0)
        
        top_drop_sum += (top_drop * batch_size)
        random_drop_sum += (random_drop * batch_size)
        
        original_conf_sum += (target_confidence(original_logits, targets) * batch_size)
        top_conf_sum += (target_confidence(top_logits, targets) * batch_size)
        random_conf_sum += (random_conf * batch_size)
        
        total += batch_size

    return {
        "method": evidence_name,
        "ratio": ratio,
        "top_drop": top_drop_sum / total,
        "random_drop": random_drop_sum / total,
        "gap": (top_drop_sum / total) - (random_drop_sum / total),
        "original_confidence": original_conf_sum / total,
        "top_confidence": top_conf_sum / total,
        "random_confidence": random_conf_sum / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/seed_42.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--ig_steps", type=int, default=16)
    parser.add_argument("--output_name", default="explainer_results.json")
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

    ratios = {"pathresolve": [], "average": [], "gradcam": [], "ig": [], "contrastive": []}

    for method in ratios:
        result = evaluate_evidence(
            model=model, loader=loader, evidence_name=method,
            device=device, ratio=args.ratio, ig_steps=args.ig_steps,
        )
        ratios[method].append(result)
        print(result)

    output_dir = Path(cfg["output"]["result_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    with open(output_path, "w") as file:
        json.dump(ratios, file, indent=2)

    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
