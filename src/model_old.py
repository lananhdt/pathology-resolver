import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class MLPScore(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class EvidenceResolver(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(feature_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features, class_evidence, morph_evidence):
        disagreement = torch.abs(class_evidence - morph_evidence)
        gate_input = torch.cat([
            features,
            class_evidence.unsqueeze(-1),
            morph_evidence.unsqueeze(-1),
        ], dim=-1)

        weights = torch.softmax(self.gate(gate_input), dim=-1)

        resolved = (
            weights[..., 0] * class_evidence +
            weights[..., 1] * morph_evidence
        )

        resolved = torch.softmax(resolved, dim=-1)

        return resolved, weights, disagreement


class PathologyResolver(nn.Module):
    def __init__(
        self,
        num_classes=9,
        pretrained=True,
        feature_dim=512,
        dropout=0.2,
        constrained=True,
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.constrained = constrained

        self.class_branch = MLPScore(feature_dim)
        self.morph_branch = MLPScore(feature_dim)

        self.resolver = EvidenceResolver(feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.shortcut_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def extract_features(self, x):
        features = self.encoder(x)
        return features.flatten(1)

    def forward(self, x, return_aux=False):
        features = self.extract_features(x)

        class_logits = self.class_branch(features)
        morph_logits = self.morph_branch(features)

        evidence = torch.stack([
            class_logits,
            morph_logits,
        ], dim=-1)

        branch_weights = torch.softmax(evidence, dim=-1)

        resolved_logits = (
            branch_weights[..., 0] * class_logits +
            branch_weights[..., 1] * morph_logits
        )

        resolved_evidence = torch.sigmoid(resolved_logits)

        if self.constrained:
            pooled = features * resolved_evidence.unsqueeze(-1)
            logits = self.classifier(pooled)
        else:
            logits = self.shortcut_classifier(features)

        if not return_aux:
            return logits

        return {
            "logits": logits,
            "features": features,
            "class_evidence": torch.sigmoid(class_logits),
            "morph_evidence": torch.sigmoid(morph_logits),
            "resolved_evidence": resolved_evidence,
            "branch_weights": branch_weights,
        }

    def forward_masked(self, x, mask):
        features = self.extract_features(x)

        class_evidence = torch.sigmoid(
            self.class_branch(features)
        )

        morph_evidence = torch.sigmoid(
            self.morph_branch(features)
        )

        resolved = (
            0.5 * class_evidence +
            0.5 * morph_evidence
        )

        resolved = resolved * mask
        logits = self.classifier(features * resolved.unsqueeze(-1))

        return logits

