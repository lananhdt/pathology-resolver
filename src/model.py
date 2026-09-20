import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class StandardPathologyClassifier(
    nn.Module
):
    def __init__(
        self,
        num_classes=9,
        pretrained=True,
        feature_dim=512,
        dropout=0.2,
    ):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        backbone = resnet18(
            weights=weights
        )

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.token_norm = nn.LayerNorm(
            feature_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                feature_dim,
                256,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(
                256,
                num_classes,
            ),
        )

    def extract_features(self, x):
        feature_map = self.encoder(x)

        tokens = feature_map.flatten(2)
        tokens = tokens.transpose(1, 2)

        return self.token_norm(tokens)

    def forward(self, x):
        tokens = self.extract_features(x)
        pooled = tokens.mean(dim=1)
        return self.classifier(pooled)

class EvidenceMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class EvidenceResolver(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim=256,
    ):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(
                feature_dim + 2,
                hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,
                2,
            ),
        )

    def forward(
        self,
        features,
        class_evidence,
        morph_evidence,
    ):
        gate_input = torch.cat(
            [
                features,
                class_evidence.unsqueeze(-1),
                morph_evidence.unsqueeze(-1),
            ],
            dim=-1,
        )

        branch_weights = torch.softmax(
            self.gate(gate_input),
            dim=-1,
        )

        branch_evidence = torch.stack(
            [
                class_evidence,
                morph_evidence,
            ],
            dim=-1,
        )

        resolved_evidence = (
            branch_weights
            * branch_evidence
        ).sum(dim=-1)

        resolved_evidence = torch.softmax(
            resolved_evidence,
            dim=-1,
        )

        disagreement = torch.abs(
            class_evidence
            - morph_evidence
        )

        return (
            resolved_evidence,
            branch_weights,
            disagreement,
        )


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

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        backbone = resnet18(
            weights=weights
        )

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.token_norm = nn.LayerNorm(
            feature_dim
        )

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.constrained = constrained

        self.class_branch = EvidenceMLP(
            input_dim=feature_dim,
            hidden_dim=256,
        )

        self.morph_branch = EvidenceMLP(
            input_dim=feature_dim * 2,
            hidden_dim=256,
        )

        self.resolver = EvidenceResolver(
            feature_dim=feature_dim,
            hidden_dim=256,
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                feature_dim,
                256,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(
                256,
                num_classes,
            ),
        )

        self.shortcut_classifier = nn.Sequential(
            nn.Linear(
                feature_dim,
                256,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(
                256,
                num_classes,
            ),
        )
        
        self.class_embedding = nn.Embedding(
            num_classes,
            feature_dim,
        )

    def extract_features(self, x):
        feature_map = self.encoder(x)

        tokens = feature_map.flatten(2)
        tokens = tokens.transpose(1, 2)

        tokens = self.token_norm(tokens)

        return tokens

        def resolve_evidence(
            self,
            tokens,
            targets=None,
        ):
            class_evidence = self.class_branch(
                tokens
            )

            if targets is None:
                class_ids = torch.zeros(
                    tokens.size(0),
                    dtype=torch.long,
                    device=tokens.device,
                )
            else:
                class_ids = targets

            class_embedding = (
                self.class_embedding(class_ids)
            )

            class_embedding = (
                class_embedding.unsqueeze(1)
                .expand(
                    -1,
                    tokens.size(1),
                    -1,
                )
            )

            morphology_input = torch.cat(
                [
                    tokens,
                    class_embedding,
                ],
                dim=-1,
            )

            morph_evidence = self.morph_branch(
                morphology_input
            )

            (
                resolved_evidence,
                branch_weights,
                disagreement,
            ) = self.resolver(
                tokens,
                class_evidence,
                morph_evidence,
            )

            return {
                "class_evidence": class_evidence,
                "morph_evidence": morph_evidence,
                "resolved_evidence": (
                    resolved_evidence
                ),
                "branch_weights": branch_weights,
                "disagreement": disagreement,
            }

    def pool_with_evidence(
        self,
        tokens,
        evidence,
    ):
        pooled = (
            evidence.unsqueeze(-1)
            * tokens
        ).sum(dim=1)

        return pooled

    def forward(
        self,
        x,
        return_aux=False,
    ):
        tokens = self.extract_features(x)

        evidence_outputs = (
            self.resolve_evidence(tokens)
        )

        resolved_evidence = (
            evidence_outputs[
                "resolved_evidence"
            ]
        )

        pooled = self.pool_with_evidence(
            tokens,
            resolved_evidence,
        )

        if self.constrained:
            logits = self.classifier(
                pooled
            )
        else:
            shortcut_pooled = tokens.mean(
                dim=1
            )

            logits = self.shortcut_classifier(
                shortcut_pooled
            )

        if not return_aux:
            return logits

        return {
            "logits": logits,
            "features": tokens,
            **evidence_outputs,
        }
    
    def forward_standard(self, x):
        tokens = self.extract_features(x)
        pooled = tokens.mean(dim=1)
        logits = self.shortcut_classifier(pooled)
        return logits, None

    def forward_uniform(self, x):
        tokens = self.extract_features(x)

        evidence = torch.ones(
            tokens.size(
                0
            ),
            tokens.size(1),
            device=tokens.device,
        )

        evidence = evidence / tokens.size(1)

        pooled = (
            evidence.unsqueeze(-1)
            * tokens
        ).sum(dim=1)

        logits = self.classifier(pooled)

        return logits, evidence
        
    def forward_average(self, x):
        tokens = self.extract_features(x)

        class_evidence = self.class_branch(
            tokens
        )

        morph_evidence = self.morph_branch(
            tokens
        )

        evidence = torch.softmax(
            0.5 * class_evidence
            + 0.5 * morph_evidence,
            dim=1,
        )

        pooled = (
            evidence.unsqueeze(-1)
            * tokens
        ).sum(dim=1)

        logits = self.classifier(
            pooled
        )

        return logits, evidence

    def forward_with_external_evidence(
        self,
        x,
        evidence,
    ):
        tokens = self.extract_features(x)

        evidence = evidence / (
            evidence.sum(
                dim=1,
                keepdim=True,
            )
            + 1e-8
        )

        pooled = (
            evidence.unsqueeze(-1)
            * tokens
        ).sum(dim=1)

        return self.classifier(
            pooled
        )

    def forward_with_external_mask(
        self,
        x,
        evidence,
        mask,
    ):
        tokens = self.extract_features(x)

        masked_evidence = evidence * mask

        masked_evidence = (
            masked_evidence
            / (
                masked_evidence.sum(
                    dim=1,
                    keepdim=True,
                )
                + 1e-8
            )
        )

        pooled = (
            masked_evidence.unsqueeze(-1)
            * tokens
        ).sum(dim=1)

        return self.classifier(
            pooled
        )

    def forward_masked(
        self,
        x,
        mask,
    ):
        tokens = self.extract_features(x)

        evidence_outputs = (
            self.resolve_evidence(tokens)
        )

        evidence = evidence_outputs[
            "resolved_evidence"
        ]

        return self.forward_masked_with_evidence(
            x=x,
            mask=mask,
            evidence=evidence,
            tokens=tokens,
        )

    def forward_masked_with_evidence(
        self,
        x,
        mask,
        evidence,
        tokens=None,
    ):
        if tokens is None:
            tokens = self.extract_features(x)

        masked_evidence = evidence * mask

        masked_evidence = (
            masked_evidence
            / (
                masked_evidence.sum(
                    dim=1,
                    keepdim=True,
                )
                + 1e-8
            )
        )

        pooled = (
            masked_evidence.unsqueeze(-1)
            * tokens
        ).sum(dim=1)

        return self.classifier(
            pooled
        )
        
    def forward_with_external_keep(
        self,
        x,
        evidence,
        mask,
    ):
        tokens = self.extract_features(x)

        kept_evidence = evidence * mask

        kept_evidence = (
            kept_evidence
            / (
                kept_evidence.sum(
                    dim=1,
                    keepdim=True,
                )
                + 1e-8
            )
        )

        pooled = (
            kept_evidence.unsqueeze(-1)
            * tokens
        ).sum(dim=1)

        return self.classifier(
            pooled
        )
    
    def get_class_conditioned_evidence(
        self,
        tokens,
        class_ids,
    ):
        class_embedding = (
            self.class_embedding(class_ids)
        )

        class_embedding = (
            class_embedding.unsqueeze(1)
            .expand(
                -1,
                tokens.size(1),
                -1,
            )
        )

        morphology_input = torch.cat(
            [
                tokens,
                class_embedding,
            ],
            dim=-1,
        )

        class_evidence = self.class_branch(
            tokens
        )

        morph_evidence = self.morph_branch(
            morphology_input
        )

        resolved_evidence, weights, disagreement = (
            self.resolver(
                tokens,
                class_evidence,
                morph_evidence,
            )
        )

        return {
            "class_evidence": class_evidence,
            "morph_evidence": morph_evidence,
            "resolved_evidence": (
                resolved_evidence
            ),
            "branch_weights": weights,
            "disagreement": disagreement,
        }
    
class StandardPathologyClassifier(nn.Module):
    def __init__(
        self,
        num_classes=9,
        pretrained=True,
        feature_dim=512,
        dropout=0.2,
    ):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        backbone = resnet18(
            weights=weights
        )

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.token_norm = nn.LayerNorm(
            feature_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                feature_dim,
                256,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(
                256,
                num_classes,
            ),
        )

    def extract_features(self, x):
        feature_map = self.encoder(x)

        tokens = feature_map.flatten(2)
        tokens = tokens.transpose(1, 2)

        tokens = self.token_norm(tokens)

        return tokens

    def forward(self, x):
        tokens = self.extract_features(x)

        pooled = tokens.mean(dim=1)

        return self.classifier(
            pooled
        )
