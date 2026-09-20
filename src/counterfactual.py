import torch


def choose_counterfactual_class(
    logits,
):
    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    ranked = torch.argsort(
        probabilities,
        dim=1,
        descending=True,
    )

    return ranked[:, 1]


def make_contrastive_evidence(
    target_evidence,
    counterfactual_evidence,
):
    contrastive = (
        target_evidence
        - counterfactual_evidence
    )

    contrastive = contrastive.clamp(
        min=0.0
    )

    return contrastive / (
        contrastive.sum(
            dim=1,
            keepdim=True,
        )
        + 1e-8
    )
