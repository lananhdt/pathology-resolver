import torch
import torch.nn.functional as F


def make_soft_mask(evidence, mask_ratio=0.2, noise=0.05):
    evidence = evidence.detach()

    threshold = torch.quantile(
        evidence,
        q=mask_ratio,
        dim=0,
        keepdim=True
    )

    mask = (evidence > threshold).float()
    mask = mask + noise * torch.rand_like(mask)
    mask = mask.clamp(0.0, 1.0)

    return mask
    

def classification_loss(logits, targets):
    return F.cross_entropy(logits, targets)


def faithfulness_proxy_loss(
    model,
    images,
    targets,
    evidence,
    mask_ratio=0.2,
):
    mask = make_soft_mask(
        evidence,
        mask_ratio=mask_ratio
    )

    masked_logits = model.forward_masked(images, mask)

    original_loss = F.cross_entropy(
        model(images),
        targets
    )

    masked_loss = F.cross_entropy(
        masked_logits,
        targets
    )

    return torch.relu(original_loss - masked_loss)


def consistency_loss(original_evidence, augmented_evidence):
    return F.mse_loss(
        original_evidence,
        augmented_evidence
    )


def sparsity_loss(evidence):
    return evidence.abs().mean()


def total_loss(
    original_logits,
    targets,
    original_evidence,
    augmented_evidence,
    masked_logits=None,
    faithfulness_weight=0.2,
    consistency_weight=0.2,
    sparsity_weight=0.0001,
):
    loss_cls = classification_loss(
        original_logits,
        targets
    )

    loss_cons = consistency_loss(
        original_evidence,
        augmented_evidence
    )

    loss_sparse = sparsity_loss(
        original_evidence
    )

    loss_faith = torch.tensor(
        0.0,
        device=original_logits.device
    )

    if masked_logits is not None:
        loss_original = F.cross_entropy(
            original_logits,
            targets
        )
        loss_masked = F.cross_entropy(
            masked_logits,
            targets
        )
        loss_faith = torch.relu(
            loss_original - loss_masked
        )

    loss = (
        loss_cls
        + faithfulness_weight * loss_faith
        + consistency_weight * loss_cons
        + sparsity_weight * loss_sparse
    )

    values = {
        "total": loss.detach().item(),
        "classification": loss_cls.detach().item(),
        "faithfulness": loss_faith.detach().item(),
        "consistency": loss_cons.detach().item(),
        "sparsity": loss_sparse.detach().item(),
    }

    return loss, values

