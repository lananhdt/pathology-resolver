import torch
import torch.nn.functional as F


def make_top_deletion_mask(
    evidence,
    ratio=0.2,
):
    num_tokens = evidence.size(1)
    num_delete = max(
        1,
        int(num_tokens * ratio),
    )

    top_indices = torch.topk(
        evidence,
        k=num_delete,
        dim=1,
        largest=True,
    ).indices

    mask = torch.ones_like(evidence)

    mask.scatter_(
        dim=1,
        index=top_indices,
        value=0.0,
    )

    return mask


def make_random_deletion_mask(
    evidence,
    ratio=0.2,
):
    num_tokens = evidence.size(1)
    num_delete = max(
        1,
        int(num_tokens * ratio),
    )

    random_scores = torch.rand_like(evidence)

    random_indices = torch.topk(
        random_scores,
        k=num_delete,
        dim=1,
        largest=True,
    ).indices

    mask = torch.ones_like(evidence)

    mask.scatter_(
        dim=1,
        index=random_indices,
        value=0.0,
    )

    return mask


def classification_loss(logits, targets):
    return F.cross_entropy(
        logits,
        targets,
    )


def consistency_loss(
    original_evidence,
    augmented_evidence,
):
    original_evidence = torch.softmax(
        original_evidence,
        dim=1,
    )

    augmented_evidence = torch.softmax(
        augmented_evidence,
        dim=1,
    )

    return F.mse_loss(
        original_evidence,
        augmented_evidence,
    )


def sparsity_loss(evidence):
    probabilities = torch.softmax(
        evidence,
        dim=1,
    )

    entropy = -(
        probabilities
        * torch.log(probabilities + 1e-8)
    ).sum(dim=1)

    return entropy.mean()


def faithfulness_loss(
    original_logits,
    masked_logits,
    targets,
):
    original_probability = torch.softmax(
        original_logits,
        dim=1,
    ).gather(
        1,
        targets.unsqueeze(1),
    ).squeeze(1)

    masked_probability = torch.softmax(
        masked_logits,
        dim=1,
    ).gather(
        1,
        targets.unsqueeze(1),
    ).squeeze(1)

    confidence_drop = (
        original_probability
        - masked_probability
    )

    return -confidence_drop.mean()


def total_loss(
    original_logits,
    masked_logits,
    targets,
    original_evidence,
    augmented_evidence,
    faithfulness_weight=0.2,
    consistency_weight=0.2,
    sparsity_weight=0.0001,
):
    loss_cls = classification_loss(
        original_logits,
        targets,
    )

    loss_faith = faithfulness_loss(
        original_logits,
        masked_logits,
        targets,
    )

    loss_cons = consistency_loss(
        original_evidence,
        augmented_evidence,
    )

    loss_sparse = sparsity_loss(
        original_evidence,
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

def make_top_keep_mask(
    evidence,
    ratio=0.2,
):
    num_tokens = evidence.size(1)

    num_keep = max(
        1,
        int(num_tokens * ratio),
    )

    top_indices = torch.topk(
        evidence,
        k=num_keep,
        dim=1,
        largest=True,
        sorted=True,
    ).indices

    mask = torch.zeros_like(
        evidence
    )

    mask.scatter_(
        dim=1,
        index=top_indices,
        value=1.0,
    )

    return mask
    
def branch_separation_loss(
    class_evidence,
    morph_evidence,
):
    class_prob = torch.softmax(
        class_evidence,
        dim=1,
    )

    morph_prob = torch.softmax(
        morph_evidence,
        dim=1,
    )

    similarity = (
        class_prob
        * morph_prob
    ).sum(dim=1)

    return similarity.mean()
