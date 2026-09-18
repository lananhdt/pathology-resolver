import torch


@torch.no_grad()
def classification_accuracy(logits, targets):
    predictions = logits.argmax(dim=1)

    return (
        predictions == targets
    ).float().mean().item()


@torch.no_grad()
def confidence_drop(
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

    return (
        original_probability
        - masked_probability
    ).mean().item()


@torch.no_grad()
def evidence_entropy(evidence):
    probabilities = torch.clamp(
        evidence,
        min=1e-8,
    )

    entropy = -(
        probabilities
        * torch.log(probabilities)
    ).sum(dim=1)

    return entropy.mean().item()


@torch.no_grad()
def effective_support(
    evidence,
    threshold=0.1,
):
    minimum = evidence.min(
        dim=1,
        keepdim=True,
    ).values

    maximum = evidence.max(
        dim=1,
        keepdim=True,
    ).values

    normalized = (
        evidence - minimum
    ) / (
        maximum - minimum + 1e-8
    )

    support = (
        normalized > threshold
    ).float().mean(dim=1)

    return support.mean().item()


@torch.no_grad()
def topk_mass(
    evidence,
    ratio=0.10,
):
    num_tokens = evidence.size(1)

    k = max(
        1,
        int(num_tokens * ratio),
    )

    values = torch.topk(
        evidence,
        k=k,
        dim=1,
        largest=True,
        sorted=True,
    ).values

    return values.sum(dim=1).mean().item()


@torch.no_grad()
def topk_indices(
    evidence,
    ratio=0.10,
):
    num_tokens = evidence.size(1)

    k = max(
        1,
        int(num_tokens * ratio),
    )

    return torch.topk(
        evidence,
        k=k,
        dim=1,
        largest=True,
        sorted=True,
    ).indices
