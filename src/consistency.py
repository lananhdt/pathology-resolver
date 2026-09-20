import torch
import torch.nn.functional as F


@torch.no_grad()
def map_cosine_similarity(
    map_a,
    map_b,
):
    map_a = F.normalize(
        map_a,
        dim=1,
    )

    map_b = F.normalize(
        map_b,
        dim=1,
    )

    return (
        map_a * map_b
    ).sum(dim=1).mean().item()


@torch.no_grad()
def rank_overlap(
    map_a,
    map_b,
    ratio=0.2,
):
    k = max(
        1,
        int(map_a.size(1) * ratio),
    )

    top_a = torch.topk(
        map_a,
        k=k,
        dim=1,
    ).indices

    top_b = torch.topk(
        map_b,
        k=k,
        dim=1,
    ).indices

    overlaps = []

    for a, b in zip(top_a, top_b):
        set_a = set(a.tolist())
        set_b = set(b.tolist())

        intersection = len(
            set_a.intersection(set_b)
        )

        union = len(
            set_a.union(set_b)
        )

        overlaps.append(
            intersection / max(union, 1)
        )

    return sum(overlaps) / len(overlaps)
