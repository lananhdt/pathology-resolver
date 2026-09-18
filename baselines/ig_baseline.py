import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients


def make_ig(
    model,
    images,
    targets,
    steps=32,
):
    model.eval()

    def forward_fn(x):
        return model(x)

    explainer = IntegratedGradients(
        forward_fn
    )

    attribution = explainer.attribute(
        images,
        target=targets,
        n_steps=steps,
    )

    attribution = attribution.abs().sum(
        dim=1,
        keepdim=True,
    )

    attribution = F.adaptive_avg_pool2d(
        attribution,
        output_size=(7, 7),
    )

    attribution = attribution.flatten(1)

    attribution = attribution / (
        attribution.sum(
            dim=1,
            keepdim=True,
        )
        + 1e-8
    )

    return attribution
