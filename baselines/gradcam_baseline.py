import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
)


class GradCAMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def make_gradcam(
    model,
    images,
    targets,
):
    model.eval()

    target_layer = model.encoder[-1][-1].conv2

    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
    )

    target_objects = [
        ClassifierOutputTarget(
            int(target)
        )
        for target in targets
    ]

    grayscale_cam = cam(
        input_tensor=images,
        targets=target_objects,
    )

    cam_map = torch.tensor(
        grayscale_cam,
        device=images.device,
        dtype=torch.float32,
    )

    cam_map = cam_map.unsqueeze(1)

    cam_map = F.adaptive_avg_pool2d(
        cam_map,
        output_size=(7, 7),
    )

    cam_map = cam_map.flatten(1)

    cam_map = cam_map / (
        cam_map.sum(
            dim=1,
            keepdim=True,
        )
        + 1e-8
    )

    return cam_map
