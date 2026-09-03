from __future__ import annotations


def build_resnet18(num_tasks: int = 6, pretrained: bool = True):
    """Build a compact multi-label image classifier with six logits."""
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_tasks)
    return model


def build_vit_tiny(num_tasks: int = 6, pretrained: bool = True):
    try:
        import timm
    except ImportError as exc:
        raise ImportError("ViT-Tiny requires `pip install timm`") from exc
    return timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=num_tasks)
