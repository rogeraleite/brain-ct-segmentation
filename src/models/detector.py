"""
Stage-1 hemorrhage-presence detector for the v15 detect-then-segment cascade.

A ResNet-18 adapted to single-channel CT slices, emitting one logit = P(slice
contains hemorrhage). ImageNet-pretrained weights are used when downloadable
(they help on this tiny 36-positive-case dataset); otherwise it falls back to
random init so the code still runs offline.
"""

import torch
import torch.nn as nn


def build_detector(pretrained: bool = True) -> nn.Module:
    from torchvision.models import resnet18

    model = None
    if pretrained:
        try:
            from torchvision.models import ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception as e:  # offline / download blocked → random init
            print(f"[detector] pretrained unavailable ({e}); using random init")
    if model is None:
        model = resnet18(weights=None)
    # 1-channel stem: average the pretrained RGB conv1 kernels so we keep the
    # learned low-level filters instead of discarding them.
    old = model.conv1
    new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=False)
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
    model.conv1 = new
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
