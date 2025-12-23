import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.gem import GeM


class ConvNeXtPartModel(nn.Module):
    """
    ConvNeXt backbone with global and part-based feature extraction
    """

    def __init__(self, num_classes, backbone_name):
        super().__init__()

        self.back = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0
        )

        # Preserve spatial resolution in the final stage
        if hasattr(self.back, "stages"):
            if hasattr(self.back.stages[3], "downsample"):
                self.back.stages[3].downsample[1].stride = (1, 1)

        dim = self.back.num_features
        self.pool = GeM()

        self.bn_global = nn.BatchNorm1d(dim)
        self.cls_global = nn.Linear(dim, num_classes, bias=False)

        self.part_bns = nn.ModuleList(
            [nn.BatchNorm1d(dim) for _ in range(5)]
        )
        self.part_cls = nn.ModuleList(
            [nn.Linear(dim, num_classes, bias=False) for _ in range(5)]
        )

    def forward(self, x):
        f = self.back.forward_features(x)
        B, C, H, W = f.shape

        # Global feature
        g = self.pool(f).view(B, C)
        g = F.normalize(self.bn_global(g), p=2, dim=1)

        feat_list = [g]

        # 2-part horizontal features
        h2 = H // 2
        for i in range(2):
            p = self.pool(f[:, :, i * h2:(i + 1) * h2, :]).view(B, C)
            feat_list.append(
                F.normalize(self.part_bns[i](p), p=2, dim=1)
            )

        # 3-part horizontal features
        h3 = H // 3
        for i in range(3):
            p = self.pool(f[:, :, i * h3:(i + 1) * h3, :]).view(B, C)
            feat_list.append(
                F.normalize(self.part_bns[2 + i](p), p=2, dim=1)
            )

        return torch.cat(feat_list, dim=1)
