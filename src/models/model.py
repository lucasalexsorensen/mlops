import torch.nn.functional as F
import kornia as K
import torch.nn as nn
from typing import Dict, Any


class Net(nn.Module):
    def __init__(self, config: Dict[str, Any] = {}):
        super(Net, self).__init__()
        embed_dim: int = config.get("embed_dim", 225)
        patch_size: int = config.get("patch_size", 4)
        depth: int = config.get("depth", 5)
        num_heads: int = config.get("num_heads", 16)
        dropout_attn: float = config.get("dropout_attn", 0.2356)
        dropout_rate: float = config.get("dropout_rate", 0.1056)
        image_size = 64

        self.model = nn.Sequential(
            K.contrib.VisionTransformer(
                image_size=image_size,
                embed_dim=embed_dim,
                patch_size=patch_size,
                depth=depth,
                num_heads=num_heads,
                dropout_attn=dropout_attn,
                dropout_rate=dropout_rate,
            ),
            K.contrib.ClassificationHead(embed_size=embed_dim, num_classes=2),
        )

    def forward(self, x):
        return self.model(x)

