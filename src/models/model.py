import torch.nn.functional as F
import kornia as K
import torch.nn as nn
from typing import Dict, Any


def make_model(config: Dict[str, Any]):
    embed_dim: int = config["embed_dim"]
    patch_size: int = config["patch_size"]
    depth: int = config["depth"]
    num_heads: int = config["num_heads"]
    dropout_attn: float = config["dropout_attn"]
    dropout_rate: float = config["dropout_rate"]
    image_size = 64

    model = nn.Sequential(
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
    return model

