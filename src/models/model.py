import torch.nn.functional as F
import kornia as K
import torch.nn as nn


def make_model(
    embed_dim: int,
    patch_size: int,
    depth: int,
    num_heads: int,
    dropout_attn: float,
    dropout_rate: float,
    image_size=64,
):
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

