from ..models.model import Net
from torch.testing import make_tensor
from torch import float32


class TestModel:
    def test_model(self):
        config = dict(
            embed_dim=64,
            patch_size=32,
            depth=10,
            num_heads=10,
            dropout_attn=0.5,
            dropout_rate=0.5,
            image_size=64,
        )
        model = Net(config)

        X = make_tensor((16, 3, 64, 64), device="cpu", dtype=float32, low=0, high=1)
        y_hat = model(X)
        assert y_hat.shape == (16, 2)
