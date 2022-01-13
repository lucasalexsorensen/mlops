from ..data import MaskDataset
from torch import Tensor

class TestDataset:
    def test_dataset(self):
        dataset = MaskDataset(root_dir='src/test/sample_data')
        X,y = dataset[0]
        assert X.shape == (3,64,64)
        assert isinstance(X, Tensor)
        assert isinstance(y, Tensor)