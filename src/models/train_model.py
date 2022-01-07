from ..data import MaskDataset
from torch.utils.data import DataLoader

train_set = MaskDataset(root_dir="data/processed", split="train")
val_set = MaskDataset(root_dir="data/processed", split="test")

train_loader = DataLoader(train_set, batch_size=32)
val_loader = DataLoader(val_set, batch_size=32)
