from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from glob import glob
from torch import Tensor
import torch
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import random


class MaskDataset(Dataset):
    def __init__(
        self, root_dir: str,
    ):
        # assert split in ["train", "test"]
        self.root_dir = root_dir
        pairs = []
        pairs += [(True, f) for f in glob("%s/mask/*.png" % root_dir)]
        pairs += [(False, f) for f in glob("%s/non_mask/*.png" % root_dir)]
        random.seed(1337)
        random.shuffle(pairs)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        label, im_path = self.pairs[index]
        image = read_image(im_path, mode=ImageReadMode.RGB).float() / 255
        label_v = [0, 0]
        label_v[label] = 1

        return image, torch.tensor(label_v).float()
