from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from glob import glob
from torch import Tensor
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import re


class MaskDataset(Dataset):
    def __init__(self, root_dir: str, split: str):
        assert split in ["train", "test"]
        self.root_dir = root_dir
        images = glob("%s/images/*.png" % root_dir)
        labels = glob("%s/annotations/*.xml" % root_dir)
        assert len(images) == len(labels)
        images_train, images_test, labels_train, labels_test = train_test_split(
            images, labels, test_size=0.3, random_state=1337, shuffle=True
        )
        if split == "train":
            self.images = images_train
            self.labels = labels_train
        elif split == "test":
            self.images = images_test
            self.labels = labels_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        image = read_image(self.images[index])
        label = int(
            bool(re.search(r"<name>with_mask<\/name>", open(self.labels[index]).read()))
        )

        return image, label
