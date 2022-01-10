# mlops
02476 - Machine Learning Operations (DTU)

## Project Description
Our task is to determine if there is a face mask present in a given image or not - i.e. a binary image classification task.

We plan to tackle the problem using [this dataset](https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset)


In terms of frameworks/tools, we plan to use the PyTorch-based computer vision library [kornia](https://github.com/kornia/kornia). Kornia has a lot of tools for us to use - both in terms of efficient image augmentations, and even pre-implemented model architectures for us to use.

We decided to try out vision transformers to see if they live up to the hype. Kornia actually has a pre-made implementation for this, but we might also try implementing it ourselves, since the architecture is quite simple (mostly consists of concatenating image patches with some positional embeddings, and then passing these through multi-head transformer layers)