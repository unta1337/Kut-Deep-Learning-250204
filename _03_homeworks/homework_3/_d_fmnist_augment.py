import os
import imageio.v2 as imageio
import torch

from torch.utils.data import ConcatDataset
from torchvision import datasets
from torchvision.transforms import transforms, v2
from torchvision.transforms.functional import to_pil_image

BASE_PATH = '/home/ksy/Develop/Kut-Deep-Learning-250204'

f_mnist_path = os.path.join(BASE_PATH, "_00_data", "j_fashion_mnist")
f_mnist_train_data = datasets.FashionMNIST(f_mnist_path, train=True, download=True)

print(f'Original: {len(f_mnist_train_data)}')

img_transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomCrop([32, 32], padding=4),
])

transformed_f_mnist_train_data = []
for img, label in f_mnist_train_data:
    timg = img_transform(img)
    transformed_f_mnist_train_data.append((timg, label))

f_mnist_train_data = ConcatDataset([f_mnist_train_data, transformed_f_mnist_train_data])

print(f'Augmented: {len(f_mnist_train_data)}')
