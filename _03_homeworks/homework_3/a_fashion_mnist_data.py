import os
from pathlib import Path
import torch
import wandb
from torch import nn

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import transforms

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)  # BASE_PATH: /Users/yhhan/git/link_dl
print(BASE_PATH)

import sys

sys.path.append(BASE_PATH)

from _01_code._99_common_utils.utils import get_num_cpu_cores, is_linux, is_windows


def get_fashion_mnist_data():
    data_path = os.path.join(BASE_PATH, "_00_data", "j_fashion_mnist")

    f_mnist_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transforms.ToTensor())
    f_mnist_train, f_mnist_validation = random_split(f_mnist_train, [55_000, 5_000])

    # mean std
    if False:
        train_imgs = torch.stack([i for i, _ in f_mnist_train], dim=3)
        print(f'f_mnist_train:')
        print(f'    mean: {train_imgs.view(1, -1).mean()}')
        print(f'    std:  {train_imgs.view(1, -1).std()}')
    else:
        # f_mnist_train:
        #     mean: 0.28632092475891113
        #     std:  0.353121280670166
        train_stat = (0.28632092475891113, 0.353121280670166)

    print("Num Train Samples: ", len(f_mnist_train))
    print("Num Validation Samples: ", len(f_mnist_validation))
    print(f"Stat Train Samples: (mean, std): {train_stat}")
    print("Sample Data Shape: ", f_mnist_train[0][0].shape)  # torch.Size([1, 28, 28])
    print("Sample Data Target: ", f_mnist_train[0][1])  # 9

    num_data_loading_workers = get_num_cpu_cores()
    print("Number of Data Loading Workers:", num_data_loading_workers)

    train_data_loader = DataLoader(
        dataset=f_mnist_train, batch_size=wandb.config.batch_size, shuffle=True,
        pin_memory=True, num_workers=num_data_loading_workers
    )

    validation_data_loader = DataLoader(
        dataset=f_mnist_validation, batch_size=wandb.config.batch_size,
        pin_memory=True, num_workers=num_data_loading_workers
    )

    f_mnist_transforms = nn.Sequential(
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=train_stat[0], std=train_stat[1]),
    )

    return train_data_loader, validation_data_loader, f_mnist_transforms


def get_fashion_mnist_test_data():
    data_path = os.path.join(BASE_PATH, "_00_data", "j_fashion_mnist")

    f_mnist_test_images = datasets.FashionMNIST(data_path, train=False, download=True)
    f_mnist_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transforms.ToTensor())

    # mean std
    if False:
        test_imgs = torch.stack([i for i, _ in f_mnist_test], dim=3)
        print(f'f_mnist_test:')
        print(f'    mean: {test_imgs.view(1, -1).mean()}')
        print(f'    std:  {test_imgs.view(1, -1).std()}')

        print('[INFO] Calculating mean, std done. quitting...')
        exit(0)
    else:
        # f_mnist_train:
        #     mean: 0.2868492901325226
        #     std:  0.3524441719055176
        test_stat = (0.2868492901325226, 0.3524441719055176)

    print("Num Test Samples: ", len(f_mnist_test))
    print(f"Stat Test Samples: (mean, std): {test_stat}")
    print("Sample Shape: ", f_mnist_test[0][0].shape)  # torch.Size([1, 28, 28])

    test_data_loader = DataLoader(dataset=f_mnist_test, batch_size=len(f_mnist_test))

    f_mnist_transforms = nn.Sequential(
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=0.0, std=0.1),
    )

    return f_mnist_test_images, test_data_loader, f_mnist_transforms


if __name__ == "__main__":
    config = {'batch_size': 2048, }
    wandb.init(mode="disabled", config=config)

    train_data_loader, validation_data_loader, f_mnist_transforms = get_fashion_mnist_data()
    print()
    f_mnist_test_images, test_data_loader, f_mnist_transforms = get_fashion_mnist_test_data()

    print('[INFO] done')
