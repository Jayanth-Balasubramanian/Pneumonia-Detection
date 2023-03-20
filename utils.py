import torch
from torchvision import datasets, transforms

import config

train_data_path = config.TRAIN_PATH
val_data_path = config.VAL_PATH
test_data_path = config.TEST_PATH


def get_data_loaders(model_input_size: int, model_center_crop: int, batch_size: int):
    transform = transforms.Compose({
        transforms.ToTensor(),
        transforms.Resize(model_input_size),
        transforms.CenterCrop(model_center_crop)
    })
    train_data = datasets.ImageFolder(train_data_path, transform=transform)
    train_size = len(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)

    validation_data = datasets.ImageFolder(val_data_path, transform=transform)
    val_size = len(validation_data)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size, shuffle=True)

    return train_loader, validation_loader
