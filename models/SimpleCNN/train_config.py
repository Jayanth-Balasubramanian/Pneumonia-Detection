import torch

train_config = {
    'OPTIMIZER': torch.optim.Adam,
    'LR': 0.001,
    'LOSS_FN': torch.nn.CrossEntropyLoss(),
    'EPOCHS': 25
}
