import torch
from models.SimpleCNN import SimpleCNN, train_config
from train import train, test
from utils import get_data_loaders, get_test_loader
from config import *

model = SimpleCNN()
train_loader, val_loader = get_data_loaders(MODEL_INPUT_SHAPE, CENTRE_CROP, BATCH_SIZE)
train(model, train_loader, val_loader, train_config)
torch.save(model.state_dict(), f'{MODEL_SAVE_PATH}/model.pth')
test_loader = get_test_loader(MODEL_INPUT_SHAPE, CENTRE_CROP, BATCH_SIZE)

# model.load_state_dict(torch.load(f'{MODEL_SAVE_PATH}/model.pth'))
# model = model.to('mps')
loss, acc = test(model, test_loader, train_config)
print(f'overall test accuracy {acc}%')
