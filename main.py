from models.SimpleCNN import SimpleCNN, train_config
from train import train
from utils import get_data_loaders
from config import *

model = SimpleCNN()
train_loader, val_loader = get_data_loaders(MODEL_INPUT_SHAPE, CENTRE_CROP, BATCH_SIZE)

train(model, train_loader, val_loader, train_config)
