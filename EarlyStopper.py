# taken from https://stackoverflow.com/a/73704579
import numpy as np


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            print(f'new early stop count {self.counter}')
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print(f'new early stop count {self.counter}')
            if self.counter >= self.patience:
                return True
        return False
