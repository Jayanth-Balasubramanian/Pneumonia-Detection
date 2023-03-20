import torch
import time
from EarlyStopper import EarlyStopper
device = torch.device('mps')


def update_loss_and_accuracy(inputs, outputs, labels, loss, prev_loss, prev_accuracy):
    new_loss = prev_loss + loss.item() * inputs.size(0)
    ret, predictions = torch.max(outputs.data, 1)
    correct_counts = predictions.eq(labels.data.view_as(predictions))
    # Convert correct_counts to float and then compute the mean
    acc = torch.mean(correct_counts.type(torch.FloatTensor))
    # Compute total accuracy in the whole batch and add to train_acc
    new_acc = prev_accuracy + acc.item() * inputs.size(0)
    return new_loss, new_acc, acc


def validate(model, val_loader, loss_fn):
    valid_loss = 0
    valid_acc = 0
    with torch.no_grad():
        # Set to evaluation mode
        model.eval()
        # Validation loop
        for j, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            # Compute loss
            loss = loss_fn(outputs, labels)
            # Compute the total loss for the batch and add it to valid_loss
            valid_loss, valid_acc, acc = update_loss_and_accuracy(inputs, outputs, labels, loss, valid_loss, valid_acc)
            print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}"
                  .format(j, loss.item(), acc.item()))
        return valid_loss, valid_acc


def train(model: torch.nn.Module, train_loader, val_loader, train_config):
    model = model.to(device)
    lr = train_config['LR']
    optimizer = train_config['OPTIMIZER'](model.parameters(), lr)
    loss_fn = train_config['LOSS_FN']
    num_epochs = train_config['EPOCHS']

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    early_stopper = EarlyStopper(patience=3, min_delta=0.1)
    history = []
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model.train()
        train_loss, train_acc = 0, 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss, train_acc, acc = update_loss_and_accuracy(inputs, outputs, labels, loss, train_loss, train_acc)
            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}"
                  .format(i, loss.item(), acc.item()))

        valid_loss, valid_acc = validate(model, val_loader, loss_fn)
        # Find average training loss and training accuracy
        avg_train_loss = train_loss / train_size
        avg_train_acc = train_acc / float(train_size)
        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss / val_size
        avg_valid_acc = valid_acc / float(val_size)
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
        epoch_end = time.time()
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%,"
              "nttValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s"
              .format(epoch, avg_train_loss, avg_train_acc * 100, avg_valid_loss,
                      avg_valid_acc * 100, epoch_end - epoch_start))
        if early_stopper.early_stop(valid_loss):
            break
    return model, history
