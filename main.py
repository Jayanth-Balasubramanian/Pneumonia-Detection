import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as tf
import time

device = torch.device('mps')

data_path = "./data/chest_xray"

train_path = f'{data_path}/train'
test_path = f'{data_path}/test'
val_path = f'{data_path}/val'

output_size = (512, 512)
batch_size = 32
num_epochs = 10

transform = tf.Compose({
    tf.ToTensor(),
    tf.Resize(output_size),
    tf.CenterCrop(900)

})

train_data = torchvision.datasets.ImageFolder(train_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)

validation_data = torchvision.datasets.ImageFolder(val_path, transform=transform)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size, shuffle=True)

train_size = len(train_data)
val_size = len(validation_data)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 5),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 121 * 121, 200),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(10, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model = SimpleCNN()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

history = []
for epoch in range(num_epochs):
    epoch_start = time.time()
    print(f'Epoch {epoch + 1}/{num_epochs}')
    model.train()
    train_loss, train_acc, valid_loss, valid_acc = 0, 0, 0, 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        # print(f'output was {outputs.data}, prediction was {torch.max(outputs.data, 1)}')
        ret, predictions = torch.max(outputs.data, 1)
        # print(labels[0])
        # print(ret[0], predictions[0])
        # print(predictions.eq(labels.data.view_as(predictions))[0])
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        # Convert correct_counts to float and then compute the mean
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        # Compute total accuracy in the whole batch and add to train_acc
        train_acc += acc.item() * inputs.size(0)
        print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

    with torch.no_grad():
        # Set to evaluation mode
        model.eval()
        # Validation loop
        for j, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            # Compute loss
            loss = loss_fn(outputs, labels)
            # Compute the total loss for the batch and add it to valid_loss
            valid_loss += loss.item() * inputs.size(0)
            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            # Compute total accuracy in the whole batch and add to valid_acc
            valid_acc += acc.item() * inputs.size(0)
            print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
    # Find average training loss and training accuracy
    avg_train_loss = train_loss / train_size
    avg_train_acc = train_acc / float(train_size)
    # Find average training loss and training accuracy
    avg_valid_loss = valid_loss / val_size
    avg_valid_acc = valid_acc / float(val_size)
    history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
    epoch_end = time.time()
    print(
        "Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, nttValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100, epoch_end - epoch_start))
