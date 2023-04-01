import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Define a custom PyTorch dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        X = np.load(os.path.join(self.root_dir, f'sentinel_{filename}'))
        y = np.load(os.path.join(self.root_dir, f'bhm_{filename}'))
        return X.flatten(), y.flatten()

# Define a custom PyTorch linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

print('Classes are well-defined')
# Create a PyTorch DataLoader object
dataset = CustomDataset('/home/valentina.unicef/sentinel_layers_500')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

# Create a PyTorch linear regression model
print('I am creating the model of your dreams')
model = LinearRegression(input_size=500 * 500 * 8, output_size=500 * 500)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    # Train the model on the training set
    train_loss_epoch = 0
    for X_batch, y_batch in train_dataloader:
        optimizer.zero_grad()
        y_pred = model(torch.from_numpy(X_batch).float())
        loss = criterion(y_pred, torch.from_numpy(y_batch).float())
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item() * X_batch.shape[0]
    train_loss_epoch /= len(train_dataset)
    train_losses.append(train_loss_epoch)

    # Compute the loss on the validation set
    val_loss_epoch = 0
    with torch.no_grad():
        for X_batch, y_batch in val_dataloader:
            y_pred = model(torch.from_numpy(X_batch).float())
            loss = criterion(y_pred, torch.from_numpy(y_batch).float())
            val_loss_epoch += loss.item() * X_batch.shape[0]
    val_loss_epoch /= len(val_dataset)
    val_losses.append(val_loss_epoch)

    print(f'Epoch {epoch + 1}: train loss = {train_loss_epoch:.4f}, val loss = {val_loss_epoch:.4f}')

## Plot the training and validation loss curves
#plt.plot(range(1, num_epochs+1), train_losses, label='Train')
#plt.plot(range(1, num_epochs+1), val_losses, label='Validation')
#plt.xlabel('Epoch')
#plt.ylabel('MSE Loss')
#plt.legend()
#plt.show()
