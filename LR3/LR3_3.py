

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import transforms

#Загрузка данных
with np.load('arrows.npz', allow_pickle=True) as f:
    x_train = f[f.files[0]]
    y_train = f[f.files[1]]
    x_test = f[f.files[2]]
    y_test = f[f.files[3]]

# Нормализация
x_train, x_test = x_train / 255.0, x_test / 255.0

# PyTorch формат: [batch, channels, height, width]
if len(x_train.shape) == 4 and x_train.shape[3] == 1:
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))

# Тензоры
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).long()
x_test_tensor = torch.from_numpy(x_test).float()
y_test_tensor = torch.from_numpy(y_test).long()

#Модель с регуляризацией
class ImprovedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = ImprovedModel()

#Optimizer с регуляризацией
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

#dataloader
train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor),
                          batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor),
                         batch_size=32)

#обучение
EPOCHS = 5

print("Начало обучения улучшенной модели на датасете arrows.npz с регуляризацией")
print("=" * 70)

for epoch in range(EPOCHS):
    # Обучение
    model.train()
    train_loss, correct, total = 0, 0, 0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_acc = 100. * correct / total
    train_loss = train_loss / len(train_loader)

    # Тестирование
    model.eval()
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_acc = 100. * correct / total
    test_loss = test_loss / len(test_loader)

    print(f'Epoch {epoch + 1}/{EPOCHS}, '
          f'Train Loss: {train_loss:.4f}, '
          f'Train Acc: {train_acc:.2f}%, '
          f'Test Loss: {test_loss:.4f}, '
          f'Test Acc: {test_acc:.2f}%')

print("=" * 70)
print("Обучение завершено!")
