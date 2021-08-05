import glob
import numpy
from torch import nn
import torch
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
global DataX, DataY
DataX = []
DataY = []


#Формирование матрицы объект(X)-ответ(Y): list->np.array(str)->np.array(float)
for filename in glob.glob('*.dat'):
    datContent = [i.strip().split() for i in open(filename).readlines()]
    DataY.append(datContent.pop(0))
    DataX.append(datContent)

DataX=numpy.array([numpy.array(xi) for xi in DataX])
DataX=DataX.astype("float")

DataY=numpy.array([numpy.array(xi) for xi in DataY])
DataY=DataY.astype("float")

#Переводим объекты в тензоры
DataX = torch.Tensor(DataX)
DataY = torch.Tensor(DataY)

#Определяем пользовательский датасет
class CustomDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

TT_SPLIT = 750
train_data = CustomDataset(DataX[:TT_SPLIT,...], DataY[:TT_SPLIT])
test_data = CustomDataset(DataX[TT_SPLIT:,...], DataY[TT_SPLIT:])
batch_size = 200
learning_rate = 0.003
train_loader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=True)

#Конструируем нейросеть
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(10, 20, 10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(20, 40, 10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),            
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1720, 600),
            nn.ReLU(),
            nn.Linear(600, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )            

    def forward(self, x):
        x = self.layer1(x)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return torch.sigmoid(x)        

#Обучаем модель
model = ConvNet()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct_count = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct_count += pred.eq(y.data.view_as(pred)).sum()

    test_loss /= num_batches
    correct = correct_count / size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} Correct: {correct_count} \n")

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, criterion, optimizer)
    test_loop(test_loader, model, criterion)
