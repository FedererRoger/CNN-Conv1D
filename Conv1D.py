import glob
import numpy
from torch import nn
import torch
from torchvision.transforms import ToTensor

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

print(DataX)

#Определяем пользовательский датасет
class CustomDataset(Dataset):

    #def __init__(self):

    def __len__(self):
        return len(DataY)
    def __getitem__(self, idx):
        image = DataX[idx]
        label = DataY[idx]
        return image, label

#Создаем экземпляр класса Dataset
Data = CustomDataset()
batch_size = 100
learning_rate = 0.001
train_loader = DataLoader(dataset=Data, batch_size=batch_size,shuffle=True)

#Конструируем нейросеть
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv1d(10, 16, 5), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2))
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.layer1(x)
        out = self.flatten(out)
        return out

#Обучаем модель
model = ConvNet()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_list = []
acc_list = []
for i, (images, labels) in enumerate(train_loader):
    # Прямой запуск
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss_list.append(loss.item())

    # Обратное распространение и оптимизатор
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Отслеживание точности
    total = labels.size(0)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    acc_list.append(correct / total)

print(acc_list)







