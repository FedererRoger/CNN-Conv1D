from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
#from sklearn.externals import joblib
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
import subprocess
import os
from scipy.linalg import lstsq
from scipy import stats
import glob
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
global accuracy_mas, loss_mas
global features, targets

# total count of wells
TOTAL_COUNT = 900

# path to wells
DATASET_PATH = '../Datasets/SynthBase2020/train/'
#DATASET_PATH = '../Datasets/SynthBase2020/validation/'
#DATASET_PATH = '../Datasets/SynthBase2020/field/'

# split wells to train and test

def add_tiles(data, tile_size):
	N = len(data)
	N_APPROX = 5

	# left
	x = np.array(range(N_APPROX))
	y = data[0:N_APPROX]
	A = np.vstack([x, np.ones(N_APPROX)]).T
	a, b = np.linalg.lstsq(A, y, rcond = None)[0]
	left_tile = a * np.array(range(-tile_size, 0)) + b

	# right
	x = np.array(range(N-N_APPROX, N))
	y = data[-N_APPROX:]
	A = np.vstack([x, np.ones(N_APPROX)]).T
	a, b = np.linalg.lstsq(A, y, rcond = None)[0]
	right_tile = a * np.array(range(N, N+tile_size)) + b

	# unite
	return np.concatenate((left_tile,data,right_tile), axis=0)


# get data from window
def getDataFromWindow(data, pos, win2_size, tile_size):
	start, end = pos - win2_size, pos + win2_size
	return data[start+tile_size:end+tile_size+1]

def getDataFromWindowStep(data, pos, win2_size, step, tile_size):
	start, end = pos - win2_size, pos + win2_size
	return data[start+tile_size:end+tile_size+1:step]

# get target from central part of window
def getTargetFromWindow(data):
	width = 3
	N = len(data)
	sum_target = 0.0
	for i in range(N//2 - width, N//2 + width + 1):
		sum_target += data[i]
	return sum_target / (2.0*width + 1)

def normFeature(f):
	v_max = np.max(f)
	v_min = np.min(f)
	if v_max - v_min < 1e-3:
		return np.zeros((len(f)))
	return 2.0*(f - np.min(f))/(v_max - np.min(f)) - 1.0


# create features from one well
def createFeatures(id):
	N_hyper = 4
	data = np.loadtxt(DATASET_PATH + f"{id}.result", unpack=True)
	# for field data - rescale to 0.2 m step after simple mnooth
	data = [v for v in data]
	Nsteps = 5*(data[0][-1] - data[0][0])
	z_new = np.linspace(data[0][0], data[0][-1], int(Nsteps))
	data[1] = np.interp(z_new, data[0], data[1])
	data[2] = np.interp(z_new, data[0], data[2])
	data[3] = np.interp(z_new, data[0], data[3])
	data[4] = np.interp(z_new, data[0], data[4])
	data[0] = z_new

	# rescale to base z
	N = len(data[0])
	z = data[0]
	T_m = data[1]
	G = data[2]	

	TILE_SIZE = 128
	G = add_tiles(G, TILE_SIZE)
	T_m = add_tiles(T_m, TILE_SIZE)
	delta = T_m - G
	inflows = add_tiles(data[3], TILE_SIZE)

	# make targets and features
	win2_size = 8
	for i in range(0, N):
		# targets
		inflow_data = getDataFromWindow(inflows, i, win2_size, TILE_SIZE)
		#print('inflow_data')
		#print(inflow_data)
		inflow_target = getTargetFromWindow(inflow_data)
		targets.append([inflow_target])

		# features
		delta_f = normFeature(getDataFromWindowStep(delta, i, win2_size * N_hyper, N_hyper, TILE_SIZE))
		T_f = normFeature(getDataFromWindowStep(T_m, i, win2_size * N_hyper, N_hyper, TILE_SIZE))
		features.append([delta_f]+[T_f])
	# return targets, features
	return 1

features = []
targets = []

for i in range(200):
	createFeatures(i)
#features_delta = np.array([np.array(xi) for xi in features_delta])
#features_delta = features_delta.astype("float")
#features_Tf = np.array([np.array(xi) for xi in features_Tf])
#features_Tf = features_Tf.astype("float")

DataX = np.array([np.array(xi) for xi in features])
DataX = DataX.astype("float")
DataY = np.array([np.array(xi) for xi in targets])
DataY = DataY.astype("float")

DataX = torch.Tensor(DataX)
DataY = torch.Tensor(DataY)

print(DataY)

print(len(DataX))

#Определяем пользовательский датасет
class CustomDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

TT_SPLIT = (3 * len(DataY)) // 4
print(TT_SPLIT)
train_data = CustomDataset(DataX[:TT_SPLIT,...], DataY[:TT_SPLIT])
test_data = CustomDataset(DataX[TT_SPLIT:,...], DataY[TT_SPLIT:])
batch_size = 100
learning_rate = 0.003
train_loader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=True)

#Конструируем нейросеть
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(2, 4, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(4, 8, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return torch.sigmoid(x)

#Обучаем модель
model = ConvNet()
criterion = nn.BCELoss()
#criterion = nn.CrossEntropyLoss()
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
    accuracy_mas.append(correct*100)
    loss_mas.append([test_loss])
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} Correct: {correct_count} \n")



epochs = 50

epochs_mas = np.arange(1, epochs+1)
accuracy_mas = []
loss_mas = []

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, criterion, optimizer)
    test_loop(test_loader, model, criterion)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.grid()
ax2.grid()
ax1.set_xlabel('Num_epoch')
ax1.set_ylabel('Accurcacy')
ax2.set_xlabel('Num_epoch')
ax2.set_ylabel('Loss')
ax1.plot(epochs_mas, accuracy_mas, color='r', linewidth=3, label = 'Accuracy')
ax1.legend()
ax2.plot(epochs_mas, loss_mas, color='g', linewidth=3, label = 'Loss')
ax2.legend()
plt.show()
