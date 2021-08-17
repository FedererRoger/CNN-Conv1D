from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
#from sklearn.externals import joblib
import joblib
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from scipy.linalg import lstsq
from scipy import stats

# total count of wells
TOTAL_COUNT = 900

# path to wells
#DATASET_PATH = '../Datasets/SynthBase2020/train/'
#DATASET_PATH = '../Datasets/SynthBase2020/validation/'
DATASET_PATH = '../Datasets/SynthBase2020/field/'

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
	targets = []
	features = []
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
		f = np.concatenate((delta_f, T_f), axis=0)
		# f = delta_f
		features.append(f)

	# return targets, features
	return targets, features, data[0], data[1], data[3]
