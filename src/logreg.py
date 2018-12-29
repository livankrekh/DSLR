import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import math


class DSLR_Logreg(object):
	def __init__(self, X, y_name, iter_n=1000, alpha=0.01):
		self.iter = iter_n
		self.alpha = alpha
		self.X = X
		self.y = X[y_name]
		self.model = []

	def get_X(self):
		return self.X

	def choose_features(self, feature_arr):
		if (len(feature_arr) < 1):
			return
		if (type(feature_arr[0]) is str):
			X = X[feature_arr]
		else:
			X = X[X.columns[feature_arr]]

	def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

	def scaling(self):
		for i in range(len(X.columns)):
			try:
				X[i] = (X[i] - X.mean()) / X.std()
			except:
				continue

''' Logistic regression realizyng without pandas and any frameworks'''

def separete_data(data):
	train = []
	test = []

	for key, elem in enumerate(data):
		tmp = np.transpose(data[key])

		np.random.shuffle(tmp)
		train.append(np.transpose(tmp[:int(len(tmp) * 0.7)]))
		test.append(np.transpose(tmp[int(len(tmp) * 0.7):]))

	return train, test

def model_func2(theta, data):
	data = np.append(data, 1)
	X = sum(theta * np.transpose(data)) * -1

	return (1 / (1 + np.exp(X)))

def model_func(theta, data):
	res = []
	theta_x = []

	for cluster in data:
		cluster = np.vstack([cluster, [1] * len(cluster[0])])
		if (theta_x == []):
			theta_x = np.transpose(cluster) * theta
		else:
			theta_x = np.append(theta_x, np.transpose(cluster) * theta, axis=0)

	for i, x in enumerate(theta_x):
		tmp = 0.0

		for j, elem in enumerate(x):
			tmp += elem * theta[j]

		res.append(1 / (1 + np.exp(tmp * -1)))

	return np.array(res)

def loss_func(model, pos_data, neg_data):
	all_len = len(np.transpose(pos_data[0]))
	res = 0.0

	for elem in np.transpose(pos_data[0]):
		h = model_func2(model, elem)
		res += np.log(h)

	for cluster in neg_data:
		tmp = np.transpose(cluster)
		all_len += len(tmp)

		for elem in neg_data:
			h = model_func2(model, elem)
			res += np.log(1 - h)

	return (res / all_len) * -1

def derivative_func(theta, pos_data, neg_data, theta_index):
	h_theta = model_func(theta, pos_data + neg_data)
	all_len = len(np.transpose(pos_data[0]))
	res = 0.0
	h_i = 0

	for i, X in enumerate(np.transpose(pos_data[0])):
		if (0 in X):
			continue
		X = np.append(X, 1)

		res += (h_theta[h_i] - 1) * X[theta_index]
		h_i += 1

	for cluster in neg_data:
		tmp = np.transpose(cluster)
		all_len += len(tmp)

		for i, X in enumerate(tmp):
			X = np.append(X, 1)

			res += (h_theta[h_i] - 0) * X[theta_index]
			h_i += 1

	return (1 / all_len) * res


def logreg_one_model(pos_data, neg_data, alpha, steps):
	tmp_model = [0.0] * (len(pos_data[0]) + 1)
	tmp_model = np.array(tmp_model)

	for i in range(steps):
		print(i)
		tmp = tmp_model[:]

		for i, theta in enumerate(tmp_model):
			der = derivative_func(tmp, pos_data, neg_data, i)
			tmp[i] = theta - alpha * der
			print("LOSS ->", loss_func(theta, pos_data, neg_data))

		tmp_model = tmp

	return tmp_model


def clearData(data, curr_cluster, feature_indexes):
	pos = []
	neg = []

	for i, cluster in enumerate(data):
		tmp = []

		for j, feature in enumerate(cluster):
			if j in feature_indexes:
				feature[feature == None] = 0.0
				tmp.append(np.array(feature))

		tmp = np.array(tmp)
		if (i == curr_cluster):
			pos.append(tmp)
		else:
			neg.append(tmp)

	return pos, neg

def logreg_all(data_labeled, label_names, feature_indexes, alpha=0.05, steps=1000):
	res_model = {}
	pos_data = []
	neg_data = []

	for i, cluster_data in enumerate(data_labeled):

		pos_data, neg_data = clearData(data_labeled, i, feature_indexes)

		res_model[label_names[i]] = logreg_one_model(pos_data, neg_data, alpha, steps)

	return res_model

def check_values(model, elem):
	vals = np.array(model)
	res_sum = sum(np.transpose(vals) * np.append(elem, [1]))

	return 1.0 / (1.0 + np.exp(res_sum * -1.0))

def model_test(model, data, feature_indexes, homes):
	res = 0
	l = 0

	for i, cluster_data in enumerate(data):

		pos_data, neg_data = clearData(data, i, feature_indexes)
		pos_data = np.transpose(pos_data)

		for elem in pos_data:
			l += 1
			if (check_values(model[homes[i]], elem) >= 0.5):
				res += 1

		for cluster in neg_data:
			for elem in np.transpose(cluster):
				l += 1
				if (check_values(model[homes[i]], elem) < 0.5):
					res += 1

	print("Pos ->", res)
	print("Neg ->", l - res)
	print("All ->", l)

	return res / l

def data_scaling(data):

	for i, arr in enumerate(data):
		data[i] = np.nan_to_num(preprocessing.scale(arr))

	return data

def plot_data(model, data, FEATURES, homes):
	plt.scatter(data[0][FEATURES[0]], data[0][FEATURES[1]], color="green", alpha=0.7, marker='o', label=homes[0])
	plt.scatter(data[1][FEATURES[0]], data[1][FEATURES[1]], color="red" , alpha=0.7, marker='o', label=homes[1])
	plt.scatter(data[2][FEATURES[0]], data[2][FEATURES[1]], color="blue" , alpha=0.7, marker='o', label=homes[2])
	plt.scatter(data[3][FEATURES[0]], data[3][FEATURES[1]], color="yellow", alpha=0.7, marker='o', label=homes[3])

	# for elem in model.values():
	# 	x = np.arange(-10, 10)
	# 	y = (x * elem[2]) + (x * elem[3])
	# 	plt.plot(x, y)

	plt.xlabel(str(FEATURES[0]) + " in range feature")
	plt.ylabel(str(FEATURES[1]) + " in range feature")

	plt.legend(loc='upper left')

	plt.show()
