import matplotlib
matplotlib.use('TkAgg')

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import math

class DSLR_Logreg(object):
	def __init__(self, X, y_name, iter_n=30000, alpha=0.0001, batch=0.1, loss=False):
		X = X.dropna()

		self.iter = iter_n
		self.alpha = alpha
		self.X, self.X_test = train_test_split(X, test_size=0.3)
		_, self.X_batch = train_test_split(self.X, test_size=batch)
		self.y = self.X[y_name]
		self.y_name = y_name
		self.y_test = self.X_test[y_name]
		self.y_batch = self.X_batch[y_name]
		self.model = {}
		self.batch_size = batch
		self.loss_on = loss
		self.loss_arr = []

	def get_X(self):
		return self.X

	def choose_features(self, feature_arr):
		if (len(feature_arr) < 1):
			return
		if (type(feature_arr[0]) is str):
			self.X = self.X[feature_arr]
			self.X_test = self.X_test[feature_arr]
			self.X_batch = self.X_batch[feature_arr]
		else:
			self.X = self.X[X.columns[feature_arr]]
			self.X_test = self.X_test[X_test.columns[feature_arr]]
			self.X_batch = self.X_batch[X_batch.columns[feature_arr]]

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def loss_plot(self):
		plt.plot(self.loss_arr)
		plt.show()

	def loss(self):
		all_i = 1
		loss = 0

		if self.y_name in self.X.columns:
			X = self.X.drop(self.y_name, axis=1)
		else:
			X = self.X

		for house in self.model.keys():
			labeled_y = np.where(self.y == house, 1, 0)
			theta = self.model[house]

			all_i += len(labeled_y)
			
			h = self.sigmoid(X.dot(theta))
			loss += sum(labeled_y * np.log(h) + ((1 - labeled_y) * np.log(1 - h)))

		return (loss / all_i) * -1


	def scaling(self):
		for i in self.X.columns:
			try:
				self.X[i] = (self.X[i] - self.X[i].mean()) / self.X[i].std()
				self.X_test[i] = (self.X_test[i] - self.X_test[i].mean()) / self.X_test[i].std()
				self.X_batch[i] = (self.X_batch[i] - self.X_batch[i].mean()) / self.X_batch[i].std()
			except:
				continue

	def fit_miniBatch(self, stohastic=True):
		theta0 = np.ones(self.X_batch.shape[0])
		self.X_batch.insert(loc=0, column="X0", value=theta0, allow_duplicates=True)
		all_i = len(np.unique(self.y)) * self.iter

		for i, house_name in enumerate(np.unique(self.y)):
			labeled_y = np.where(self.y_batch == house_name, 1, 0)
			tmp = np.ones(self.X_batch.shape[1])

			for j in range(self.iter):
				new_X = self.X_batch.dot(tmp)
				pressicion = labeled_y - self.sigmoid(new_X)
				grad = np.dot(self.X_batch.T, pressicion)
				tmp += self.alpha * grad

				if self.loss_on and i % 500 == 0:
					self.model[house_name] = tmp
					self.loss_arr.append(self.loss())

				printProgressBar(i * self.iter + j, all_i, 'Training progress')

			self.model[house_name] = tmp

		printProgressBar(all_i, all_i, 'Training progress')

		return self.model

	def fit_stohastic(self):
		theta0 = np.ones(self.X.shape[0])
		self.X.insert(loc=1, column="X0", value=theta0, allow_duplicates=True)
		self.X.insert(loc=0, column=self.y_name, value=self.y, allow_duplicates=True)
		all_i = len(np.unique(self.y)) * self.iter

		for i, house_name in enumerate(np.unique(self.y)):
			tmp = np.ones(self.X.shape[1] - 1)

			for j in range(self.iter):
				ex = self.X.sample(n=1)
				y = np.where(ex[self.y_name] == house_name, 1, 0)
				ex = ex.drop(self.y_name, axis=1)

				new_X = ex.dot(tmp)
				pressicion = y - self.sigmoid(new_X)
				grad = np.dot(ex.T, pressicion)
				tmp += self.alpha * grad

				self.model[house_name] = tmp

				if self.loss_on and i % 500 == 0:
					self.model[house_name] = tmp
					self.loss_arr.append(self.loss())

				printProgressBar(i * self.iter + j, all_i, 'Training progress')

			self.model[house_name] = tmp

		printProgressBar(all_i, all_i, 'Training progress')

		self.X = self.X.drop(self.y_name, axis=1)

		return self.model

	def fit(self):
		theta0 = np.ones(self.X.shape[0])
		self.X.insert(loc=0, column="X0", value=theta0, allow_duplicates=True)
		all_i = len(np.unique(self.y)) * self.iter

		for i, house_name in enumerate(np.unique(self.y)):
			labeled_y = np.where(self.y == house_name, 1, 0)
			tmp = np.ones(self.X.shape[1])

			for j in range(self.iter):
				printProgressBar(i * self.iter + j, all_i, 'Training progress')
				new_X = self.X.dot(tmp)
				pressicion = labeled_y - self.sigmoid(new_X)
				grad = np.dot(self.X.T, pressicion)
				tmp += self.alpha * grad

				if self.loss_on and i % 500 == 0:
					self.model[house_name] = tmp
					self.loss_arr.append(self.loss())

			self.model[house_name] = tmp

		printProgressBar(all_i, all_i, 'Training progress')

		return self.model

	def test(self):
		X0 = np.ones(self.X_test.shape[0])
		self.X_test.insert(loc=0, column="X0", value=X0, allow_duplicates=True)
		res = 0

		for house in np.unique(self.y):
			labeled_y = np.where(self.y_test == house, 1, 0)
			theta = self.model[house]

			new_y = self.sigmoid(self.X_test.dot(theta))
			new_y = np.where(new_y >= 0.5, 1, 0)
			loss = (labeled_y - new_y) ** 2
			res += sum(loss)

		all_len = len(self.X_test) * len(np.unique(self.y))

		print("All ->", all_len)
		print("Pos ->", all_len - res)
		print("Neg ->", res)
		print("\033[1m\033[32mSuccess -> ", int(((all_len - res) / all_len) * 100), "%\033[0m", sep='')


		return (all_len - res) / all_len

	def save(self, path):
		np.save(path, self.model)
		print("\033[1m\033[32mModel saved to '" + path + ".npy'\033[0m")

class DSLR_Predict(object):
	def __init__(self, X, model, y_name):
		self.X = X
		self.model = model

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def choose_features(self, feature_arr):
		if (len(feature_arr) < 1):
			return
		if (type(feature_arr[0]) is str):
			self.X = self.X[feature_arr]
		else:
			self.X = self.X[X.columns[feature_arr]]

		self.X = self.X.dropna()

	def scaling(self):
		for i in self.X.columns:
			try:
				self.X[i] = (self.X[i] - self.X[i].mean()) / self.X[i].std()
			except:
				continue	

	def predict(self):
		theta0 = np.ones(self.X.shape[0])
		self.X.insert(loc=0, column="X0", value=theta0, allow_duplicates=True)
		y = []

		for i in range(len(self.X.iloc[:])):
			res = (0.0, "")

			for key in self.model.keys():
				res_key = self.sigmoid(self.X.iloc[i].dot(self.model[key]))

				if (res_key > res[0]):
					res = (res_key, key)

			y.append(res[1])

		return y

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total:
        print()
