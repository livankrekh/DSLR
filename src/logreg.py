from sklearn.model_selection import train_test_split
import numpy as np
import math

class DSLR_Logreg(object):
	def __init__(self, X, y_name, iter_n=1000, alpha=0.01):
		X = X.dropna()

		self.iter = iter_n
		self.alpha = alpha
		self.X, self.X_test = train_test_split(X, test_size=0.3)
		self.y = self.X[y_name]
		self.y_test = self.X_test[y_name]
		self.model = {}

	def get_X(self):
		return self.X

	def choose_features(self, feature_arr):
		if (len(feature_arr) < 1):
			return
		if (type(feature_arr[0]) is str):
			self.X = self.X[feature_arr]
			self.X_test = self.X_test[feature_arr]
		else:
			self.X = self.X[X.columns[feature_arr]]
			self.X_test = self.X_test[X_test.columns[feature_arr]]

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def scaling(self):
		for i in self.X.columns:
			try:
				self.X[i] = (self.X[i] - self.X[i].mean()) / self.X[i].std()
				self.X_test[i] = (self.X_test[i] - self.X_test[i].mean()) / self.X_test[i].std()
			except:
				continue

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
		print("Success -> ", int(((all_len - res) / all_len) * 100), "%", sep='')


		return (all_len - res) / all_len

	def save(self, path):
		np.save(path, self.model)

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


''' Logistic regression implementation without pandas and any frameworks '''

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total:
        print()
