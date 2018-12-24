import numpy as np
import math

def separete_data(data):
	train = []
	test = []

	for key, elem in enumerate(data):
		tmp = np.transpose(data[key])

		np.random.shuffle(tmp)
		train.append(np.transpose(tmp[:int(len(tmp) * 0.7)]))
		test.append(np.transpose(tmp[int(len(tmp) * 0.7):]))

	return train, test

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

def derivative_func(theta, pos_data, neg_data, theta_index):
	h_theta = model_func(theta, pos_data + neg_data)
	all_len = len(np.transpose(pos_data[0]))
	res = 0.0
	h_i = 0

	for i, X in enumerate(np.transpose(pos_data[0])):
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
			tmp_model[i] = theta - alpha * derivative_func(tmp, pos_data, neg_data, i)

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

	return res / l


