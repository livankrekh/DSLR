import numpy as np
import math

def get_other_data(all_data, current):
	data = all_data[:]

def logreg_one_model(data):
	tmp_model = [0.0] * len(data)


def logreg_all(data_labeled, label_names, feature_indexes):
	res_model = {}

	for i, cluster_data in enumerate(data_labeled):

		X = np.take(cluster_data, feature_indexes)
		other = get_other_data(data_labeled, i)
		res_model[label_names[i]] = logreg_one_model(X, )
