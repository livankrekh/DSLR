import pandas as pd
import numpy as np
import datetime
import csv
import re

# types: 0 - number, 1 - str, 2 - date, 3 - int, 4 - undefined (None)

def feature_type(feature_arr):
	count_float = len(list(filter(lambda e: isinstance(e, float), feature_arr)))
	count_str = len(list(filter(lambda e: isinstance(e, str), feature_arr)))
	count_date = len(list(filter(lambda e: isinstance(e, datetime.date), feature_arr)))
	count_int = len(list(filter(lambda e: isinstance(e, int), feature_arr)))

	if (len(feature_arr) / 2 < count_float):
		return 0
	if (len(feature_arr) / 2 < count_str):
		return 1
	if (len(feature_arr) / 2 < count_date):
		return 2
	if (len(feature_arr) / 2 < count_int):
		return 3

	return 4

def get_feature(dataset, i):
	feature_arr = []

	if (len(dataset[0]) - 1 < i):
		return []

	for row in dataset:
		if (row[i] == '' or row[i] == None):
			feature_arr.append(None)
		else:
			feature_arr.append(row[i])

	return feature_arr

def non_repeatable(feature_arr):
	non = []

	for elem in feature_arr:
		if (elem not in non and elem != None):
			non.append(elem)

	return non

def is_cluster_feature(feature_arr):
	non = non_repeatable(feature_arr)

	return len(non) < len(feature_arr) / 2

def validate(file_name):
	dataset = []
	regex_date = re.compile('\d+\-\d+\-?\d*')
	regex = re.compile('\-?\d+\.?\d*')

	with open(file_name) as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			tmp = []
			for elem in row:
				if (regex_date.match(elem)):
					tmp.append(datetime.datetime.strptime(regex_date.match(elem).group(0), "%Y-%m-%d").date())
				elif (regex.match(elem)):
					tmp.append(float(regex.match(elem).group(0)))
				else:
					tmp.append(elem)
			dataset.append(tmp)

	return dataset

def get_mean(feature_arr):
	averege = sum(feature_arr) / float(len(feature_arr))
	res = 0
	
	for elem in feature_arr:
		koff = abs(elem - averege)

		if (koff < abs(res - averege)):
			res = elem

	return res

def quantiles(feature_arr, quantile):
	sort = sorted(feature_arr)

	return sort[int((len(sort) - 1) * quantile)]

def standart_homogeneous(feature_arr):
	return homogeneous(feature_arr) ** 0.5

def homogeneous(feature_arr):
	averege = sum(feature_arr) / float(len(feature_arr))
	S = 0

	for elem in feature_arr:
		S += (elem - averege) ** 2

	S = S / float(len(feature_arr))

	return abs(S)

def get_clusters(raw):
	transposed = np.transpose(raw).tolist()
	clusters_arr = []
	names = []

	for feature_arr in transposed:
		non = non_repeatable(feature_arr)

		if (len(non) < len(feature_arr) / 2):
			names = non

			for elem in feature_arr:
				if (elem == None or elem not in non):
					clusters_arr.append(None)
				else:
					clusters_arr.append(non.index(elem))

			break

	return clusters_arr, names

def transform_data_by_homes(raw):
	transformed = np.transpose(transform_data(raw)).tolist()
	y, names = get_clusters(raw)
	result = []

	for _ in range(len(names)):
		result += [[]]

	for i, row in enumerate(transformed):
		result[y[i]].append(row)

	return result, names

def transform_data(raw):
	new_data = []

	for i in range(1, len(raw[0])):
		new_feature = []
		feature_arr = get_feature(raw, i)
		non = non_repeatable(feature_arr)

		if (feature_type(feature_arr) == 2):
			now = datetime.datetime.now().date()

			for elem in feature_arr:
				if (type(elem) is datetime.date):
					delta = now - elem
					new_feature.append(delta.days / 365)
				else:
					new_feature.append(None)

			new_data.append(new_feature)

		elif (feature_type(feature_arr) == 0):
			for elem in feature_arr:
				new_feature.append(elem)

			new_data.append(new_feature)

	return new_data

