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

def transform_data(raw):
	new_data = []

	for i in range(1, len(raw[0])):
		new_feature = []
		feature_arr = get_feature(raw, i)
		non = non_repeatable(feature_arr)

		if (len(non) < len(feature_arr) / 2):
			for elem in feature_arr:
				if (elem == None or elem not in non):
					new_feature.append(None)
				else:
					new_feature.append(non.index(elem))

			new_data.append(new_feature)

		elif (feature_type(feature_arr) == 2):
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