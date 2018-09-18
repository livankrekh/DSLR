#!./venv/bin/python3

import tabulate

import csv
import sys
import re

def feature_type_is_number(feature_arr):
	count_float = len(list(filter(lambda e: isinstance(e, float), feature_arr)))

	return len(feature_arr) / 2 < count_float

def get_feature(dataset, i):
	feature_arr = []

	if (len(dataset[0]) - 1 < i):
		return []

	for row in dataset:
		feature_arr.append(row[i])

	feature_arr = list(filter(None, feature_arr))

	return feature_arr

def validate(file_name):
	dataset = []
	regex = re.compile('\-?\d+\.?\d*')

	with open(file_name) as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			tmp = []
			for elem in row:
				if (regex.match(elem)):
					tmp.append(float(regex.match(elem).group(0)))
				else:
					tmp.append(elem)
			dataset.append(tmp)

	return dataset

def print_in_table(dataset, names):
	intBased = []
	header_arr = [""]
	tableInfo = []
	properties = ["Count", "Min", "Max"]

	for i in range(1, len(dataset[0])):
		if (feature_type_is_number(get_feature(dataset, i))):
			intBased.append(i)
			header_arr.append(names[i])

	for prop in properties:
		tmp = [prop]
		
		for index in intBased:
			if (prop == "Count"):
				tmp.append(len(get_feature(dataset, index)))
			if (prop == "Min"):
				tmp.append(min(get_feature(dataset, index)))
			if (prop == "Max"):
				tmp.append(max(get_feature(dataset, index)))

		tableInfo.append(tmp)

	print(tabulate.tabulate(tableInfo, headers=header_arr, tablefmt='orgtbl'))



if __name__ == "__main__":
	dataset = []
	names = []

	if (len(sys.argv) < 2):
		print("Error: no input file!")
		exit()

	try:
		dataset = validate(sys.argv[1])
		names = dataset[:1][0]
		dataset = dataset[1:]
	except OSError as error:
		print("\033[1m\033[31mError: cannot open file ->", error, "\033[0m")
	except Exception as err:
		print("\033[1m\033[31mUnknown error:", err, "\033[0m")

	print_in_table(dataset, names)
