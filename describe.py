#!./venv/bin/python3

from src.tools import *

import tabulate
import sys

def print_in_table(dataset, names):
	header_arr1 = [""]
	header_arr2 = [""]
	tableInfo = [["Count"], ["Mean"], ["Min"], ["Max"]]
	clusterInfo = [["Count"], ["Mean"]]


	for i, row in enumerate(dataset):
		if (len(non_repeatable(row)) < len(row) / 2):
			header_arr2.append("Feature #" + str(len(header_arr2)))

			clusterInfo[0].append(len(list(filter(None.__ne__, row))))

		elif (feature_type(row) == 0):
			header_arr1.append("Feature #" + str(len(header_arr1)))

			tableInfo[0].append(len(list(filter(None.__ne__, row))))
			tableInfo[1].append(get_mean(list(filter(None.__ne__, row))))
			tableInfo[2].append(min(list(filter(None.__ne__, row))))
			tableInfo[3].append(max(list(filter(None.__ne__, row))))

	print("\033[1m\033[32mNumeric feature (float):\033[0m")
	print(tabulate.tabulate(tableInfo, headers=header_arr1, tablefmt='orgtbl'))
	print("\n\033[1m\033[32mCluster feature:\033[0m")
	print(tabulate.tabulate(clusterInfo, headers=header_arr2, tablefmt='orgtbl'))

if __name__ == "__main__":
	raw_data = []
	names = []
	dataset = []

	if (len(sys.argv) < 2):
		print("Error: no input file!")
		exit()

	try:
		raw_data = validate(sys.argv[1])
		names = raw_data[:1][0]
		raw_data = raw_data[1:]
		dataset = transform_data(raw_data)

		print_in_table(dataset, names)

	except OSError as error:
		print("\033[1m\033[31mError: cannot open file ->", error, "\033[0m")
	except Exception as err:
		print("\033[1m\033[31mUnknown error:", err, "\033[0m")
