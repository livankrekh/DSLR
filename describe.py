#!./venv/bin/python3

from src.tools import *

import tabulate
import sys

def print_in_table(dataset, names):
	header_arr1 = [""]
	header_arr2 = [""]
	tableInfo = [["Count"], ["Mean"], ["Dispersion (%)"], ["Min"], ["Max"], ["25%"], ["50%"], ["75%"]]
	clusterInfo = [["Count"]]


	for i, row in enumerate(dataset):
		if (is_cluster_feature(row)):
			header_arr2.append("Feature #" + str(len(header_arr2)))

			clusterInfo[0].append(len(list(filter(None.__ne__, row))))

		elif (feature_type(row) == 0):
			header_arr1.append("Feature #" + str(len(header_arr1)))
			clear_data = list(filter(None.__ne__, row))

			tableInfo[0].append(len(clear_data))
			tableInfo[1].append(get_mean(clear_data))
			tableInfo[2].append(homogeneous(clear_data) * 100)
			tableInfo[3].append(min(clear_data))
			tableInfo[4].append(max(clear_data))
			tableInfo[5].append((sum(clear_data) / len(clear_data)) / 4)
			tableInfo[6].append(sum(clear_data) / len(clear_data))
			tableInfo[7].append(((sum(clear_data) / len(clear_data)) / 4) * 3)

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

		# print_in_table(dataset, names)

	except OSError as error:
		print("\033[1m\033[31mError: cannot open file ->", error, "\033[0m")
	except Exception as err:
		print("\033[1m\033[31mUnknown error:", err, "\033[0m")

	data_by_homes(raw_data)
