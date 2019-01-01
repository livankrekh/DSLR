#!./venv/bin/python3

from src.tools import *

import tabulate
import sys

def print_in_table(dataset, names):
	header_arr1 = [""]
	tableInfo = [["Count"], ["Mean"], ["Std"], ["Dispersion"], ["Min"], ["Max"], ["25%"], ["50%"], ["75%"]]

	for i, row in enumerate(dataset):

		if (feature_type(row) == 0):
			header_arr1.append(names[i])
			clear_data = list(filter(None.__ne__, row))

			tableInfo[0].append(len(clear_data))
			tableInfo[1].append(get_mean(clear_data))
			tableInfo[2].append(standart_homogeneous(clear_data))
			tableInfo[3].append(homogeneous(clear_data))
			tableInfo[4].append(min(clear_data))
			tableInfo[5].append(max(clear_data))
			tableInfo[6].append(quantiles(clear_data, 0.25))
			tableInfo[7].append(quantiles(clear_data, 0.5))
			tableInfo[8].append(quantiles(clear_data, 0.75))

	print("\033[1m\033[32mNumeric feature (float):\033[0m")
	print(tabulate.tabulate(tableInfo, headers=header_arr1, tablefmt='orgtbl'))

if __name__ == "__main__":
	raw_data = []
	names = []
	dataset = []

	if (len(sys.argv) < 2):
		print("Error: no input file!")
		exit()

	try:

		l = pd.read_csv(sys.argv[1], index_col = "Index")
		print(l.describe())

		raw_data = validate(sys.argv[1])
		names = raw_data[:1][0][6:]
		raw_data = raw_data[1:]
		dataset = transform_data(raw_data)

		names = ["Age"] + names

		print("\033[1m\033[32mFeatures from 1 to 7\033[0m")
		print_in_table(dataset[:7], names[:7])

		print("\033[1m\033[32mFeatures from 7\033[0m")
		print_in_table(dataset[7:], names[7:])

	except OSError as error:
		print("\033[1m\033[31mError: cannot open file ->", error, "\033[0m")
	except Exception as err:
		print("\033[1m\033[31mUnknown error:", err, "\033[0m")
