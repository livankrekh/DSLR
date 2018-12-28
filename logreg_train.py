import numpy as np
import sys
import csv
import os

from src.logreg import *
from src.tools import *

if __name__ == "__main__":

	if (len(sys.argv) < 2):
		print("\033[1m\033[31mError: no input csv file!\033[0m")
		exit()

	raw_data = validate(sys.argv[1])
	names = raw_data[:1][0][6:]
	raw_data = raw_data[1:]
	names = ["Age"] + names
	homes_data, homes = transform_data_by_homes(raw_data)

	for i in range(len(homes_data)):
		homes_data[i] = np.transpose(homes_data[i])

	train, test = separete_data(homes_data)

	model = logreg_all(train, homes, [2,3,4,7], 0.0001, 100)
	print("Success ->", model_test(model, test, [2,3,4,7], homes))
	plot_data(model, homes_data, [4,7], homes)

	if not os.path.exists("./model"):
		os.mkdir("model")

	with open("./model/model.csv", 'w', newline='') as file:
		writer = csv.DictWriter(file, fieldnames=model.keys())

		writer.writeheader()
		writer.writerow(model)
