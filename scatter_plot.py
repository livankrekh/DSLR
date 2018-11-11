#!./venv/bin/python3

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys

from src.tools import *

FEATURES = [2,3]

if __name__ == "__main__":

	if (len(sys.argv) < 2):
		print("Error: no input file!")
		exit()

	raw_data = validate(sys.argv[1])
	names = raw_data[:1][0][6:]
	raw_data = raw_data[1:]
	feature_names = ["Age"] + names
	homes_data, homes = transform_data_by_homes(raw_data)

	for i in range(len(homes_data)):
		homes_data[i] = np.transpose(homes_data[i])

	plt.scatter(homes_data[0][FEATURES[0]], homes_data[0][FEATURES[1]], color="green", alpha=0.7, marker='o', label=homes[0])
	plt.scatter(homes_data[1][FEATURES[0]], homes_data[1][FEATURES[1]], color="red" , alpha=0.7, marker='o', label=homes[1])
	plt.scatter(homes_data[2][FEATURES[0]], homes_data[2][FEATURES[1]], color="blue" , alpha=0.7, marker='o', label=homes[2])
	plt.scatter(homes_data[3][FEATURES[0]], homes_data[3][FEATURES[1]], color="yellow", alpha=0.7, marker='o', label=homes[3])

	plt.xlabel(feature_names[FEATURES[0]])
	plt.ylabel(feature_names[FEATURES[1]])

	plt.legend(loc='upper left')

	plt.show()
