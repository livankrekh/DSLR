#!./venv/bin/python3

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys

from src.tools import *

FEATURES = [1,2,3,4]

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

		for j in range(len(homes_data[i])):
			tmp_indexes = []

			for k, elem in enumerate(homes_data[i][j]):
				if (type(elem) is str) or (elem is None):
					homes_data[i][j][k] = 0


	fig, all_vis = plt.subplots(4, 4)

	for i in range(len(FEATURES)):
		for j in range(len(FEATURES)):
			if (i == j):
				all_vis[i][j].hist(homes_data[0][FEATURES[j]].tolist(), bins=25, alpha=0.5, histtype='bar', facecolor='green')
				all_vis[i][j].hist(homes_data[1][FEATURES[j]].tolist(), bins=25, alpha=0.5, histtype='bar', facecolor='red')
				all_vis[i][j].hist(homes_data[2][FEATURES[j]].tolist(), bins=25, alpha=0.5, histtype='bar', facecolor='blue')
				all_vis[i][j].hist(homes_data[3][FEATURES[j]].tolist(), bins=25, alpha=0.5, histtype='bar', facecolor='yellow')
			else:
				all_vis[i][j].scatter(homes_data[0][FEATURES[i]], homes_data[0][FEATURES[j]], color="green", alpha=0.7, marker='o', label=homes[0])
				all_vis[i][j].scatter(homes_data[1][FEATURES[i]], homes_data[1][FEATURES[j]], color="red" , alpha=0.7, marker='o', label=homes[1])
				all_vis[i][j].scatter(homes_data[2][FEATURES[i]], homes_data[2][FEATURES[j]], color="blue" , alpha=0.7, marker='o', label=homes[2])
				all_vis[i][j].scatter(homes_data[3][FEATURES[i]], homes_data[3][FEATURES[j]], color="yellow", alpha=0.7, marker='o', label=homes[3])
				# all_vis[i][j].legend(loc='upper left')

			all_vis[i][j].grid()


	fig.tight_layout()
	plt.show()
