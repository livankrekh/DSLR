#!./venv/bin/python3

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys

from src.tools import *

CURRENT_INDEXES = [1, 10, 11]

if __name__ == "__main__":

	if (len(sys.argv) < 2):
		print("Error: no input file!")
		exit()

	raw_data = validate(sys.argv[1])
	names = raw_data[:1][0][6:]
	raw_data = raw_data[1:]
	names = ["Age"] + names
	transposed = np.array(transform_data(raw_data))
	to_show = []

	for index in CURRENT_INDEXES:
		indexes = []
		tmp = []

		for i, elem in enumerate(transposed[index]):
			if (type(elem) is str) or (elem is None):
				indexes.append(i)

		tmp = np.delete(transposed[index], indexes)
		to_show.append(tmp)

	fig, axs = plt.subplots(1, len(CURRENT_INDEXES), figsize=(9, 3), sharey=True)

	for i in range(len(axs)):
		axs[i].hist(to_show[i].tolist(), bins=25, histtype = 'bar', facecolor = 'blue')
		axs[i].set_title(names[CURRENT_INDEXES[i]])
		print(to_show[i])

	plt.show()
