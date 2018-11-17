#!./venv/bin/python3

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys

from src.tools import *

CURRENT_INDEXES = [2, 3, 4]

def show_histogram(filename):

	raw_data = validate(filename)
	names = raw_data[:1][0][6:]
	raw_data = raw_data[1:]
	names = ["Age"] + names
	homes_data, homes = transform_data_by_homes(raw_data)

	for i in range(len(homes_data)):
		homes_data[i] = np.transpose(homes_data[i])

	fig, axs = plt.subplots(1, len(CURRENT_INDEXES), figsize=(9, 3), sharey=True)

	for i in range(len(axs)):
		axs[i].hist(homes_data[0][CURRENT_INDEXES[i]][homes_data[0][CURRENT_INDEXES[i]] != None].tolist(), bins=10, histtype = 'bar', facecolor = 'green', label=homes[0])
		axs[i].hist(homes_data[1][CURRENT_INDEXES[i]][homes_data[1][CURRENT_INDEXES[i]] != None].tolist(), bins=10, histtype = 'bar', facecolor = 'red', label=homes[1])
		axs[i].hist(homes_data[2][CURRENT_INDEXES[i]][homes_data[2][CURRENT_INDEXES[i]] != None].tolist(), bins=10, histtype = 'bar', facecolor = 'blue', label=homes[2])
		axs[i].hist(homes_data[3][CURRENT_INDEXES[i]][homes_data[3][CURRENT_INDEXES[i]] != None].tolist(), bins=10, histtype = 'bar', facecolor = 'yellow', label=homes[3])
		axs[i].set_title(names[CURRENT_INDEXES[i]])
	
	plt.legend(loc='upper right')
	plt.show()

if __name__ == "__main__":

	if (len(sys.argv) < 2):
		print("\033[1m\033[31mError: no input csv file!\033[0m")
		exit()

	try:
		show_histogram(sys.argv[1])
	except UnicodeDecodeError:
		print("\033[1m\033[31mMatplolib bug: no mouse scrolling supporting for OSX!\033[0m")
		exit()
	except KeyboardInterrupt:
		print("\033[1mBye, bye!\033[0m")
		exit()
	except Exception as err:
		print("\033[1m\033[31mUnexpected error: incorrect csv input data! Details:", err, "\033[0m")
