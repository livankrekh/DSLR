#!./venv/bin/python3

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sys

from src.tools import *

sns.set(style="ticks")

FEATURES = [2,3,4,7]

def show_pairplot(filename):
	pd_data = pd.read_csv(filename)
	names = pd_data.columns
	pd_data = pd_data.drop(columns="Index")
	pd_data = pd_data.drop(columns=names[2:6])

	pd_data = pd_data.iloc[:, [0] + FEATURES]
	pd_data = pd_data.dropna()

	sns.pairplot(pd_data, hue="Hogwarts House")
	plt.show()

if __name__ == "__main__":

	if (len(sys.argv) < 2):
		print("Error: no input file!")
		exit()

	try:
		show_pairplot(sys.argv[1])
	except UnicodeDecodeError:
		print("\033[1m\033[31mMatplolib bug: no mouse scrolling supporting for OSX!\033[0m")
		exit()
	except KeyboardInterrupt:
		print("\033[1mBye, bye!\033[0m")
		exit()
	except Exception as err:
		print("\033[1m\033[31mUnexpected error: incorrect csv input data! Details:", err, "\033[0m")
