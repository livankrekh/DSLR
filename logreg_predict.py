#!./venv/bin/python3

import numpy as np
import sys

from src.tools import *

if __name__ == "__main__":

	if (len(sys.argv) < 2):
		print("\033[1m\033[31mError: no input csv file!\033[0m")
		exit()

	test = pd.read_csv(sys.argv[1], index_col = "Index")
	
