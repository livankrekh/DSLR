import pandas as pd
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

	raw = pd.read_csv(sys.argv[1], index_col = "Index")
	