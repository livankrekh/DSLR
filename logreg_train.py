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
	
	fitter = DSLR_Logreg(raw, "Hogwarts House", 30000, 0.0001)
	fitter.choose_features(["Astronomy", "Herbology", "Defense Against the Dark Arts", "Ancient Runes"])
	fitter.scaling()
	model = fitter.fit()
	fitter.test()
	if (not os.path.exists("model/") or not os.path.isdir("model/")):
		os.mkdir("model")
	fitter.save("model/model")
