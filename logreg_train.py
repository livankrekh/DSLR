#!./venv/bin/python3

import pandas as pd
import numpy as np
import sys
import csv
import os
import argparse

from src.logreg import *
from src.tools import *

if __name__ == "__main__":

	if (len(sys.argv) < 2):
		print("\033[1m\033[31mError: no input csv file!\033[0m")
		exit()

	i = -1
	batch = -1
	alpha = -1

	ap = argparse.ArgumentParser()
	ap.add_argument("-b", "--batch", required = False, help = "Path to video")
	ap.add_argument("-i", "--iter", required = False, help = "Path to video")
	ap.add_argument("-a", "--alpha", required = False, help = "Path to video")
	ap.add_argument("-s", "--stohastic", required = False, action="store_true", help = "Path to video")
	ap.add_argument("-l", "--loss", required = False, action="store_true", help = "Path to video")
	ap.add_argument("-f", "--file", required = False, help = "Path to video")
	args = vars(ap.parse_args())

	try:
		if (args["iter"] != None):
			i = int(args["iter"])
		if (args["batch"] != None):
			batch = float(args["batch"])
			if batch > 1:
				batch = int(batch)
		if (args["alpha"] != None):
			alpha = float(args["alpha"])
	except Exception as err:
		print("\033[1m\033[31mArgument warning:", err, "\033[0m")

	raw = pd.read_csv(args["file"], index_col = "Index")
	
	fitter = DSLR_Logreg(raw, "Hogwarts House", batch = (batch if batch != -1 else 0.1), iter_n = (i if i >= 1 else 30000), alpha = (alpha if alpha != -1 else 0.0001), loss = args["loss"])
	fitter.choose_features(["Astronomy", "Herbology", "Defense Against the Dark Arts", "Ancient Runes", "Charms"])
	fitter.scaling()

	if args["stohastic"]:
		model = fitter.fit_stohastic()
	elif batch != -1:
		model = fitter.fit_miniBatch()
	else:
		model = fitter.fit()
	fitter.test()

	if (not os.path.exists("model/") or not os.path.isdir("model/")):
		os.mkdir("model")
	fitter.save("model/model")

	if args["loss"]:
		print("\033[1m\033[31mLoss ->", fitter.loss(), "\033[0m")
		fitter.loss_plot()
