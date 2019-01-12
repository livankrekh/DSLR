#!./venv/bin/python3

import pandas as pd
import numpy as np
import sys

from src.logreg import *
from src.tools import *

if __name__ == "__main__":

	if (len(sys.argv) < 3):
		print("\033[1m\033[31mError: no input csv file!\033[0m")
		exit()

	test = pd.read_csv(sys.argv[1], index_col = "Index")
	model = np.load(sys.argv[2])
	model = dict(model.tolist())

	predict = DSLR_Predict(test, model, "Hogwarts House")
	predict.choose_features(["Astronomy", "Herbology", "Defense Against the Dark Arts", "Ancient Runes", "Charms"])
	predict.scaling()
	predicts = predict.predict()

	houses = pd.DataFrame({'Index':range(len(predicts)), 'Hogwarts House':predicts})
	houses.to_csv('houses.csv', index=False)
