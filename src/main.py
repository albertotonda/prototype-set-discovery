# This script has been designed to perform multi-objective learning of archetypes
# by Alberto Tonda, Pietro Barbiero, and Giovanni Squillero, 2018 <alberto.tonda@gmail.com> <pietro.barbiero@studenti.polito.it>

#basic libraries
import argparse
import copy
import datetime
import inspyred
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys
import time
import logging

from archetypes import evolveArchetypes, make_meshgrid, plot_contours, evaluate_core

# tensorflow library
import tensorflow as tf

# sklearn library
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

# pandas
from pandas import read_csv

import warnings
warnings.filterwarnings("ignore")

def main():
	# a few hard-coded values
	seed = 42
	pop_size = 100
	offspring_size = 2 * pop_size
	max_generations = 100
	maximize = False
	selectedDataset = "iris"
	selectedClassifiers = ["SVC"]

	# a list of classifiers
	allClassifiers = [
			[RandomForestClassifier, "RandomForestClassifier", 1],
			[AdaBoostClassifier, "AdaBoostClassifier", 1],
			[BaggingClassifier, "BaggingClassifier", 1],
			[ExtraTreesClassifier, "ExtraTreesClassifier", 1],
			[GradientBoostingClassifier, "GradientBoostingClassifier", 1],
			[SGDClassifier, "SGDClassifier", 1],
			[SVC, "SVC", 1],
			[PassiveAggressiveClassifier, "PassiveAggressiveClassifier", 1],
			[LogisticRegression, "LogisticRegression", 1],
			[RidgeClassifier, "RidgeClassifier", 1],
			[LogisticRegressionCV, "LogisticRegressionCV", 1],
			[RidgeClassifierCV, "RidgeClassifierCV", 0],
			]
	
	selectedClassifiers = [classifier[1] for classifier in allClassifiers]
	
	folder_name = "archetypes-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
	if not os.path.exists(folder_name) : 
		os.makedirs(folder_name)
	else :
		sys.stderr.write("Error: folder \"" + folder_name + "\" already exists. Aborting...\n")
		sys.exit(0)
	# open the logging file
	logfilename = os.path.join(folder_name, 'logfile.log')
	logger = setup_logger('logfile_' + folder_name, logfilename)
	logger.info("All results will be saved in folder \"%s\"" % folder_name)

	# load different datasets, prepare them for use
	logger.info("Preparing data...")
	iris = datasets.load_iris()
	digits = datasets.load_digits()
	forest_X, forest_y = loadForestCoverageType() # local function
#	mnist_X, mnist_y = loadMNIST() # local function

	dataList = [
	        [iris.data, iris.target, 0, "iris"],
	        [digits.data, digits.target, 0, "digits"],
			[forest_X, forest_y, 0, "covtype"],
#			[mnist_X, mnist_y, 0, "mnist"]
		      ]

	# argparse; all arguments are optional
	parser = argparse.ArgumentParser()

	parser.add_argument("--classifiers", "-c", nargs='+', help="Classifier(s) to be tested. Default: %s. Accepted values: %s" % (selectedClassifiers[0], [x[1] for x in allClassifiers]))
	parser.add_argument("--dataset", "-d", help="Dataset to be tested. Default: %s. Accepted values: %s" % (selectedDataset,[x[3] for x in dataList]))

	parser.add_argument("--pop_size", "-p", type=int, help="EA population size. Default: %d" % pop_size)
	parser.add_argument("--offspring_size", "-o", type=int, help="Ea offspring size. Default: %d" % offspring_size)
	parser.add_argument("--max_generations", "-mg", type=int, help="Maximum number of generations. Default: %d" % max_generations)

	# finally, parse the arguments
	args = parser.parse_args()

	# a few checks on the (optional) inputs
	if args.dataset :
		selectedDataset = args.dataset
		if selectedDataset not in [x[3] for x in dataList] :
			logger.info("Error: dataset \"%s\" is not an accepted value. Accepted values: %s" % (selectedDataset, [x[3] for x in dataList]))
			sys.exit(0)

	if args.classifiers != None and len(args.classifiers) > 0 :
		selectedClassifiers = args.classifiers
		for c in selectedClassifiers :
			if c not in [x[1] for x in allClassifiers] :
				logger.info("Error: classifier \"%s\" is not an accepted value. Accepted values: %s" % (c, [x[1] for x in allClassifiers]))
				sys.exit(0)

	if args.max_generations : max_generations = args.max_generations
	if args.pop_size : pop_size = args.pop_size
	if args.offspring_size : offspring_size = args.offspring_size

	# TODO: check that min_points < max_points and max_generations > 0


	# print out the current settings
	logger.info("Settings of the experiment...")
	logger.info("Fixed random seed: %d" %(seed))
	logger.info("Selected dataset: %s; Selected classifier(s): %s" % (selectedDataset, selectedClassifiers))
	logger.info("Population size in EA: %d; Offspring size: %d; Max generations: %d" % (pop_size, offspring_size, max_generations))

	# create the list of classifiers
	classifierList = [ x for x in allClassifiers if x[1] in selectedClassifiers ]

	# pick the dataset
	db_index = -1
	for i in range(0, len(dataList)) :
		if dataList[i][3] == selectedDataset :
			db_index = i

	dbname = dataList[db_index][3]

	X, y = dataList[db_index][0], dataList[db_index][1]
	number_classes = np.unique(y).shape[0]

	logger.info("Creating train/test split...")
	from sklearn.model_selection import StratifiedKFold
	skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
	listOfSplits = [split for split in skf.split(X, y)]
	train_index, test_index = listOfSplits[0]
	X_train, y_train = X[train_index], y[train_index]
	X_test, y_test = X[test_index], y[test_index]
	logger.info("Training set: %d lines (%.2f%%); test set: %d lines (%.2f%%)" % (X_train.shape[0], (100.0 * float(X_train.shape[0]/X.shape[0])), X_test.shape[0], (100.0 * float(X_test.shape[0]/X.shape[0]))))
	
	# rescale data
	scaler = StandardScaler()
	sc = scaler.fit(X_train)
	X_train = sc.transform(X_train)
	X_test = sc.transform(X_test)
	
	for classifier in classifierList:

		classifier_name = classifier[1]

		# start creating folder name
		experiment_name = os.path.join(folder_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-archetypes-evolution-" + dbname + "-" + classifier_name)
		if not os.path.exists(experiment_name) : os.makedirs(experiment_name)

		logger.info("Classifier used: " + classifier_name)
		
		start = time.time()
		solutions = evolveArchetypes(X, y, X_train, y_train, X_test, y_test, 
									classifier, pop_size, offspring_size, max_generations,
									number_classes=number_classes, 
									maximize=maximize, seed=seed, experiment_name=experiment_name)
		end = time.time()
		exec_time = end - start
		
		# only candidates with all classes are considered
		final_archive = []
		for sol in solutions :
			c = sol.candidate
			c = np.array(c)
			y_core = c[:, -1]
			if len(set(y_core)) == number_classes :
				final_archive.append(sol)
		
		logger.info("Now saving final Pareto front in a figure...")
		pareto_front_x = [ f.fitness[0] for f in final_archive ]
		pareto_front_y = [ f.fitness[1] for f in final_archive ]

		figure = plt.figure()
		ax = figure.add_subplot(111)
		ax.plot(pareto_front_x, pareto_front_y, "bo-", label="Solutions in final archive")
		ax.set_title("Final archive")
		ax.set_xlabel("Number of points in the core set")
		ax.set_ylabel("Performance")
		ax.set_xlim([1, X_train.shape[0]])
		ax.set_ylim([0, 1])
		plt.savefig( os.path.join(experiment_name, "final_archive.png") )
		plt.savefig( os.path.join(experiment_name, "final_archive.pdf") )
		plt.close(figure)
		
		figure = plt.figure()
		ax = figure.add_subplot(111)
		ax.plot(pareto_front_x, pareto_front_y, "bo-", label="Solutions in final archive")
		ax.set_title("Final archive")
		ax.set_xlabel("Number of points in the core set")
		ax.set_ylabel("Performance")
		plt.savefig( os.path.join(experiment_name, "final_archive_zoom.png") )
		plt.savefig( os.path.join(experiment_name, "final_archive_zoom.pdf") )
		plt.close(figure)
		
		# select "best" individuals
		logger.info("Compute performances!")
		logger.info("Elapsed time (seconds): %.4f" %(exec_time))
		
		# best compromise
		final_archive = sorted(final_archive, key = lambda x : x.fitness[0] * x.fitness[1])
		individual = np.array(final_archive[0].candidate)
		n_features = X_train.shape[1]
		X_core, y_core = individual[:, :n_features], individual[:, -1]
		X_err, accuracy, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_test, y_test, classifier[0], cname=classifier_name, SEED=seed)
		X_err_train, accuracy_train, model_train, fail_points_train, y_pred_train = evaluate_core(X_core, y_core, X_train, y_train, classifier[0], cname=classifier_name, SEED=seed)
		logger.info("Best compromise: train: %.4f, test: %.4f, size: %d" %(accuracy_train, accuracy, X_core.shape[0]))
		
		# minimal set size
		final_archive = sorted(final_archive, key = lambda x : x.fitness[0])
		individual = np.array(final_archive[0].candidate)
		n_features = X_train.shape[1]
		X_core, y_core = individual[:, :n_features], individual[:, -1]
		X_err, accuracy, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_test, y_test, classifier[0], cname=classifier_name, SEED=seed)
		X_err_train, accuracy_train, model_train, fail_points_train, y_pred_train = evaluate_core(X_core, y_core, X_train, y_train, classifier[0], cname=classifier_name, SEED=seed)
		logger.info("Minimal set size: train: %.4f, test: %.4f, size: %d" %(accuracy_train, accuracy, X_core.shape[0]))
		
		# minimal training error
		final_archive = sorted(final_archive, key = lambda x : x.fitness[1])
		individual = np.array(final_archive[0].candidate)
		n_features = X_train.shape[1]
		X_core, y_core = individual[:, :n_features], individual[:, -1]
		X_err, accuracy, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_test, y_test, classifier[0], cname=classifier_name, SEED=seed)
		X_err_train, accuracy_train, model_train, fail_points_train, y_pred_train = evaluate_core(X_core, y_core, X_train, y_train, classifier[0], cname=classifier_name, SEED=seed)
		logger.info("Minimal training error: train: %.4f, test: %.4f, size: %d" %(accuracy_train, accuracy, X_core.shape[0]))
		
		if dbname == "mnist" or dbname == "digits":
			
			if dbname == "mnist":
				H, W = 28, 28
			if dbname == "digits":
				H, W = 8, 8
			
			logger.info("Now saving figures...")
			final_archive = sorted(final_archive, key = lambda x : x.fitness[0] * x.fitness[1])
			individual = np.array(final_archive[0].candidate)
			n_features = X_train.shape[1]
			X_core, y_core = individual[:, :n_features], individual[:, -1]
			
			X_err, accuracy, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_test, y_test, classifier[0], cname=classifier_name, SEED=seed)
			X_err_train, accuracy_train, model_train, fail_points_train, y_pred_train = evaluate_core(X_core, y_core, X_train, y_train, classifier[0], cname=classifier_name, SEED=seed)
			
			# save archetypes
			for index in range(0, len(y_core)):
				image = np.reshape(X_core[index, :], (H, W))
				plt.figure()
				plt.axis('off')
				plt.imshow(image, cmap=plt.cm.gray_r)
				plt.title('Label: %d' %(y_core[index]))
				plt.savefig( os.path.join(experiment_name, "digit_%d_idx_%d.pdf" %(y_core[index], index)) )
				plt.savefig( os.path.join(experiment_name, "digit_%d_idx_%d.png" %(y_core[index], index)) )
				plt.close()
			
			# save test errors
			e = 1
			for index in range(0, len(y_test)):
				if fail_points[index] == True:
					image = np.reshape(X_test[index, :], (H, W))
					plt.figure()
					plt.axis('off')
					plt.imshow(image, cmap=plt.cm.gray_r)
					plt.title('Label: %d - Prediction: %d' %(y_test[index], y_pred[index]))
					plt.savefig( os.path.join(experiment_name, "err_lab_%d_pred_%d_idx_%d.pdf" %(y_test[index], y_pred[index], e)) )
					plt.savefig( os.path.join(experiment_name, "err_lab_%d_pred_%d_idx_%d.png" %(y_test[index], y_pred[index], e)) )
					plt.close()
					e = e + 1
		
		if False:
			logger.info("Now saving points to CSV...")
			with open( os.path.join(experiment_name, "final_archive.csv"), "w") as fp :
				fp.write("#points,accuracy,individual\n")
				for f in final_archive :
					fp.write( str(f.fitness[0]) )
					fp.write( "," + str(f.fitness[1]) )
	
					for g in f.candidate :
						fp.write( "," + str(g) )
	
					fp.write("\n")
	
	logger.handlers.pop()

	return


def setup_logger(name, log_file, level=logging.INFO):
	"""Function setup as many loggers as you want"""
	
	formatter = logging.Formatter('%(asctime)s %(message)s')
	handler = logging.FileHandler(log_file)        
	handler.setFormatter(formatter)
	
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)
	
	return logger

# utility function to load the covtype dataset
def loadForestCoverageType() :

	inputFile = "../data/covtype.csv"
	#logger.info("Loading file \"" + inputFile + "\"...")
	df_covtype = read_csv(inputFile, delimiter=',', header=None)

	# class is the last column
	covtype = df_covtype.as_matrix()
	X = covtype[:,:-1]
	y = covtype[:,-1].ravel()-1

	return X, y

def loadMNIST():
	mnist = tf.keras.datasets.mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	
	X = np.concatenate((x_train, x_test))
	X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[1]))
	y = np.concatenate((y_train, y_test))
	
	return X, y

if __name__ == "__main__" :
	sys.exit( main() )
