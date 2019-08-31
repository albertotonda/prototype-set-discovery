# This script has been designed to perform multi-objective learning of archetypes
# by Alberto Tonda, Pietro Barbiero, and Giovanni Squillero, 2018 <alberto.tonda@gmail.com> <pietro.barbiero@studenti.polito.it>

#basic libraries
import argparse
import copy
import datetime
import inspyred
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
import random
import os
import sys
import time
import logging
from sklearn.decomposition import PCA

from archetypes import evolveArchetypes, make_meshgrid, plot_contours, evaluate_core

#from coresets.algorithms import (construct_lr_coreset_with_kmeans,
#                                 random_data_subset,
#                                 full_data)
import bayesiancoresets as bc

# tensorflow library
import tensorflow as tf

# sklearn library
from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs, make_circles, make_moons
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
	figsize = [5, 4]
	seed = 42
	pop_size = 100
	offspring_size = 2 * pop_size
	max_generations = 100
	maximize = False
	selectedDataset = "mnist"
	selectedClassifiers = ["SVC"]

	# a list of classifiers
	allClassifiers = [
			[RandomForestClassifier, "RandomForestClassifier", 1],
#			[AdaBoostClassifier, "AdaBoostClassifier", 1],
#			[BaggingClassifier, "BaggingClassifier", 1],
#			[ExtraTreesClassifier, "ExtraTreesClassifier", 1],
#			[GradientBoostingClassifier, "GradientBoostingClassifier", 1],
#			[SGDClassifier, "SGDClassifier", 1],
#			[PassiveAggressiveClassifier, "PassiveAggressiveClassifier", 1],
#			[LogisticRegression, "LogisticRegression", 1],
#			[SVC, "SVC", 1],
			[RidgeClassifier, "RidgeClassifier", 1]
			]
	
	# a list of methods for coreset discovery
	algorithmList = [
			[bc.GIGA, "GIGA"],
			[bc.FrankWolfe, "FrankWolfe"],
			[bc.MatchingPursuit, "MatchingPursuit"],
			[bc.ForwardStagewise, "ForwardStagewise"],
			[bc.OrthoPursuit, "OrthoPursuit"],
			[bc.LAR, "LAR"],
#			[bc.ImportanceSampling, "ImportanceSampling"],
#			[bc.RandomSubsampling, "RandomSubsampling"]
			]
	
	selectedClassifiers = [classifier[1] for classifier in allClassifiers]
	
	folder_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-sota-" + selectedDataset
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
	# synthetic databases
	centers = [[1, 1], [-1, -1], [1, -1]]
	blobs_X, blobs_y = make_blobs(n_samples=400, centers=centers, n_features=2, cluster_std=0.6, random_state=seed)
	circles_X, circles_y = make_circles(n_samples=400, noise=0.15, factor=0.4, random_state=seed)
	moons_X, moons_y = make_moons(n_samples=400, noise=0.2, random_state=seed)
	iris = datasets.load_iris()
	digits = datasets.load_digits()
#	forest_X, forest_y = loadForestCoverageType() # local function
	mnist_X, mnist_y = loadMNIST() # local function

	dataList = [
			[blobs_X, blobs_y, 0, "blobs"],
			[circles_X, circles_y, 0, "circles"],
			[moons_X, moons_y, 0, "moons"],
	        [iris.data, iris.target, 0, "iris4"],
	        [iris.data[:, 2:4], iris.target, 0, "iris2"],
	        [digits.data, digits.target, 0, "digits"],
#			[forest_X, forest_y, 0, "covtype"],
			[mnist_X, mnist_y, 0, "mnist"]
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
	X = sc.transform(X)
	X_train = sc.transform(X_train)
	X_test = sc.transform(X_test)
		
	for algorithm in algorithmList:
		
		algorithm_name = algorithm[1]
		algorithm_class = algorithm[0]
		
		for classifier in classifierList:
	
			classifier_name = classifier[1]
			classifier_class = classifier[0]
			
			# start creating folder name
			experiment_name = os.path.join(folder_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-" + algorithm_name + "-" + dbname + "-" + classifier_name)
			if not os.path.exists(experiment_name) : os.makedirs(experiment_name)
	
			logger.info("Classifier used: " + classifier_name)
			
			start = time.time()
			core_set, trainAccuracy, testAccuracy = bayesian_coreset(X_train, y_train, X_test, y_test, algorithm_class, classifier_class, experiment_name=experiment_name, cname=classifier_name)
			end = time.time()
			exec_time = end - start
			
			# select "best" individuals
			logger.info("Compute performances!")
			logger.info("Elapsed time (seconds): %.4f" %(exec_time))
			logger.info("Initial performance: train=%.4f, test=%.4f, size: %d" % (trainAccuracy, testAccuracy, X_train.shape[0]))
			
#			# extract the core set
			X_core, y_core = X_train[core_set], y_train[core_set]
			X_err, accuracy, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_test, y_test, classifier[0], cname=classifier_name, SEED=seed)
			X_err_train, accuracy_train, model_train, fail_points_train, y_pred_train = evaluate_core(X_core, y_core, X_train, y_train, classifier[0], cname=classifier_name, SEED=seed)
			logger.info("%s core set: train: %.4f, test: %.4f, size: %d" %(algorithm_name, accuracy_train, accuracy, X_core.shape[0]))
			
			if False: #dbname == "mnist" or dbname == "digits":
				
				if dbname == "mnist":
					H, W = 28, 28
				if dbname == "digits":
					H, W = 8, 8
				
				logger.info("Now saving figures...")
				X_err, accuracy, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_test, y_test, classifier[0], cname=classifier_name, SEED=seed)
				X_err_train, accuracy_train, model_train, fail_points_train, y_pred_train = evaluate_core(X_core, y_core, X_train, y_train, classifier[0], cname=classifier_name, SEED=seed)
				
				# save archetypes
				for index in range(0, len(y_core)):
					image = np.reshape(X_core[index, :], (H, W))
					plt.figure()
					plt.axis('off')
					plt.imshow(image, cmap=plt.cm.gray_r)
					plt.title('Label: %d' %(y_core[index]))
					plt.tight_layout()
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
			
			# plot decision boundaries if we have only 2 dimensions!
			if X.shape[1] == 2:
				
				X_err, accuracy, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_test, y_test, classifier[0], cname=classifier_name, SEED=seed)
				X_err_train, accuracy_train, model_train, fail_points_train, y_pred_train = evaluate_core(X_core, y_core, X_train, y_train, classifier[0], cname=classifier_name, SEED=seed)
				
				cmap = plt.cm.jet
				xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1])
				figure = plt.figure(figsize=figsize)
				_, Z_0 = plot_contours(model, xx, yy, cmap=cmap, alpha=0.2)
				plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, marker='s', alpha=0.4, label="train")
				plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, marker='+', alpha=0.4, label="test")
				plt.scatter(X_core[:, 0], X_core[:, 1], marker='D', facecolors='none', edgecolors='k', alpha=1, label="archetypes")
				plt.scatter(X_err[:, 0], X_err[:, 1], marker='x', facecolors='k', edgecolors='k', alpha=1, label="errors")
				plt.legend()
				plt.title("%s - acc. %.4f" %(classifier_name, accuracy))
				plt.tight_layout()
				plt.savefig( os.path.join(experiment_name, "%s_%s_%s.png" %(dbname, algorithm_name, classifier_name)) )
				plt.savefig( os.path.join(experiment_name, "%s_%s_%s.pdf" %(dbname, algorithm_name, classifier_name)) )
				plt.close(figure)
				
				
				# using all samples in the training set
				X_core, y_core = X_train, y_train
				X_err, accuracy, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_test, y_test, classifier[0], cname=classifier_name, SEED=seed)
				X_err_train, accuracy_train, model_train, fail_points_train, y_pred_train = evaluate_core(X_core, y_core, X_train, y_train, classifier[0], cname=classifier_name, SEED=seed)
				
				cmap = plt.cm.jet
				xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1])
				figure = plt.figure(figsize=figsize)
				_, Z_0 = plot_contours(model, xx, yy, cmap=cmap, alpha=0.2)
				plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, marker='s', alpha=0.4, label="train")
				plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, marker='+', alpha=0.4, label="test")
				plt.scatter(X_err[:, 0], X_err[:, 1], marker='x', facecolors='k', edgecolors='k', alpha=1, label="errors")
				plt.legend()
				plt.title("%s - acc. %.4f" %(classifier_name, accuracy))
				plt.tight_layout()
				plt.savefig( os.path.join(experiment_name, "%s_%s_%s_alltrain.png" %(dbname, algorithm_name, classifier_name)) )
				plt.savefig( os.path.join(experiment_name, "%s_%s_%s_alltrain.pdf" %(dbname, algorithm_name, classifier_name)) )
				plt.close(figure)
				
				
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
	
#	pca = PCA(n_components=5)
#	X = pca.fit_transform(X)
	
	return X, y

def missing_class_fix(core_set, y_train):
	
	n_classes = len(np.unique(y_train))
	y_core = y_train[core_set]
	
	if n_classes != len(np.unique(y_core)):
		
		missing_classes = np.setdiff1d(np.unique(y_train), np.unique(y_core))
		
		for mc in missing_classes:
			indeces = np.argwhere(y_train==mc)
			core_set[indeces[0]] = 1
	
	return core_set

def bayesian_coreset(X_train, y_train, X_test, y_test, algorithm, classifier, experiment_name, cname=None):
	
	n_trials = 1
	Ms = np.unique(np.logspace(0., 4., 100, dtype=np.int32))
	wts = []
	for tr in range(n_trials):
		alg = algorithm(X_train)
		for m, M in enumerate(Ms):
			alg.run(M)
			wts = alg.weights()
	core_set = wts>0
	core_set = missing_class_fix(core_set, y_train)
	
	referenceClassifier = copy.deepcopy(classifier(random_state=42))
	referenceClassifier.fit(X_train, y_train)
	y_train_pred = referenceClassifier.predict(X_train)
	y_test_pred = referenceClassifier.predict(X_test)
	trainAccuracy = accuracy_score(y_train, y_train_pred)
	testAccuracy = accuracy_score(y_test, y_test_pred)
	print("Initial performance: train=%.4f, test=%.4f" % (trainAccuracy, testAccuracy))
	
	return core_set, trainAccuracy, testAccuracy

if __name__ == "__main__" :
	sys.exit( main() )
