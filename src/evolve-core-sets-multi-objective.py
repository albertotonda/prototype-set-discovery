# This script has been designed to perform multi-objective learning of core sets 
# by Alberto Tonda and Pietro Barbiero, 2018 <alberto.tonda@gmail.com> <pietro.barbiero@studenti.polito.it>

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

# sklearn library
from sklearn import datasets
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

def main() :
	
	# a few hard-coded values
	seed = 42
	max_points_in_core_set = 99
	min_points_in_core_set = 1 # later redefined as 1 per class
	pop_size = 100
	offspring_size = 2 * pop_size
	max_generations = 100
	maximize = True
	selectedDataset = "iris"
	selectedClassifiers = ["RandomForestClassifier"]

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
	
	# load different datasets, prepare them for use
	print("Preparing data...")
	iris = datasets.load_iris()
	#wine = datasets.load_wine()
	#breast = datasets.load_breast_cancer()
	#digits = datasets.load_digits()
	forest_X, forest_y = loadForestCoverageType() # local function 
	
	dataList = [
	        [iris.data, iris.target, 0, "iris"],
	        #[wine.data, wine.target, 0, "wine"],
	        #[breast.data, breast.target, 0, "breast"],
	        #[digits.data, digits.target, 0, "digits"],
		[forest_X, forest_y, 0, "covtype"]
	        ]
	
	# argparse; all arguments are optional
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--classifiers", "-c", nargs='+', help="Classifier(s) to be tested. Default: %s. Accepted values: %s" % (selectedClassifiers[0], [x[1] for x in allClassifiers]))
	parser.add_argument("--dataset", "-d", help="Dataset to be tested. Default: %s. Accepted values: %s" % (selectedDataset,[x[3] for x in dataList]))
	
	parser.add_argument("--pop_size", "-p", type=int, help="EA population size. Default: %d" % pop_size)
	parser.add_argument("--offspring_size", "-o", type=int, help="Ea offspring size. Default: %d" % offspring_size)
	parser.add_argument("--max_generations", "-mg", type=int, help="Maximum number of generations. Default: %d" % max_generations)
	
	parser.add_argument("--min_points", "-mip", type=int, help="Minimum number of points in the core set. Default: %d" % min_points_in_core_set)
	parser.add_argument("--max_points", "-mxp", type=int, help="Maximum number of points in the core set. Default: %d" % max_points_in_core_set)
	
	# finally, parse the arguments
	args = parser.parse_args()
	
	# a few checks on the (optional) inputs
	if args.dataset : 
		selectedDataset = args.dataset
		if selectedDataset not in [x[3] for x in dataList] :
			print("Error: dataset \"%s\" is not an accepted value. Accepted values: %s" % (selectedDataset, [x[3] for x in dataList]))
			sys.exit(0)
	
	if args.classifiers != None and len(args.classifiers) > 0 :
		selectedClassifiers = args.classifiers
		for c in selectedClassifiers :
			if c not in [x[1] for x in allClassifiers] :
				print("Error: classifier \"%s\" is not an accepted value. Accepted values: %s" % (c, [x[1] for x in allClassifiers]))
				sys.exit(0)
	
	if args.min_points : min_points_in_core_set = args.min_points
	if args.max_points : max_points_in_core_set = args.max_points
	if args.max_generations : max_generations = args.max_generations
	if args.pop_size : pop_size = args.pop_size
	if args.offspring_size : offspring_size = args.offspring_size
	
	# TODO: check that min_points < max_points and max_generations > 0
	
	
	# print out the current settings
	print("Settings of the experiment...")
	print("Fixed random seed:", seed)
	print("Selected dataset: %s; Selected classifier(s): %s" % (selectedDataset, selectedClassifiers))
	print("Min points in candidate core set: %d; Max points in candidate core set: %d" % (min_points_in_core_set, max_points_in_core_set))
	print("Population size in EA: %d; Offspring size: %d; Max generations: %d" % (pop_size, offspring_size, max_generations))

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
	
	print("Creating train/test split...")
	from sklearn.model_selection import StratifiedKFold
	skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
	listOfSplits = [split for split in skf.split(X, y)]
	train_index, test_index = listOfSplits[0]
	X_train, y_train = X[train_index], y[train_index]
	X_test, y_test = X[test_index], y[test_index]
	print("Training set: %d lines (%.2f%%); test set: %d lines (%.2f%%)" % (X_train.shape[0], (100.0 * float(X_train.shape[0]/X.shape[0])), X_test.shape[0], (100.0 * float(X_test.shape[0]/X.shape[0]))))
	
	for classifier in classifierList:
		
		classifier_name = classifier[1]

		# start creating folder name
		experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-core-set-evolution-" + dbname + "-" + classifier_name
		if not os.path.exists(experiment_name) : os.makedirs(experiment_name)
	   
		print("\nClassifier used: " + classifier_name)
		print("All results will be saved in folder \"%s\"" % experiment_name)

		final_archive = evolveCoreSets(X, y, X_train, y_train, X_test, y_test, classifier, pop_size, offspring_size, max_generations, min_points_in_core_set, max_points_in_core_set, number_classes, maximize, seed=seed, experiment_name=experiment_name) 
		
		final_archive_file = os.path.join(experiment_name, "final_archive.csv")
		print("Now saving final Pareto front to file...")
		with open(final_archive_file, "w") as fp :
			fp.write("#points,accuracy,individual\n")
			
			for c in final_archive :
				fp.write(str(X_train.shape[0] - c.fitness[0]) + "," + str(c.fitness[1]))
				for g in c.candidate : fp.write("," + str(g))
				fp.write("\n")
		
		print("Now saving final Pareto front in a figure...") 
		pareto_front_x = [ (X_train.shape[0]  - f.fitness[0]) for f in final_archive ]
		pareto_front_y = [ f.fitness[1] for f in final_archive ]

		figure = plt.figure()
		ax = figure.add_subplot(111)
		ax.plot(pareto_front_x, pareto_front_y, "bo-", label="Solutions in final archive")    
		ax.set_title("Final archive")
		ax.set_xlabel("Number of points in the core set")
		ax.set_ylabel("Performance")
		plt.savefig( os.path.join(experiment_name, "final_archive.png") )
		plt.savefig( os.path.join(experiment_name, "final_archive.pdf") )
		plt.close(figure)
	
		print("Now saving all individuals generated during the evolution to a figure...")
		all_population_file_name = os.path.join(experiment_name, "all_individuals.csv")
		all_individuals_x = []
		all_individuals_y = []
		with open(all_population_file_name, "r") as fp :
			lines = fp.readlines()
			lines.pop(0)
			
			for line in lines :
				tokens = line.rstrip().split(',')
				all_individuals_x.append( float(tokens[0]) )
				all_individuals_y.append( float(tokens[1]) )
		
		figure = plt.figure()
		ax = figure.add_subplot(111)
		ax.plot(all_individuals_x, all_individuals_y, "b.", linestyle="None", label="All individuals")    
		ax.plot(pareto_front_x, pareto_front_y, "D", color="red", fillstyle="none", label="Pareto front")
		ax.set_title("All individuals created during the evolution")
		ax.set_xlabel("Number of points in the core set")
		ax.set_ylabel("Accuracy")
		ax.legend(loc="best")
		plt.savefig( os.path.join(experiment_name, "all_population.png") )
		plt.savefig( os.path.join(experiment_name, "all_population.pdf") )
		plt.close(figure)
		
		print("Now plotting the 'best individual'...")
		# compute PCA
		from sklearn.decomposition import PCA
		pca = PCA(n_components=2)
		X_pca = pca.fit_transform(X)
		X_train_pca = pca.transform(X_train)
		
		# separate data by class
		X_pca_by_class = dict()
		classes = np.unique(y)
		for c in classes : X_pca_by_class[c] = X_pca[ y == c ]
		
		figure = plt.figure()
		ax = figure.add_subplot(111)
		
		# first plot all the points, divided by class
		for c in classes : ax.plot(X_pca_by_class[c][:,0], X_pca_by_class[c][:,1], marker='.', linestyle='None', label="Class %d" % c)
		
		# then, mark ones in the 'best' individual; here is the one with the highest accuracy
		best_individual = final_archive[-1]
		best_individual_boolArray = np.array(best_individual.candidate, dtype=bool)

		# debugging
		#print("Best individual has fitness: fitness=%.2f, fitness=%.2f" % (X_train.shape[0] - best_individual.fitness[0], best_individual.fitness[1]))
		#print("Debugging: best individual has computed size:", len([x for x in best_individual.candidate if x == 1]))
		#print("X_train_pca has shape:", X_train_pca.shape)
		#X_train_pca_reduced = X_train_pca[best_individual_boolArray]
		#print("X_train_pca, reduced, has shape:", X_train_pca_reduced.shape)

		ax.plot(X_train_pca[best_individual_boolArray,0], X_train_pca[best_individual_boolArray,1], marker='D', color='red', fillstyle='none', linestyle='none', label="Core set")
		ax.set_title("Best individual, %d points, %.2f accuracy" % (X_train.shape[0] - best_individual.fitness[0], best_individual.fitness[1]))
		ax.set_xlabel("PCA dimension 1")
		ax.set_ylabel("PCA dimension 1")
		ax.legend(loc='best')

		plt.savefig( os.path.join(experiment_name, "best_individual.png") )
		plt.savefig( os.path.join(experiment_name, "best_individual.pdf") )
		plt.close(figure)
		
		print("Now saving points to CSV...")
		with open( os.path.join(experiment_name, "final_archive.csv"), "w") as fp :
			fp.write("#points,accuracy,individual\n")
			for f in final_archive :
				fp.write( str(X_train.shape[0] - f.fitness[0]) )
				fp.write( "," + str(f.fitness[1]) )
				
				for g in f.candidate :
					fp.write( "," + str(g) )
				
				fp.write("\n")

	return

# function that does most of the work
def evolveCoreSets(X, y, X_train, y_train, X_test, y_test, classifier, pop_size, offspring_size, max_generations, min_points_in_core_set, max_points_in_core_set, number_classes, maximize=True, seed=None, experiment_name=None, split="") :

	classifier_class = classifier[0]
	classifier_name = classifier[1]
	classifier_type = classifier[2]
	
	# a few checks on the arguments
	if seed == None : seed = int( time.time() )
	if experiment_name == None : 
		experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-ea-impair-" + classifier_name  
	elif split != "" : 
		experiment_name = experiment_name + "/" + classifier_name + "-split-" + split
	
	# create filename that will be later used to store whole population
	all_population_file = os.path.join(experiment_name, "all_individuals.csv") 

	# initialize classifier; some classifiers have random elements, and
	# for our purpose, we are working with a specific instance, so we fix
	# the classifier's behavior with a random seed
	if classifier_type == 1: classifier = classifier_class(random_state=seed) 
	else : classifier = classifier_class()
	
	# initialize pseudo-random number generation
	prng = random.Random()
	prng.seed(seed)

	print("Computing initial classifier performance...")
	referenceClassifier = copy.deepcopy(classifier)
	referenceClassifier.fit(X_train, y_train)
	y_train_pred = referenceClassifier.predict(X_train)
	y_test_pred = referenceClassifier.predict(X_test)
	y_pred = referenceClassifier.predict(X)
	trainAccuracy = accuracy_score(y_train, y_train_pred)
	testAccuracy = accuracy_score(y_test, y_test_pred)
	overallAccuracy = accuracy_score(y, y_pred)
	print("Initial performance: train=%.4f, test=%.4f, overall=%.4f" % (trainAccuracy, testAccuracy, overallAccuracy))

	print("\nSetting up evolutionary algorithm...")
	ea = inspyred.ec.emo.NSGA2(prng)
	ea.variator = [ variate ]
	ea.terminator = inspyred.ec.terminators.generation_termination
	ea.observer = observeCoreSets

	final_population = ea.evolve(    
					generator = generateCoreSets,
					evaluator = evaluateCoreSets,
					pop_size = pop_size,
					num_selected = offspring_size,
					maximize = maximize, 
					max_generations = max_generations,
					
					# extra arguments here
					n_classes = number_classes,
					classifier = classifier,
					X=X,
					y=y,
					X_train = X_train,
					y_train = y_train,
					X_test = X_test,
					y_test = y_test,
					min_points_in_core_set = min_points_in_core_set, 
					max_points_in_core_set = max_points_in_core_set,
					experimentName = experiment_name,
					all_population_file = all_population_file,
					current_time = datetime.datetime.now()
					)

	final_archive = sorted(ea.archive, key = lambda x : x.fitness[1])

	return final_archive

# utility function to load the covtype dataset
def loadForestCoverageType() :
	
	inputFile = "../data/covtype.csv"
	#print("Loading file \"" + inputFile + "\"...") 
	df_covtype = read_csv(inputFile, delimiter=',', header=None)
		
	# class is the last column
	covtype = df_covtype.as_matrix()
	X = covtype[:,:-1]
	y = covtype[:,-1].ravel()
	
	return X, y

# initial random generation of core sets (as binary strings)
def generateCoreSets(random, args) :

	individual_length = args["X_train"].shape[0]
	individual = [0] * individual_length
	
	points_in_core_set = random.randint( args["min_points_in_core_set"], args["max_points_in_core_set"] )
	for i in range(points_in_core_set) :
		random_index = random.randint(0, individual_length-1)
		individual[random_index] = 1
	
	return individual

# using inspyred's notation, here is a single operator that performs both
# crossover and mutation, sequentially
@inspyred.ec.variators.crossover
def variate(random, parent1, parent2, args) :
	
	# well, for starters we just crossover two individuals, then mutate
	children = [ list(parent1), list(parent2) ]
	
	# one-point crossover!
	cutPoint = random.randint(0, len(children[0])-1)
	for index in range(0, cutPoint+1) :
		temp = children[0][index]
		children[0][index] = children[1][index]
		children[1][index] = temp 
	
	# mutate!
	for child in children : 
		mutationPoint = random.randint(0, len(child)-1)
		if child[mutationPoint] == 0 :
			child[mutationPoint] = 1
		else :
			child[mutationPoint] = 0
	
	# check if individual is still valid, and (in case it isn't) repair it
	for child in children :
		
		if args.get("max_points_in_core_set", None) != None and args.get("min_points_in_core_set", None) != None :
			
			points_in_core_set = [ index for index, value in enumerate(child) if value == 1 ]
			
			while len(points_in_core_set) > args["max_points_in_core_set"] :
				index = random.choice( points_in_core_set )
				child[index] = 0
				points_in_core_set = [ index for index, value in enumerate(child) if value == 1 ]
			
			if len(points_in_core_set) < args["min_points_in_core_set"] :
				index = random.choice( [ index for index, value in enumerate(child) if value == 0 ] )
				child[index] = 1
				points_in_core_set = [ index for index, value in enumerate(child) if value == 1 ]
	
	return children

# function that evaluates the core sets
def evaluateCoreSets(candidates, args) :
	fitness = []

	for c in candidates :
		#print("candidate:", c)
		cAsBoolArray = np.array(c, dtype=bool)
		X_train_reduced = args["X_train"][cAsBoolArray,:]
		y_train_reduced = args["y_train"][cAsBoolArray]

		#print("Reduced training set:", X_train_reduced.shape[0])
		#print("Reduced training set:", y_train_reduced.shape[0])
		
		if len(set(y_train_reduced)) == args["n_classes"] :
			classifier = copy.deepcopy( args["classifier"] )
			classifier.fit(X_train_reduced, y_train_reduced)
			
			# evaluate accuracy for every point (training, test)
			y_pred_train = classifier.predict( args["X_train"] )
			#y_pred_test = classifier.predict( args["X_test"] )
			#y_pred = np.concatenate((y_pred_train, y_pred_test))
			#y = np.concatenate((args["y_train"], args["y_test"]))
			#accuracy = accuracy_score(y, y_pred)
			
			accuracy = accuracy_score(args["y_train"], y_pred_train)
			
			# also store valid individual; however, we only write down the two fitness values
			# because storing genome for ALL individuals would create files with hundreds of GBs
			all_population_file = args.get("all_population_file", None)
			if all_population_file != None :
				
				# if the file does not exist, write header
				if not os.path.exists(all_population_file) : 
					with open(all_population_file, "w") as fp :
						fp.write("#points,accuracy\n")
				
				# in any case, append individual
				with open(all_population_file, "a") as fp :
					fp.write( str(len([ x for x in c if x == 1])) )
					fp.write( "," + str(accuracy) )
					
					# we no longer store information about the individual
					#for g in c :
					#	fp.write( "," + str(g) )
					fp.write("\n")
		else:
			# individual gets a horrible fitness value
			maximize = args["_ec"].maximize # let's fetch the bool that tells us if we are maximizing or minimizing
			if maximize == True :
				accuracy = -np.inf
			else :
				accuracy = np.inf
				
		# maximizing the points removed also means minimizing the number of points taken (LOL)
		pointsRemoved = len([ x for x in c if x == 0])
		fitness.append( inspyred.ec.emo.Pareto( [pointsRemoved, accuracy] ) )
	
	return fitness

# the 'observer' function is called by inspyred algorithms at the end of every generation
def observeCoreSets(population, num_generations, num_evaluations, args) :
	
	training_set_size = args["X_train"].shape[0]
	old_time = args["current_time"]
	current_time = datetime.datetime.now()
	delta_time = current_time - old_time 
	
	# I don't like the 'timedelta' string format, so here is some fancy formatting
	delta_time_string = str(delta_time)[:-7] + "s"
	
	print("[%s] Generation %d, Random individual: size=%.2f, accuracy=%.2f" % (delta_time_string, num_generations, training_set_size - population[0].fitness[0], population[0].fitness[1]))
	
	args["current_time"] = current_time

	return

if __name__ == "__main__" :
	sys.exit( main() )
