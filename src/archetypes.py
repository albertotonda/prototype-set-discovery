# The evolutionary algorithm that we used in previous experiments is here wrapped up in a convenient function. Maybe.
# by Alberto Tonda, 2018 <alberto.tonda@gmail.com>

import copy
import datetime
import inspyred
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import time
import gc

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

from sklearn.model_selection import StratifiedKFold
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn import decomposition
from sklearn.cluster import KMeans
from scipy.spatial import distance

def generate(random, args) :
	
	X_train = args["X_train"]
	y_train = args["y_train"]
	nprandom = args["nprandom"]
	
	n_classes = len(np.unique(y_train))
	
	individual = []
	for c in range(0, n_classes):
			
		X_train_c = X_train[y_train==c]
		y_train_c = y_train[y_train==c]
		y_train_c = np.reshape(y_train_c, (len(y_train_c), 1))
		
		# take some samples from the class
		size = int( nprandom.randint(1, len(y_train_c)) ) # normal or uniform!?
#		size = int( nprandom.randint(1, 500) ) # normal or uniform!?
#		size = int(nprandom.normal(loc=len(y_train_c)/2, scale=len(y_train_c)/4))
#		if size < 1: size = 1
#		if size > len(y_train_c): size = len(y_train_c)
		
		sample_indeces = nprandom.choice(np.arange(0, len(y_train_c)), size=size, replace=False)
#		sample_indeces = nprandom.choice(np.arange(0, len(y_train_c)), size=size)
		
		sample = np.concatenate( (X_train_c[sample_indeces], y_train_c[sample_indeces]), axis=1 )
		
		if len(individual)==0:
			individual = sample
		else:
			individual = np.concatenate( (individual, sample), axis=0 )
		
#		for i in range(0, size):
#			sample_index = nprandom.randint(0, len(y_train_c))
#			sample = np.concatenate( (X_train_c[sample_index], np.array([y_train_c[sample_index]]) ) )
#			individual.append(sample)
#	
#	print("Individual generated with " + str(pointsRemoved) + " points removed:", individual)
#	
#	individual = np.array(individual)
	
	return individual.tolist()

@inspyred.ec.variators.crossover
def variate(random, parent1, parent2, args) :
	
	X_train = args["X_train"]
	y_train = args["y_train"]
	nprandom = args["nprandom"]
	std = 0.9999 * args["std"]

	classes = np.unique(y_train)
	
	parent_list = [ np.array(parent1.copy()), np.array(parent2.copy()) ]
	operator = nprandom.randint(0, 6)
	
	children = []
	
	if operator == 0:
		# change class
		
		for child in parent_list:
			
			# take random archetype
			sample_index = nprandom.randint(0, len(child))
			sample = child[sample_index]
			
			# take all classes except the old one
			class_choices = classes[classes != sample[-1]]
			new_class = nprandom.choice(class_choices)
			
			sample[-1] = new_class
			
			children.append(child.tolist())
		
	elif operator == 1:
		# change features
		
		for child in parent_list:
			
			# take random archetype
			sample_index = nprandom.randint(0, len(child))
			sample = child[sample_index]
			
			# generate normal random noise
			noise = nprandom.normal(0, std, len(sample)-1) # suppose features are scaled between (-1, 1)
			
			sample[:-1] = sample[:-1] + noise
			
			children.append(child.tolist())
			
		args["std"] = std
			
		
	elif operator == 2:
		# add sample
		
		for child in parent_list:
			
			# take random sample
			sample_index = nprandom.randint(0, len(y_train))
			sample = np.concatenate( (X_train[sample_index], np.array([y_train[sample_index]]) ) )
			
			child = np.concatenate( (child, np.array([sample])) )
			
			children.append(child.tolist())
		
	elif operator == 3:
		# remove sample
		
		for child in parent_list:
			
			if len(child) > len(np.unique(classes)):
				# take random sample
				sample_index = nprandom.randint(0, len(child))
				
				# remove sample
				child = np.delete(child, sample_index, axis=0)
			
			children.append(child.tolist())
		
	elif operator == 4:
		# crossover sample
		
		for child in parent_list:
			
			# take 2 random samples
			sample_index = nprandom.choice( np.arange(0, len(child)), size=2, replace=False )
			old_sample1, old_sample2 = child[sample_index[0]], child[sample_index[1]]
			
			# select random crossover position
			cross_pos = nprandom.randint(1, len(old_sample1)-1)
			
			# merge the two samples
			new_sample1 = np.concatenate(( np.array([ old_sample1[:cross_pos] ]), np.array([ old_sample2[cross_pos:] ]) ), axis=1)
			new_sample2 = np.concatenate(( np.array([ old_sample2[:cross_pos] ]), np.array([ old_sample1[cross_pos:] ]) ), axis=1)
			
			child[sample_index[0]] = new_sample1
			child[sample_index[1]] = new_sample2
			
			children.append(child.tolist())
		
	elif operator == 5:
		# crossover between archetypes
		
		# select random archetypes
		archetype_index1 = nprandom.randint(0, len(parent1))
		archetype_index2 = nprandom.randint(0, len(parent2))
	
		child1 = parent1
		child2 = parent2
		
		child1[archetype_index1] = parent2[archetype_index2]
		child2[archetype_index2] = parent1[archetype_index1]
		
		children.append(child1)
		children.append(child2)
			
	
	#print("Child created:", child1)
	return children

def evaluateCoreSets(candidates, args) :
	fitness = []
	
	X_train = args["X_train"]
	y_train = args["y_train"]
	n_features = X_train.shape[1]
	
	for c in candidates :
		c = np.array(c)
#		print("candidate:", c)
#		print(c.shape)
#		print(n_features)
		X_core = c[:, :n_features]
		y_core = c[:, -1]
		
		if len(set(y_core)) == args["n_classes"] :
			
			avg_accuracy = []
#			for rs in range(0, 3):
				
			classifier = copy.deepcopy( args["classifier"](random_state=42) )
			classifier.fit(X_core, y_core)
			
			# evaluate accuracy for every point (training, test)
#			y_pred_core = classifier.predict( X_core )
			y_pred_train = classifier.predict( X_train )
#			y_pred = np.concatenate((y_pred_core, y_pred_train))
#			y = np.concatenate((y_core, y_train))
			accuracy = accuracy_score(y_train, y_pred_train)
			avg_accuracy.append(accuracy)
					
			accuracy = np.mean(avg_accuracy)
			error = round(1-accuracy, 4)
			
#			# also store valid individual
#			all_population_file = args.get("all_population_file", None)
#			if all_population_file != None :
#				
#				# if the file does not exist, write header
#				if not os.path.exists(all_population_file) : 
#					with open(all_population_file, "w") as fp :
#						fp.write("#points,error,individual\n")
#				
#				# in any case, append individual
#				with open(all_population_file, "a") as fp :
#					fp.write( str( len(c) ) )
#					fp.write( "," + str(1-accuracy) )
#					
#					for g in c :
#						for e in g:
#							fp.write( "," + str(e) )
#					fp.write("\n")
					
					
		else:
			# individual gets a horrible fitness value
			maximize = args["_ec"].maximize # let's fetch the bool that tells us if we are maximizing or minimizing
			if maximize == True :
				error = -np.inf
			else :
				error = np.inf
				
		# maximizing the points removed also means minimizing the number of points taken (LOL)
		coreset_size = len(c)
		fitness.append( inspyred.ec.emo.Pareto( [coreset_size, error] ) )
		
	
	return fitness

def observeCoreSets(population, num_generations, num_evaluations, args) :
	
	gc.collect()
#	print("Generation %d; Random individual: size=%.2f, error=%.2f; std=%.5f" % (num_generations, population[0].fitness[0], population[0].fitness[1], args["std"] ))
	print("Generation %d, Number of best individuals: size=%.2f; std=%.5f" % ( num_generations, len(args["_ec"].archive), args["std"] ))

	return

# this part evolves core sets bottom-up, using a multi-objective approach
def evolveArchetypes(X, y, X_train, y_train, X_test, y_test, classifier, pop_size, offspring_size, max_generations, number_classes, maximize=False, seed=None, experiment_name=None) :

	classifier_class = classifier[0]
	classifier_name = classifier[1]
	classifier_type = classifier[2]
	
	# a few checks on the arguments
	# a few checks on the arguments
#	if seed == None : seed = int( time.time() )
#	if experiment_name == None : 
#		experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-ea-impair-" + classifier_name  
#	elif split != "" :
#		if db_name != None:
#			experiment_name = experiment_name + "/" + db_name + "-EvoImp-" + classifier_name + "-split-" + split
#			os.makedirs(experiment_name)
#		else: experiment_name = experiment_name + "/" + classifier_name + "-split-" + split

	# initialize classifier
	if classifier_type == 1: classifier = classifier_class(random_state=seed) 
	else : classifier = classifier_class()
	
	# initialize pseudo-random number generation
	prng = random.Random()
	prng.seed(seed)
	
	prng_np = np.random.RandomState(seed)

	# TODO maybe remove printouts? or make them optional?
#	print("Training set: %d lines; test set: %d lines" % (X_train.shape[0], X_test.shape[0]))

#	print("Compute initial classifier performance...")
	referenceClassifier = copy.deepcopy(classifier)
	referenceClassifier.fit(X_train, y_train)
	y_train_pred = referenceClassifier.predict(X_train)
	y_test_pred = referenceClassifier.predict(X_test)
#	y_pred = referenceClassifier.predict(X)
	trainAccuracy = accuracy_score(y_train, y_train_pred)
	testAccuracy = accuracy_score(y_test, y_test_pred)
#	overallAccuracy = accuracy_score(y, y_pred)
#	print("Initial performance: train=%.4f, test=%.4f, overall=%.4f" % (trainAccuracy, testAccuracy, overallAccuracy))
#	print("Initial performance: train=%.4f, test=%.4f" % (trainAccuracy, testAccuracy))

	print("\nSetting up evolutionary algorithm...")
	ea = inspyred.ec.emo.NSGA2(prng)
	ea.variator = [ variate ]
	ea.terminator = inspyred.ec.terminators.generation_termination
	ea.observer = observeCoreSets
	
	centroids = []
	for c in range(0, number_classes):
		centroids.append( np.mean(X_train[y_train==c], axis=0) )
	centroids = np.array(centroids)
	
	# create filename that will be later used to store whole population
	all_population_file = experiment_name + "/allIndividuals.csv"
	
	final_population = ea.evolve(    
					generator = generate,
					evaluator = evaluateCoreSets,
#					evaluator = inspyred.ec.evaluators.parallel_evaluation_mp,
#					mp_evaluator = evaluateCoreSets,
#					mp_num_cpus=4,
					num_selected = offspring_size,
					pop_size = pop_size,
					maximize = maximize, 
					max_generations = max_generations,
					std = 1,
					
					# extra arguments here
					n_classes = number_classes,
					classifier = classifier_class,
#					X=X,
#					y=y,
					X_train = X_train,
					y_train = y_train,
					X_test = X_test,
					y_test = y_test,
					experimentName = experiment_name,
					centroids = centroids,
					all_population_file = all_population_file,
					
					nprandom = prng_np,
					)

	final_archive = sorted(ea.archive, key = lambda x : x.fitness[1])
#	final_archive = sorted(ea.archive, key = lambda x : x.fitness[0] * x.fitness[1])

	return final_archive, trainAccuracy, testAccuracy


def make_meshgrid(x, y, h=.02):
	x_min, x_max = x.min() - 1, x.max() + 1
	y_min, y_max = y.min() - 1, y.max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						np.arange(y_min, y_max, h))
	return xx, yy

def plot_contours(clf, xx, yy, **params):
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	out = plt.contourf(xx, yy, Z, **params)
	return out, Z

def evaluate_core(X_core, y_core, X, y, classifier, cname=None, SEED=0):
	
	if cname == "SVC":
		referenceClassifier = copy.deepcopy(classifier(random_state=SEED, probability=True))
	else:
		referenceClassifier = copy.deepcopy(classifier(random_state=SEED))
	referenceClassifier.fit(X_core, y_core)
	y_pred = referenceClassifier.predict(X)
	
	fail_points = y != y_pred
	
	X_err = X[fail_points]
	accuracy = accuracy_score( y, y_pred)
	
	return X_err, accuracy, referenceClassifier, fail_points, y_pred

# this "main" here is kinda fake, just used to test the algorithm
def main() :
	
	N_SPLITS = 3
	SEED = 42
	POP_SIZE = 100
	MAX_GENERATIONS = 300
	
	classifier = [SVC, "SVC", 1]
	#classifier = [RandomForestClassifier, "RandomForestClassifier", 1]
	
	X, y = datasets.load_iris(True)
	X = X[:, 2:4]
	
	skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
	list_of_splits = [split for split in skf.split(X, y)]
	train_index, test_index = list_of_splits[0]
	X_train, y_train = X[train_index], y[train_index]
	X_test, y_test = X[test_index], y[test_index]
	
	final_archive = evolveCoreSets(X, y, X_train, y_train, X_test, y_test, 
						classifier, POP_SIZE, MAX_GENERATIONS, number_classes=len(np.unique(y)), 
						maximize=False, seed=SEED, experiment_name=None, split=str(SEED))
	
	pareto_front_x = [ f.fitness[0] for f in final_archive ]
	pareto_front_y = [ f.fitness[1] for f in final_archive ]
	
	figure = plt.figure()
	ax = figure.add_subplot(111)
	ax.plot(pareto_front_x, pareto_front_y, "bo-", label="Solutions in final archive")    
	ax.set_title("Final archive")
	ax.set_xlabel("Number of points in the core set")
	ax.set_ylabel("Performance")
	#plt.savefig( os.path.join(experiment_name, "final_archive.png") )
	#plt.savefig( os.path.join(experiment_name, "final_archive.pdf") )
	#plt.close(figure)
	
	print("Now saving all individuals generated during the evolution to a figure...")
	all_population_file_name = "allIndividuals.csv"
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
	ax.set_ylabel("Classification error")
	ax.legend(loc="best")
	plt.savefig( "all_population.png" ) 
	#plt.savefig( os.path.join(experiment_name, "all_population.pdf") )
	#plt.close(figure)
	
	#%%
	
	archive = [  ]
	individual = np.array(final_archive[0].candidate)
	n_features = X_train.shape[1]
	X_core, y_core = individual[:, :n_features], individual[:, -1]
	
	#individual = np.array(final_archive[165].candidate)
	#n_features = X_train.shape[1]
	#X_core, y_core = individual[:, :n_features], individual[:, -1]
	
	coreset_algorithm = "GEN"
	X_err, accuracy, model = evaluate_core(X_core, y_core, X_test, y_test, classifier[0], cname=classifier[1])
	cmap = plt.cm.jet
	xx, yy = make_meshgrid(X[:, 0], X[:, 1])
	plt.figure()
	_, Z_0 = plot_contours(model, xx, yy, cmap=cmap, alpha=0.2)
	plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, marker='s', alpha=0.4, label="train")
	plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, marker='+', alpha=0.4, label="test")
	plt.scatter(X_core[:, 0], X_core[:, 1], marker='D', facecolors='none', edgecolors='k', alpha=1, label="core set")
	plt.scatter(X_err[:, 0], X_err[:, 1], marker='x', facecolors='k', edgecolors='k', alpha=1, label="errors")
	plt.legend()
	plt.title("Iris Core Sets (acc. %.4f) - %s + %s" %(accuracy, coreset_algorithm, classifier[1]))
	plt.savefig("decision_boundaries_coreset_0.png" , dpi=1000)
	plt.show()

	return

if __name__ == "__main__" :
    sys.exit( main() )
