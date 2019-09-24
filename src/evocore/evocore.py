# -*- coding: utf-8 -*-
#
# Copyright 2019 Alberto Tonda and Pietro Barbiero
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script has been designed to perform multi-objective
# learning of core sets.
# by Alberto Tonda and Pietro Barbiero, 2019
# <alberto.tonda@gmail.com> <pietro.barbiero@studenti.polito.it>

import os
import random
import copy
import inspyred
import datetime
import numpy as np
import traceback
import sys
import pandas as pd

from argparse import ArgumentParser

from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from .logging import initialize_logging, close_logging

LOG_DIR = "../log"
RESULTS_DIR = "../results"
VERSION = (1, 1)
#DEBUG = True
DEBUG = False


class EvoCore(object):
    """
    Evocore class.
    """

    def __init__(self, data_id=None,
                 dataset_name="iris",
                 classifier_name="RandomForestClassifier",
                 n_splits=10,
                 seed=42,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.version = VERSION
        self.logger = None

        self.seed = seed
        self.n_splits = n_splits

        self.data_id = data_id
        self.dataset_name = dataset_name
        self.classifier_name = classifier_name

        if DEBUG:
            self.max_points_in_core_set = 5
            self.min_points_in_core_set = 2
            self.pop_size = 4
            self.max_generations = 5
        else:
            self.max_points_in_core_set = None
            self.min_points_in_core_set = None
            self.pop_size = 100
            self.max_generations = 100

        self.offspring_size = None
        self.maximize = True

        self.X = None
        self.y = None

        self.X_train = None
        self.y_train = None

        self.classifiers = {
                "RandomForestClassifier": RandomForestClassifier,
                "BaggingClassifier": BaggingClassifier,
                "SVC": SVC,
                "LogisticRegression": LogisticRegression,
                "RidgeClassifier": RidgeClassifier,

                "AdaBoostClassifier": AdaBoostClassifier,
                "ExtraTreesClassifier": ExtraTreesClassifier,
                "GradientBoostingClassifier": GradientBoostingClassifier,
                "SGDClassifier": SGDClassifier,
                "PassiveAggressiveClassifier": PassiveAggressiveClassifier,
        }

        self.accuracy = {}
        self.accuracy["train_base"] = []
        self.accuracy["test_base"] = []
        self.accuracy["core_cv"] = []
        self.accuracy["train_cv"] = []
        self.accuracy["test_cv"] = []
        self.accuracy["train_blind"] = None
        self.accuracy["test_blind"] = None

        self.individuals = []
        self.coreset = []

        self.results = None

#    def run_cv(self):
#        self._setup()
#
#        for dataset in self.selected_dataset:
#            for classifier in self.selected_classifiers:
#                self._run_evocore_cv(dataset, classifier)

    def run_cv(self):

        self._setup()
        self._start_logging()

        self._load_openML_dataset()

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.seed)

        split_index = 0
        for train_index, test_index in skf.split(self.X, self.y):

            # unlocking random seed!
            self.seed = split_index

            self.logger.info("Split %d" % (split_index))

            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            coreset = self.fit(X_train, y_train)

            # extract coreset indeces wth resplect to
            # the whole dataset
            self.coreset.append(train_index[coreset])
            X_core, y_core = X_train[coreset, :], y_train[coreset]

            # save results
            self.accuracy["core_cv"].append(self.score(X_core, y_core))
            self.accuracy["train_cv"].append(self.score(X_train, y_train))
            self.accuracy["test_cv"].append(self.score(X_test, y_test))
            self.accuracy["train_base"].append(
                    self._baseline_accuracy(X_train, y_train))
            self.accuracy["test_base"].append(
                    self._baseline_accuracy(X_test, y_test))

            split_index += 1

        self.save_results(save=True)

        close_logging(self.logger)

    def fit(self, X_train, y_train):

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.seed)
        listOfSplits = [split for split in skf.split(X_train, y_train)]
        train_index, val_index = listOfSplits[0]

        self.X_train, X_val = X_train[train_index], X_train[val_index]
        self.y_train, y_val = y_train[train_index], y_train[val_index]

        # initialize pseudo-random number generation
        prng = random.Random()
        prng.seed(self.seed)

        self.ea = inspyred.ec.emo.NSGA2(prng)
        self.ea.variator = [self._variate]
        self.ea.terminator = inspyred.ec.terminators.generation_termination
        self.ea.observer = self._observe_core_sets

        self.ea.evolve(
                generator=self._generate_core_sets,
                evaluator=self._evaluate_core_sets,

                pop_size=self.pop_size,
                num_selected=self.offspring_size,
                maximize=self.maximize,
                max_generations=self.max_generations,

                # extra arguments here
                current_time=datetime.datetime.now()
        )

        # find best individual, the one with the highest accuracy
        # on the training set
        accuracy_best = 0
        coreset_best = None
        for individual in self.ea.archive:
            c_bool = np.array(individual.candidate, dtype=bool)

            coreset = train_index[c_bool]

            X_core = X_train[coreset, :]
            y_core = y_train[coreset]

            model = copy.deepcopy(self.classifier)
            model.fit(X_core, y_core)

            # compute validation accuracy
            accuracy_val = model.score(X_val, y_val)

            if accuracy_best < accuracy_val:
                coreset_best = coreset
                self.model = model
                accuracy_best = accuracy_val

        return coreset_best

    def score(self, X, y):
        return self.model.score(X, y)

    def save_results(self, save=False):
        """
        TODO: comment
        """

        columns = [
                "version",
                "data_id",
                "dataset_name",
                "classifier_name",
                "seed",
                "n_splits",
                "max_points_in_core_set",
                "min_points_in_core_set",
                "pop_size",
                "offspring_size",
                "max_generations",

                "train_blind",
                "test_blind",
                "train_baseline",
                "test_baseline",

                "core_cv",
                "train_cv",
                "test_cv",

                "coreset_size",
                "coreset",
        ]

        df = pd.DataFrame(columns=columns)

        default_row = [
                self.version,
                self.data_id,
                self.dataset_name,
                self.classifier_name,
                self.seed,
                self.n_splits,
                self.max_points_in_core_set,
                self.min_points_in_core_set,
                self.pop_size,
                self.offspring_size,
                self.max_generations,

                self.accuracy["train_blind"],
                self.accuracy["test_blind"],
        ]

        for i in range(0, self.n_splits-1):

            row = []
            row.extend(default_row)

            row.append(self.accuracy["train_base"][i])
            row.append(self.accuracy["test_base"][i])

            row.append(self.accuracy["core_cv"][i])
            row.append(self.accuracy["train_cv"][i])
            row.append(self.accuracy["test_cv"][i])

            row.append(len(self.coreset[i]))
            row.append(self.coreset[i])

            df = df.append(pd.DataFrame([row], columns=columns),
                           ignore_index=True)

        self.results = df

        if save:
            experiment = self.dataset_name + "_" + \
                self.classifier_name + "_v" + \
                str(self.version[0]) + "_" + str(self.version[1]) + \
                ".csv"
            df.to_csv(os.path.join(RESULTS_DIR, experiment))

        return

    def _baseline_accuracy(self, X, y):
        # compute baseline: accuracy using all samples
        model = copy.deepcopy(self.classifier)
        model.fit(self.X_train, self.y_train)
        return model.score(X, y)

    def _start_logging(self):

        log_name = self.dataset_name + "_" + self.classifier_name
        self.logger = initialize_logging(log_name)

    # initial random generation of core sets (as binary strings)
    def _generate_core_sets(self, random, args):

        individual_length = self.X_train.shape[0]
        individual = [0] * individual_length

        points_in_core_set = random.randint(self.min_points_in_core_set,
                                            self.max_points_in_core_set)
        for i in range(points_in_core_set):
            random_index = random.randint(0, individual_length-1)
            individual[random_index] = 1

        return individual

    # using inspyred's notation, here is a single operator that performs both
    # crossover and mutation, sequentially
    def _variate(self, random, candidates, args):

        split_idx = int(len(candidates) / 2)
        fathers = candidates[:split_idx]
        mothers = candidates[split_idx:]

        next_generation = []

        for parent1, parent2 in zip(fathers, mothers):

            # well, for starters we just crossover two individuals, then mutate
            children = [list(parent1), list(parent2)]

            # one-point crossover!
            cutPoint = random.randint(0, len(children[0])-1)
            for index in range(0, cutPoint+1):
                temp = children[0][index]
                children[0][index] = children[1][index]
                children[1][index] = temp

            # mutate!
            for child in children:
                mutationPoint = random.randint(0, len(child)-1)
                if child[mutationPoint] == 0:
                    child[mutationPoint] = 1
                else:
                    child[mutationPoint] = 0

            # check if individual is still valid, and
            # (in case it isn't) repair it
            for child in children:

                points_in_core_set = self._points_in_core_set(child)

                # if it has too many coresets, delete one
                while len(points_in_core_set) > self.max_points_in_core_set:
                    index = random.choice(points_in_core_set)
                    child[index] = 0
                    points_in_core_set = self._points_in_core_set(child)

                # if it has too less coresets, add one
                if len(points_in_core_set) < self.min_points_in_core_set:
                    index = random.choice(points_in_core_set)
                    child[index] = 1
                    points_in_core_set = self._points_in_core_set(child)

            next_generation.append(children[0])
            next_generation.append(children[1])

        return next_generation

    def _points_in_core_set(self, individual):
        points_in_core_set = []
        for index, value in enumerate(individual):
            if value == 1:
                points_in_core_set.append(index)
        return points_in_core_set

    # function that evaluates the core sets
    def _evaluate_core_sets(self, candidates, args):

        fitness = []

        for c in candidates:

            c_bool = np.array(c, dtype=bool)

            X_core = self.X_train[c_bool, :]
            y_core = self.y_train[c_bool]

            if np.unique(y_core).shape[0] == self.n_classes:

                model = copy.deepcopy(self.classifier)
                model.fit(X_core, y_core)

                # compute train accuracy
                accuracy_train = model.score(self.X_train, self.y_train)

                # compute numer of points outside the coreset
                points_removed = sum(1-c_bool)

#                # compute core accuracy
#                accuracy_core = model.score(X_core, y_core)
#
#                coreset_size = sum(c_bool)
#                # save individual
#                self.individuals.append([
#                        coreset_size,
#                        accuracy_core,
#                        accuracy_train,
#                ])

            else:
                # individual gets a horrible fitness value
                maximize = args["_ec"].maximize
                if maximize is True:
                    accuracy_train = -np.inf
                else:
                    accuracy_train = np.inf
                points_removed = 0

            # maximizing the points removed also means
            # minimizing the number of points taken (LOL)
            fitness.append(inspyred.ec.emo.Pareto([
                    points_removed,
                    accuracy_train,
            ]))

        return fitness

    # the 'observer' function is called by inspyred algorithms
    # at the end of every generation
    def _observe_core_sets(self, population, num_generations,
                           num_evaluations, args):

        training_set_size = self.X_train.shape[0]
        old_time = args["current_time"]
        current_time = datetime.datetime.now()
        delta_time = current_time - old_time

        # I don't like the 'timedelta' string format,
        # so here is some fancy formatting
        delta_time_string = str(delta_time)[:-7] + "s"

        log = "[%s] Generation %d, Random individual: size=%d, accuracy=%.2f" \
            % (delta_time_string, num_generations,
               training_set_size - population[0].fitness[0],
               population[0].fitness[1])
        self.logger.info(log)

        args["current_time"] = current_time

    def _setup(self):
        # argparse; all arguments are optional
        p = ArgumentParser()

        p.add_argument("--classifier_name", "-c",
                       help="Classifier(s) to be tested. Default: %s."
                       % (self.classifier_name))
        p.add_argument("--dataset_name", "-d",
                       help="Dataset to be tested. Default: %s."
                       % (self.dataset_name))

        p.add_argument("--pop_size", "-p", type=int,
                       help="EA population size. Default: %d"
                       % (self.pop_size))
#        p.add_argument("--offspring_size", "-o", type=int,
#                       help="Ea offspring size. Default: %d"
#                       % (self.offspring_size))
        p.add_argument("--max_generations", "-mg", type=int,
                       help="Maximum number of generations." +
                       "Default: %d" % (self.max_generations))

#        p.add_argument("--min_points", "-mip", type=int,
#                       help="Minimum number of points in the core set." +
#                       "Default: %d" % (self.min_points_in_core_set))
#        p.add_argument("--max_points", "-mxp", type=int,
#                       help="Maximum number of points in the core set." +
#                       "Default: %d" % (self.max_points_in_core_set))

        # finally, parse the arguments
        args = p.parse_args()

        if args.dataset_name:
            self.dataset_name = args.dataset_name
        if args.classifier_name:
            self.classifier_name = args.classifier_name
        if args.max_generations:
            self.max_generations = args.max_generations
        if args.pop_size:
            self.pop_size = args.pop_size

        self.offspring_size = 2 * self.pop_size

        # initialize classifier; some classifiers have random elements, and
        # for our purpose, we are working with a specific instance, so we fix
        # the classifier's behavior with a random seed
        classifier_class = self.classifiers[self.classifier_name]
        self.classifier = classifier_class(random_state=self.seed)

        if _is_odd(self.pop_size):
            self.pop_size += 1

        if not os.path.isdir(LOG_DIR):
            os.makedirs(LOG_DIR)
        if not os.path.isdir(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

    def _load_openML_dataset(self):

        try:
            if self.data_id is not None:
                self.X, self.y = fetch_openml(data_id=self.data_id,
                                              return_X_y=True)
            else:
                self.X, self.y = fetch_openml(name=self.dataset_name,
                                              return_X_y=True)

            if not isinstance(self.X, np.ndarray):
                self.X = self.X.toarray()

            si = SimpleImputer(missing_values=np.nan, strategy='mean')
            self.X = si.fit_transform(self.X)

            le = LabelEncoder()
            self.y = le.fit_transform(self.y)
            self.n_classes = np.unique(self.y).shape[0]

            self.max_points_in_core_set = self.X.shape[0]
            self.min_points_in_core_set = self.n_classes

        except Exception:
            self.logger.error(traceback.format_exc())
            sys.exit()


def _is_odd(num):
    return num % 2 != 0
