# -*- coding: utf-8 -*-
#
# Copyright 2019 Pietro Barbiero and Alberto Tonda
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
import sys
import pandas as pd
import multiprocessing
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedKFold

LOG_DIR = "../log"
RESULTS_DIR = "../results"
VERSION = (1, 2)
# DEBUG = True
DEBUG = False

sys.path.insert(0, '../cross_validator')
from cross_validator import CrossValidator


class EvoCore(CrossValidator):
    """
    Evocore class.
    """

    def __init__(self, pop_size=100, max_generations=100, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.name = "evocore"

        self.log_dir = LOG_DIR
        self.results_dir = RESULTS_DIR
        self.debug = DEBUG
        self.version = VERSION

        if self.debug:
            self.max_points_in_coreset = 5
            self.min_points_in_coreset = 2
            self.pop_size = 4
            self.max_generations = 2
        else:
            self.max_points_in_coreset = None
            self.min_points_in_coreset = None
            self.pop_size = pop_size
            self.max_generations = max_generations

        self.offspring_size = None
        self.maximize = True

        self.individuals = []
        self.coreset = []

    def run(self):
        self.run_cv()

    def fit(self, X_train, y_train):

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.seed)
        listOfSplits = [split for split in skf.split(X_train, y_train)]
        train_index, val_index = listOfSplits[0]

        self.X_train, X_val = X_train[train_index], X_train[val_index]
        self.y_train, y_val = y_train[train_index], y_train[val_index]

        self.max_points_in_coreset = self.X_train.shape[0]
        self.min_points_in_coreset = self.n_classes

        # initialize pseudo-random number generation
        prng = random.Random()
        prng.seed(self.seed)

        self.ea = inspyred.ec.emo.NSGA2(prng)
        self.ea.variator = [self._variate]
        self.ea.terminator = inspyred.ec.terminators.generation_termination
        self.ea.observer = self._observe_coresets

        self.ea.evolve(
                generator=self._generate_coresets,

                # evaluator=self._evaluate_coresets,
                # this part is defined to use multi-process evaluations
                evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,
                mp_evaluator=self._evaluate_coresets,
                mp_num_cpus=multiprocessing.cpu_count()-2,

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

        self._update_results(coreset_best)

        return

    def _update_results(self, coreset):

        X_train = self.X[self.train_index]
        y_train = self.y[self.train_index]

        # extract coreset indeces with respect to
        # the whole dataset
        self.coreset.append(self.train_index[coreset])
        X_core, y_core = X_train[coreset, :], y_train[coreset]

        # save results
        self.accuracy["core_cv"].append(self.score(X_core, y_core))

    def results_extra_arguments(self):
        """
        TODO: comment
        """

        columns = [
                "max_points_in_coreset",
                "min_points_in_coreset",
                "pop_size",
                "offspring_size",
                "max_generations",

                "core_cv",
                "coreset_size",
                "coreset",
        ]

        df = pd.DataFrame(columns=columns)

        default_row = [
                self.max_points_in_coreset,
                self.min_points_in_coreset,
                self.pop_size,
                self.offspring_size,
                self.max_generations,
        ]

        for i in range(0, self.n_splits):

            row = []
            row.extend(default_row)

            row.append(self.accuracy["core_cv"][i])
            row.append(len(self.coreset[i]))
            row.append(self.coreset[i])

            df = df.append(pd.DataFrame([row], columns=columns),
                           ignore_index=True)

        self.results = pd.concat([self.results, df], axis=1)

    # initial random generation of core sets (as binary strings)
    def _generate_coresets(self, random, args):

        individual_length = self.X_train.shape[0]
        individual = [0] * individual_length

        points_in_coreset = random.randint(self.min_points_in_coreset,
                                           self.max_points_in_coreset)
        for i in range(points_in_coreset):
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

                points_in_coreset = self._points_in_coreset(child)

                # if it has too many coresets, delete one
                while len(points_in_coreset) > self.max_points_in_coreset:
                    index = random.choice(points_in_coreset)
                    child[index] = 0
                    points_in_coreset = self._points_in_coreset(child)

                # if it has too less coresets, add one
                if len(points_in_coreset) < self.min_points_in_coreset:
                    index = random.choice(points_in_coreset)
                    child[index] = 1
                    points_in_coreset = self._points_in_coreset(child)

            next_generation.append(children[0])
            next_generation.append(children[1])

        return next_generation

    def _points_in_coreset(self, individual):
        points_in_coreset = []
        for index, value in enumerate(individual):
            if value == 1:
                points_in_coreset.append(index)
        return points_in_coreset

    # function that evaluates the core sets
    def _evaluate_coresets(self, candidates, args):

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
    def _observe_coresets(self, population, num_generations,
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

    def setup(self):
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
#                       "Default: %d" % (self.min_points_in_coreset))
#        p.add_argument("--max_points", "-mxp", type=int,
#                       help="Maximum number of points in the core set." +
#                       "Default: %d" % (self.max_points_in_coreset))

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

        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)


def _is_odd(num):
    return num % 2 != 0
