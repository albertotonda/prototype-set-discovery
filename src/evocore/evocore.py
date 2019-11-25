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

import random
import copy
import inspyred
import datetime
import numpy as np
import multiprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")


class EvoCore(BaseEstimator, TransformerMixin):
    """
    Evocore class.
    """

    def __init__(self, estimator, pop_size=100, max_generations=100, n_splits=10, seed=42):

        self.__name__ = "evocore"
        self.__version__ = "0.0.0"

        self.n_splits = n_splits
        self.seed = seed

        self.max_points_in_core_set = None
        self.min_points_in_core_set = 0

        self.estimator = estimator
        self.x_train = None
        self.y_train = None
        self.n_classes = None

        self.pop_size = pop_size
        self.max_generations = max_generations
        self.offspring_size = 2*pop_size
        self.maximize = True

        self.individuals = []
        self.core_set = []
        self.x_core = None
        self.y_core = None

    def __repr__(self, N_CHAR_MAX=700):
        return (f'{self.__class__.__name__}('
                f'{self.pop_size!r}, {self.max_generations!r})')

    def fit(self, X, y=None, **fit_params):
        logger = fit_params.get("logger")

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.seed)
        list_of_splits = [split for split in skf.split(X, y)]
        train_index, val_index = list_of_splits[0]

        self.x_train, x_val = X[train_index], X[val_index]
        self.y_train, y_val = y[train_index], y[val_index]

        self.n_classes = len(np.unique(self.y_train))
        self.max_points_in_core_set = self.x_train.shape[0]
        self.min_points_in_core_set = self.n_classes

        # initialize pseudo-random number generation
        prng = random.Random()
        prng.seed(self.seed)

        ea = inspyred.ec.emo.NSGA2(prng)
        ea.variator = [self._variate]
        ea.terminator = inspyred.ec.terminators.generation_termination
        ea.observer = self._observe_core_sets

        ea.evolve(
            generator=self._generate_core_sets,

            evaluator=self._evaluate_core_sets,
            # this part is defined to use multi-process evaluations
            # evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,
            # mp_evaluator=self._evaluate_core_sets,
            # mp_num_cpus=multiprocessing.cpu_count()-2,

            pop_size=self.pop_size,
            num_selected=self.offspring_size,
            maximize=self.maximize,
            max_generations=self.max_generations,
            logger=logger,

            # extra arguments here
            current_time=datetime.datetime.now()
        )

        # find best individual, the one with the highest accuracy on the validation set
        accuracy_best = 0
        for individual in ea.archive:
            c_bool = np.array(individual.candidate, dtype=bool)

            core_set = train_index[c_bool]

            x_core = X[core_set, :]
            y_core = y[core_set]

            model = copy.deepcopy(self.estimator)
            model.fit(x_core, y_core)

            # compute validation accuracy
            accuracy_val = model.score(x_val, y_val)

            if accuracy_best < accuracy_val:
                self.core_set = core_set
                self.x_core = x_core
                self.y_core = y_core
                accuracy_best = accuracy_val

        return self.x_core, self.y_core

    def transform(self, X, **fit_params):
        return self.x_core, self.y_core

    # initial random generation of core sets (as binary strings)
    def _generate_core_sets(self, random, args):

        individual_length = self.x_train.shape[0]
        individual = [0] * individual_length

        points_in_core_set = random.randint(self.min_points_in_core_set,
                                           self.max_points_in_core_set)
        for i in range(points_in_core_set):
            random_index = random.randint(0, individual_length-1)
            individual[random_index] = 1

        return individual

    # using inspyred's notation, here is a single operator that performs both crossover and mutation, sequentially
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

                points_in_core_set = _points_in_core_set(child)

                # if it has too many core_sets, delete one
                while len(points_in_core_set) > self.max_points_in_core_set:
                    index = random.choice(points_in_core_set)
                    child[index] = 0
                    points_in_core_set = _points_in_core_set(child)

                # if it has too less core_sets, add one
                if len(points_in_core_set) < self.min_points_in_core_set:
                    index = random.choice(points_in_core_set)
                    child[index] = 1
                    points_in_core_set = _points_in_core_set(child)

            next_generation.append(children[0])
            next_generation.append(children[1])

        return next_generation

    # function that evaluates the core sets
    def _evaluate_core_sets(self, candidates, args):

        fitness = []

        for c in candidates:

            c_bool = np.array(c, dtype=bool)

            x_core = self.x_train[c_bool, :]
            y_core = self.y_train[c_bool]

            if np.unique(y_core).shape[0] == self.n_classes:

                model = copy.deepcopy(self.estimator)
                model.fit(x_core, y_core)

                # compute train accuracy
                accuracy_train = model.score(self.x_train, self.y_train)

                # compute numer of points outside the core_set
                points_removed = sum(1-c_bool)

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

    # the 'observer' function is called by inspyred algorithms at the end of every generation
    def _observe_core_sets(self, population, num_generations, num_evaluations, args):

        training_set_size = self.x_train.shape[0]
        old_time = args["current_time"]
        logger = args["logger"]
        current_time = datetime.datetime.now()
        delta_time = current_time - old_time

        # I don't like the 'timedelta' string format,
        # so here is some fancy formatting
        delta_time_string = str(delta_time)[:-7] + "s"

        log = "[%s] Generation %d, Random individual: size=%d, accuracy=%.2f" \
            % (delta_time_string, num_generations,
               training_set_size - population[0].fitness[0],
               population[0].fitness[1])
        if logger:
            logger.info(log)

        args["current_time"] = current_time


def _points_in_core_set(individual):
    points_in_core_set = []
    for index, value in enumerate(individual):
        if value == 1:
            points_in_core_set.append(index)
    return points_in_core_set
