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

import random
import copy
import traceback

import inspyred
import datetime
import numpy as np
import multiprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")


class EvoCore(BaseEstimator, TransformerMixin):
    """
    Evocore class.
    """

    def __init__(self, estimator, pop_size: int = 100, max_generations: int = 100, max_points_in_core_set: int = None,
                 n_splits: int = 10, random_state: int = 42, scoring: str = "f1_weighted"):

        self.estimator = estimator
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.max_points_in_core_set = max_points_in_core_set
        self.n_splits = n_splits
        self.random_state = random_state
        self.scoring = scoring

    def fit(self, X, y=None, **fit_params):

        self.offspring_size_ = 2 * self.pop_size
        self.maximize_ = True
        self.individuals_ = []
        self.scorer_ = get_scorer(self.scoring)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.random_state)
        list_of_splits = [split for split in skf.split(X, y)]
        train_index, val_index = list_of_splits[0]

        self.x_train_, x_val = X.iloc[train_index], X.iloc[val_index]
        self.y_train_, y_val = y[train_index], y[val_index]

        self.n_classes_ = len(np.unique(self.y_train_))
        if self.max_points_in_core_set:
            self.max_points_in_core_set_ = np.min([self.max_points_in_core_set, self.x_train_.shape[0]])
        else:
            self.max_points_in_core_set_ = self.x_train_.shape[0]
        self.min_points_in_core_set_ = self.n_classes_

        # initialize pseudo-random number generation
        prng = random.Random()
        prng.seed(self.random_state)

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
            num_selected=self.offspring_size_,
            maximize=self.maximize_,
            max_generations=self.max_generations,

            # extra arguments here
            current_time=datetime.datetime.now()
        )

        # find best individual, the one with the highest accuracy on the validation set
        accuracy_best = 0
        for individual in ea.archive:

            core_set = train_index[individual.candidate]

            x_core = X.iloc[core_set]
            y_core = y[core_set]

            model = copy.deepcopy(self.estimator)
            model.fit(x_core, y_core)

            # compute validation accuracy
            accuracy_val = self.scorer_(model, x_val, y_val)

            if accuracy_best < accuracy_val:
                self.core_set_ = core_set
                self.x_core_ = x_core
                self.y_core_ = y_core
                accuracy_best = accuracy_val

        return self

    def transform(self, X, **fit_params):
        return (self.x_core_, self.y_core_)

    # initial random generation of core sets (as binary strings)
    def _generate_core_sets(self, random, args):

        points_in_core_set = random.randint(self.min_points_in_core_set_, self.max_points_in_core_set_)
        individual = np.random.choice(self.x_train_.shape[0], size=(points_in_core_set,), replace=False).tolist()
        individual = np.sort(individual).tolist()

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
            cut_point1 = random.randint(1, len(children[0])-1)
            cut_point2 = random.randint(1, len(children[1])-1)
            child1 = children[0][cut_point1:] + children[1][:cut_point2]
            child2 = children[1][cut_point2:] + children[0][:cut_point1]

            # remove duplicates
            # indexes = np.unique(child1, return_index=True)[1]
            # child1 = [child1[index] for index in sorted(indexes)]
            # indexes = np.unique(child2, return_index=True)[1]
            # child2 = [child2[index] for index in sorted(indexes)]
            child1 = np.unique(child1).tolist()
            child2 = np.unique(child2).tolist()

            children = [child1, child2]

            # mutate!
            for child in children:
                mutation_point = random.randint(0, len(child)-1)
                while True:
                    new_val = np.random.choice(self.x_train_.shape[0])
                    if new_val not in child:
                        child[mutation_point] = new_val
                        break

            # check if individual is still valid, and
            # (in case it isn't) repair it
            for child in children:

                # if it has too many core_sets, delete them
                if len(child) > self.max_points_in_core_set_:
                    n_surplus = len(child) - self.max_points_in_core_set_
                    indexes = np.random.choice(len(child), size=(n_surplus,))
                    child = np.delete(child, indexes).tolist()

                # if it has too less core_sets, add more
                if len(child) < self.min_points_in_core_set_:
                    n_surplus = self.min_points_in_core_set_ - len(child)
                    for _ in range(n_surplus):
                        while True:
                            new_val = np.random.choice(self.x_train_.shape[0])
                            if new_val not in child:
                                child.append(new_val)
                                break

            children[0] = np.sort(children[0]).tolist()
            children[1] = np.sort(children[1]).tolist()

            next_generation.append(children[0])
            next_generation.append(children[1])

        return next_generation

    # function that evaluates the core sets
    def _evaluate_core_sets(self, candidates, args):

        fitness = []

        for c in candidates:

            x_core = self.x_train_.iloc[c]
            y_core = self.y_train_[c]

            if np.unique(y_core).shape[0] == self.n_classes_:

                model = copy.deepcopy(self.estimator)
                model.fit(x_core, y_core)

                # compute train accuracy
                accuracy_train = self.scorer_(model, self.x_train_, self.y_train_)

                # compute numer of points outside the core_set
                points_removed = len(self.y_train_) - len(c)

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

        training_set_size = self.x_train_.shape[0]
        old_time = args["current_time"]
        # logger = args["logger"]
        current_time = datetime.datetime.now()
        delta_time = current_time - old_time

        # I don't like the 'timedelta' string format,
        # so here is some fancy formatting
        delta_time_string = str(delta_time)[:-7] + "s"

        log = "[%s] Generation %d, Random individual: size=%d, score=%.2f" \
            % (delta_time_string, num_generations,
               training_set_size - population[0].fitness[0],
               population[0].fitness[1])
        print(log)
        # if logger:
        #     logger.info(log)

        args["current_time"] = current_time
