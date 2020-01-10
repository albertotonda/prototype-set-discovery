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

import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")


def _is_survivor(individual, survivors):
    for s in survivors:
        if np.all(individual.candidate.indices == s.candidate.indices):
            return True
    return False


class EvoCore(BaseEstimator, TransformerMixin):
    """
    Evocore class.
    """

    def __init__(self, estimator, pop_size: int = 100, max_generations: int = 100,
                 n_splits: int = 10, random_state: int = 42, scoring: str = "f1_weighted"):

        self.estimator = estimator
        self.pop_size = pop_size
        self.max_generations = max_generations
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
        self.max_points_in_core_set_ = self.x_train_.shape[0]
        self.min_points_in_core_set_ = self.n_classes_

        # initialize pseudo-random number generation
        prng = random.Random()
        prng.seed(self.random_state)

        ea = inspyred.ec.emo.NSGA2(prng)
        ea.variator = [self._variate]
        ea.terminator = inspyred.ec.terminators.generation_termination
        ea.observer = self._observe_core_sets
        ea.archiver = self._best_archiver
        ea.replacer = self._nsga_replacement

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

            core_set = train_index[individual.candidate.indices]

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

        individual_length = self.x_train_.shape[0]
        individual = [0] * individual_length

        points_in_core_set = random.randint(self.min_points_in_core_set_,
                                            self.max_points_in_core_set_)
        for i in range(points_in_core_set):
            random_index = random.randint(0, individual_length-1)
            individual[random_index] = 1

        return scipy.sparse.csr_matrix(individual)

    # using inspyred's notation, here is a single operator that performs both crossover and mutation, sequentially
    def _variate(self, random, candidates, args):

        split_idx = int(len(candidates) / 2)
        fathers = candidates[:split_idx]
        mothers = candidates[split_idx:]

        next_generation = []

        for parent1, parent2 in zip(fathers, mothers):

            # well, for starters we just crossover two individuals, then mutate
            children = [parent1, parent2]

            # one-point crossover!
            cut_point = random.randint(0, children[0].shape[1]-1)
            child1 = scipy.sparse.hstack([children[0][:, :cut_point], children[1][:, cut_point:]], format="csr")
            child2 = scipy.sparse.hstack([children[1][:, :cut_point], children[0][:, cut_point:]], format="csr")

            children = [child1, child2]

            # mutate!
            for child in children:
                mutation_point = random.randint(0, child.shape[1]-1)
                if mutation_point not in child.indices:
                    child[:, mutation_point] = 1
                else:
                    child[:, mutation_point] = 0

            # check if individual is still valid, and
            # (in case it isn't) repair it
            for child in children:

                # if it has too many core_sets, delete one
                while len(child.indices) > self.max_points_in_core_set_:
                    index = random.choice(self.y_train_.shape[0])
                    child[:, index] = 0

                # if it has too less core_sets, add one
                if len(child.indices) < self.min_points_in_core_set_:
                    index = random.choice(self.y_train_.shape[0])
                    child[:, index] = 1

            next_generation.append(children[0])
            next_generation.append(children[1])

        return next_generation

    # function that evaluates the core sets
    def _evaluate_core_sets(self, candidates, args):

        fitness = []

        for c in candidates:

            x_core = self.x_train_.iloc[c.indices]
            y_core = self.y_train_[c.indices]

            if np.unique(y_core).shape[0] == self.n_classes_:

                model = copy.deepcopy(self.estimator)
                model.fit(x_core, y_core)

                # compute train accuracy
                accuracy_train = self.scorer_(model, self.x_train_, self.y_train_)

                # compute numer of points outside the core_set
                points_removed = self.y_train_.shape[0] - y_core.shape[0]

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

    def _best_archiver(self, random, population, archive, args):
        """Archive only the best individual(s).

        This function archives the best solutions and removes inferior ones.
        If the comparison operators have been overloaded to define Pareto
        preference (as in the ``Pareto`` class), then this archiver will form
        a Pareto archive.

        .. Arguments:
           random -- the random number generator object
           population -- the population of individuals
           archive -- the current archive of individuals
           args -- a dictionary of keyword arguments

        """
        new_archive = archive
        for ind in population:
            if len(new_archive) == 0:
                new_archive.append(ind)
            else:
                should_remove = []
                should_add = True
                for a in new_archive:
                    if np.all(ind.candidate.indices == a.candidate.indices):
                        should_add = False
                        break
                    elif ind < a:
                        should_add = False
                    elif ind > a:
                        should_remove.append(a)
                for r in should_remove:
                    for pos, item in enumerate(new_archive):
                        if np.all(r.candidate.indices == item.candidate.indices):
                            del new_archive[pos]
                if should_add:
                    new_archive.append(ind)
        return new_archive

    def _nsga_replacement(self, random, population, parents, offspring, args):
        """Replaces population using the non-dominated sorting technique from NSGA-II.

        .. Arguments:
           random -- the random number generator object
           population -- the population of individuals
           parents -- the list of parent individuals
           offspring -- the list of offspring individuals
           args -- a dictionary of keyword arguments

        """
        survivors = []
        combined = list(population)
        combined.extend(offspring)

        # Perform the non-dominated sorting to determine the fronts.
        fronts = []
        pop = set(range(len(combined)))
        while len(pop) > 0:
            front = []
            for p in pop:
                dominated = False
                for q in pop:
                    if combined[p] < combined[q]:
                        dominated = True
                        break
                if not dominated:
                    front.append(p)
            fronts.append([dict(individual=combined[f], index=f) for f in front])
            pop = pop - set(front)

        # Go through each front and add all the elements until doing so
        # would put you above the population limit. At that point, fall
        # back to the crowding distance to determine who to put into the
        # next population. Individuals with higher crowding distances
        # (i.e., more distance between neighbors) are preferred.
        for i, front in enumerate(fronts):
            if len(survivors) + len(front) > len(population):
                # Determine the crowding distance.
                distance = [0 for _ in range(len(combined))]
                individuals = list(front)
                num_individuals = len(individuals)
                num_objectives = len(individuals[0]['individual'].fitness)
                for obj in range(num_objectives):
                    individuals.sort(key=lambda x: x['individual'].fitness[obj])
                    distance[individuals[0]['index']] = float('inf')
                    distance[individuals[-1]['index']] = float('inf')
                    for i in range(1, num_individuals - 1):
                        distance[individuals[i]['index']] = (distance[individuals[i]['index']] +
                                                             (individuals[i + 1]['individual'].fitness[obj] -
                                                              individuals[i - 1]['individual'].fitness[obj]))

                crowd = [dict(dist=distance[f['index']], index=f['index']) for f in front]
                crowd.sort(key=lambda x: x['dist'], reverse=True)
                last_rank = [combined[c['index']] for c in crowd]
                r = 0
                num_added = 0
                num_left_to_add = len(population) - len(survivors)
                while r < len(last_rank) and num_added < num_left_to_add:
                    if not _is_survivor(last_rank[r], survivors):
                        survivors.append(last_rank[r])
                        num_added += 1
                    r += 1
                # If we've filled out our survivor list, then stop.
                # Otherwise, process the next front in the list.
                if len(survivors) == len(population):
                    break
            else:
                for f in front:
                    if not _is_survivor(f['individual'], survivors):
                        survivors.append(f['individual'])
        return survivors

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

        log = "[%s] Generation %d, Random individual: size=%d, accuracy=%.2f" \
            % (delta_time_string, num_generations,
               training_set_size - population[0].fitness[0],
               population[0].fitness[1])
        print(log)
        # if logger:
        #     logger.info(log)

        args["current_time"] = current_time
