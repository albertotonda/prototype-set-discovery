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
import bayesiancoresets as bc
from argparse import ArgumentParser

LOG_DIR = "../log"
RESULTS_DIR = "../results"
VERSION = (1, 0)

sys.path.insert(0, '../cross_validator')
from cross_validator import CrossValidator


class BayesCore(CrossValidator):
    """
    Evocore class.
    """

    def __init__(self, algorithms, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.name = None

        self.log_dir = LOG_DIR
        self.results_dir = RESULTS_DIR
        self.version = VERSION

        self.algorithms = {
                "GIGA": bc.GIGA,
                "FrankWolfe": bc.FrankWolfe,
                "MatchingPursuit": bc.MatchingPursuit,
                "ForwardStagewise": bc.ForwardStagewise,
                "OrthoPursuit": bc.OrthoPursuit,
                "LAR": bc.LAR,
                "ImportanceSampling": bc.ImportanceSampling,
                "RandomSubsampling": bc.RandomSubsampling,
        }

        if type(algorithms) is str:
            self.algorithm_names = [algorithms]
        elif type(algorithms) is list:
            self.algorithm_names = algorithms

        self.coreset = []

    def run(self):
        for algorithm_name in self.algorithm_names:
            self.name = algorithm_name
            self.algorithm = self.algorithms[self.name]
            self.run_cv()

    def fit(self, X_train, y_train):

        self.logger.info("Coreset discovery using %s" % (self.name))

        self.X_train, self.y_train = X_train, y_train

        n_trials = 1
        Ms = np.unique(np.logspace(0., 4., 100, dtype=np.int32))
        wts = []
        for tr in range(n_trials):
            alg = self.algorithm(X_train)
            for m, M in enumerate(Ms):
                alg.run(M)
                wts = alg.weights()
        coreset = wts > 0
        coreset = self.missing_class_fix(coreset, y_train)

        X_core = X_train[coreset, :]
        y_core = y_train[coreset]

        self.model = copy.deepcopy(self.classifier)
        self.model.fit(X_core, y_core)

        self._update_results(coreset)

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
                "core_cv",
                "coreset_size",
                "coreset",
        ]

        df = pd.DataFrame(columns=columns)

        for i in range(0, self.n_splits):

            row = []

            row.append(self.accuracy["core_cv"][i])
            row.append(len(self.coreset[i]))
            row.append(self.coreset[i])

            df = df.append(pd.DataFrame([row], columns=columns),
                           ignore_index=True)

        self.results = pd.concat([self.results, df], axis=1)

    def missing_class_fix(self, coreset, y_train):

        n_classes = len(np.unique(y_train))
        y_core = y_train[coreset]

        if n_classes != len(np.unique(y_core)):

            missing_classes = np.setdiff1d(np.unique(y_train),
                                           np.unique(y_core))

            for mc in missing_classes:
                indeces = np.argwhere(y_train == mc)
                coreset[indeces[0]] = 1

        return coreset

    def setup(self):
        # argparse; all arguments are optional
        p = ArgumentParser()

        p.add_argument("--classifier_name", "-c",
                       help="Classifier(s) to be tested. Default: %s."
                       % (self.classifier_name))
        p.add_argument("--dataset_name", "-d",
                       help="Dataset to be tested. Default: %s."
                       % (self.dataset_name))

        # finally, parse the arguments
        args = p.parse_args()

        if args.dataset_name:
            self.dataset_name = args.dataset_name
        if args.classifier_name:
            self.classifier_name = args.classifier_name

        # initialize classifier; some classifiers have random elements, and
        # for our purpose, we are working with a specific instance, so we fix
        # the classifier's behavior with a random seed
        classifier_class = self.classifiers[self.classifier_name]
        self.classifier = classifier_class(random_state=self.seed)

        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)
