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
import copy
import numpy as np
import traceback
import sys
import pandas as pd

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


class CrossValidator(object):
    """
    Cross validator class.
    """

    def __init__(self, data_ids=None,
                 datasets="iris",
                 models="RandomForestClassifier",
                 n_splits=10,
                 seed=42,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.name = None

        self.log_dir = None
        self.results_dir = None
        self.debug = None

        self.version = None
        self.logger = None

        self.main_seed = seed
        self.seed = None
        self.seed_cv = []
        self.n_splits = n_splits

        # check data type for datasets
        if type(datasets) is str:
            self.data_ids = [data_ids]
            self.datasets = [datasets]
        elif type(datasets) == list:
            self.data_ids = data_ids
            self.datasets = datasets
        else:
            raise TypeError('Dataset type must be str or list')
            sys.exit(1)

        # check data type for classifiers
        if type(models) is str:
            self.models = [models]
        elif type(models) == list:
            self.models = models
        else:
            raise TypeError('Classifier type must be str or list')
            sys.exit(1)

        self.data_id = None
        self.dataset_name = None
        self.classifier_name = None

        self.X = None
        self.y = None
        self.n_classes = None

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

        self.model = None

        self.accuracy = {}
        self.accuracy["train_blind"] = []
        self.accuracy["test_blind"] = []
        self.accuracy["train_base"] = []
        self.accuracy["test_base"] = []
        self.accuracy["core_cv"] = []
        self.accuracy["train_cv"] = []
        self.accuracy["test_cv"] = []

        self.results = None
        self.columns = [
                "version",
                "data_id",
                "dataset_name",
                "classifier_name",
                "seed",
                "n_splits",

                "train_blind",
                "test_blind",

                "train_baseline",
                "test_baseline",

                "seed_cv",
                "train_cv",
                "test_cv",
        ]

    def run(self):
        raise NotImplementedError

    def run_cv(self):

        for self.dataset_name in self.datasets:
            for self.classifier_name in self.models:
                self._run()

    def _run(self):

        self.setup()
        self.start_logging()
        self.load_openml_dataset()

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.seed)

        split_index = 0
        for train_index, test_index in skf.split(self.X, self.y):

            self.train_index = train_index
            self.test_index = test_index

            # unlocking random seed!
            self.seed = split_index
            self.seed_cv.append(split_index)

            self.logger.info("Split %d" % (split_index))

            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            self.fit(X_train, y_train)

            self.accuracy["train_cv"].append(self.score(X_train, y_train))
            self.accuracy["test_cv"].append(self.score(X_test, y_test))
            self.accuracy["train_base"].append(
                    self.baseline_score(X_train, y_train))
            self.accuracy["test_base"].append(
                    self.baseline_score(X_test, y_test))

            split_index += 1

        self.compute_cv_results()

        close_logging(self.logger)

    def score(self, X, y):
        return self.model.score(X, y)

    def save_results(self):

        if self.debug:
            mode = "_debug_"
        else:
            mode = ""
        experiment = self.dataset_name + "_" + \
            self.classifier_name + "_" + \
            self.name + "_" + \
            "v" + str(self.version[0]) + "_" + str(self.version[1]) + \
            mode + ".csv"
        self.results.to_csv(os.path.join(self.results_dir, experiment))

    def baseline_score(self, X, y):
        # compute baseline: accuracy using all samples
        model = copy.deepcopy(self.classifier)
        model.fit(self.X_train, self.y_train)
        return model.score(X, y)

    def start_logging(self):
        if self.debug:
            mode = "_debug_"
        else:
            mode = ""

        log_name = self.dataset_name + "_" + \
            self.classifier_name + \
            mode
        self.logger = initialize_logging(log_name)

    def compute_cv_results(self):
        """
        TODO: comment
        """

        df = pd.DataFrame(columns=self.columns)

        default_row = [
                self.version,
                self.data_id,
                self.dataset_name,
                self.classifier_name,
                self.main_seed,
                self.n_splits,

                self.accuracy["train_blind"],
                self.accuracy["test_blind"],
        ]

        for i in range(0, self.n_splits):

            row = []
            row.extend(default_row)

            row.append(self.accuracy["train_base"][i])
            row.append(self.accuracy["test_base"][i])

            row.append(self.seed_cv[i])
            row.append(self.accuracy["train_cv"][i])
            row.append(self.accuracy["test_cv"][i])

            df = df.append(pd.DataFrame([row], columns=self.columns),
                           ignore_index=True)

        self.results = df
        self.results_extra_arguments()
        self.save_results()

    def setup(self):
        raise NotImplementedError

    def fit(self, X_train, y_train):
        raise NotImplementedError

    def results_extra_arguments(self):
        pass

    def load_openml_dataset(self):

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

        except Exception:
            self.logger.error(traceback.format_exc())
            sys.exit()
