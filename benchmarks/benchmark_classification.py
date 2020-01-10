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

import os
import sys
import traceback

import pandas as pd
import numpy as np
import scipy
from sklearn import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from scipy.stats import sem
from sklearn.svm import SVC
from tqdm import tqdm
from prototypes.evocore import EvoCore
from prototypes import bayesian_core_set as bc
from prototypes import coreset_pipeline as cp
import lazygrid as lg

data_sets = [
    "iris",                         # 150 samples       5 features      3 classes
    "micro-mass",                   # 571 samples       1301 features   20 classes
    "soybean",                      # 683 samples       36 features     19 classes (2337 NaNs)
    "credit-g",                     # 1000 samples      21 features     2 classes
    "kr-vs-kp",                     # 3196 samples      37 features     2 classes
    "abalone",                      # 4177 samples      9 features      28 classes
    "isolet",                       # 7797 samples      618 features    26 classes
    "jm1",                          # 10885 samples     22 features     2 classes (25 NaNs)
    "gas-drift",                    # 13910 samples     129 features    6 classes
    "mozilla4",                     # 15545 samples     6 features      2 classes
    "letter",                       # 20000 samples     17 samples      26 classes
    "Amazon_employee_access",       # 32769 samples     10 features     2 classes
    "electricity",                  # 45312 samples     9 features      2 classes
    "mnist_784",                    # 70000 samples     785 features    10 classes
    "covertype",                    # 581012 samples    55 features     7 classes
]


def main():

    # Cross-validation params
    cv = 10
    n_jobs = -1
    seed = 42
    scoring = "f1_weighted"
    results_dir = "./results/"

    # EvoCore params
    n_splits = 10
    pop_size = 100
    max_generations = 100
    max_points_in_core_set = 500

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    for data_set_name in data_sets:

        # Load data
        db_name = os.path.join("./db", data_set_name)
        x, y, n_classes = lg.datasets.load_openml_dataset(dataset_name=data_set_name)
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        x = imp_mean.fit_transform(x)
        x = pd.DataFrame(x)
        n_classes = len(np.unique(y))

        scaler = StandardScaler()
        clf = RidgeClassifier(random_state=seed)

        # Bayesian pipelines
        estimators = []
        for algorithm_name in bc.algorithms.keys():
            core_alg = bc.BayesianPrototypes(algorithm_name=algorithm_name, random_state=seed)
            estimators.append(cp.CoresetPipeline([('pre', scaler), ('clf', clone(clf))],
                                                 core_step=('core', core_alg), database=db_name))

        # EvoCore pipeline
        ec = EvoCore(estimator=clone(clf), pop_size=pop_size, max_generations=max_generations,
                     max_points_in_core_set=max_points_in_core_set, n_splits=n_splits, random_state=seed)
        estimators.append(cp.CoresetPipeline([('pre', scaler), ('clf', clone(clf))],
                                             core_step=('core', ec), database=db_name))

        results_verbose = pd.DataFrame()
        results_summary = pd.DataFrame()
        progress_bar = tqdm(estimators, position=0)
        for estimator in progress_bar:

            scores = cross_validate(estimator, x, y, scoring=scoring,
                                    cv=cv, return_estimator=True,
                                    return_train_score=True, n_jobs=n_jobs)

            if isinstance(estimator.core_step[1], bc.BayesianPrototypes):
                core_name = estimator.core_step[1].algorithm_name
            else:
                core_name = "EvoCore"
            clf_name = estimator.steps[-1][1].__class__.__name__

            core_size = [len(est.core_step[1].y_core_) for est in scores["estimator"]]

            scores.update({
                "dataset name": data_set_name,
                "samples": x.shape[0],
                "features": x.shape[1],
                "classes": n_classes,
                "core": core_name,
                "classifier": clf_name,
                "core size": core_size,
            })

            summary = {
                "dataset name": data_set_name,
                "samples": x.shape[0],
                "features": x.shape[1],
                "classes": n_classes,
                "core": core_name,
                "classifier": clf_name,
                "avg fit time": scipy.average(scores["fit_time"]),
                "sem fit time": scipy.stats.sem(scores["fit_time"]),
                "avg train score": scipy.average(scores["train_score"]),
                "sem train score": scipy.stats.sem(scores["train_score"]),
                "avg test score": scipy.average(scores["test_score"]),
                "sem test score": scipy.stats.sem(scores["test_score"]),
                "avg core size": scipy.average(core_size),
                "sem core size": scipy.stats.sem(core_size)
            }

            scores = pd.DataFrame.from_records(scores)
            results_verbose = pd.concat([results_verbose, scores], ignore_index=True)
            results_summary = pd.concat([results_summary,
                                         pd.DataFrame.from_records(summary, index=["dataset name"])],
                                        ignore_index=True)

            results_verbose.to_csv(os.path.join(results_dir, data_set_name + "_results_verbose.csv"))
            results_summary.to_csv(os.path.join(results_dir, data_set_name + "_results_summary.csv"))

    return


if __name__ == "__main__":
    sys.exit(main())
