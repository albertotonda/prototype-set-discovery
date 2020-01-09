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
import pandas as pd
import numpy as np
import scipy
from sklearn import clone
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from scipy.stats import sem
from sklearn.svm import SVC
from tqdm import tqdm
from prototypes.evocore import EvoCore
from prototypes import bayesian_core_set as bc
from prototypes import coreset_pipeline as cp

data_sets = [
    "iris",
    # "digits",
    # ""
]


def main():

    cv = 10
    n_jobs = 3

    data_set_name = data_sets[0]
    seed = 42
    scoring = "f1_weighted"
    results_dir = "./results/"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # EvoCore params
    n_splits = 10
    pop_size = 100
    max_generations = 100

    # Load data
    db_name = os.path.join("./db", data_set_name)
    # x, y, n_classes = lg.datasets.load_openml_dataset(dataset_name=data_set_name)
    x, y = load_iris(return_X_y=True)
    x = pd.DataFrame(x)
    n_classes = len(np.unique(y))

    scaler = StandardScaler()
    clf = SVC(random_state=seed)

    # Bayesian pipelines
    estimators = []
    for algorithm_name in bc.algorithms.keys():
        core_alg = bc.BayesianPrototypes(algorithm_name=algorithm_name, random_state=seed)
        estimators.append(cp.CoresetPipeline([('pre', scaler), ('clf', clone(clf))],
                                             core_step=('core', core_alg), database=db_name))

    # EvoCore pipeline
    ec = EvoCore(estimator=clf, pop_size=pop_size, max_generations=max_generations, n_splits=n_splits, random_state=seed)
    estimators.append(cp.CoresetPipeline([('pre', scaler), ('clf', clf)], core_step=('core', ec), database=db_name))

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
