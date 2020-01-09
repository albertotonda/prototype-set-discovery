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

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import bayesiancoresets as bc
import warnings

warnings.filterwarnings("ignore")

algorithms = {
    "GIGA": bc.GIGA,
    "FrankWolfe": bc.FrankWolfe,
    "MatchingPursuit": bc.MatchingPursuit,
    "ForwardStagewise": bc.ForwardStagewise,
    "OrthoPursuit": bc.OrthoPursuit,
    "LAR": bc.LAR,
    "ImportanceSampling": bc.ImportanceSampling,
    "RandomSubsampling": bc.RandomSubsampling,
}


class BayesianPrototypes(BaseEstimator, TransformerMixin):
    """
    BayesianPrototypes class.
    """

    def __init__(self, algorithm_name: str, random_state: int):

        self.algorithm_name = algorithm_name
        self.random_state = random_state

    def fit(self, X, y=None, **fit_params):

        assert self.algorithm_name in algorithms.keys(), \
            "avaiable algorithm_name are: %s" % list(algorithms.keys())

        self.algorithm_ = algorithms[self.algorithm_name]

        n_trials = 1
        Ms = np.unique(np.logspace(0., 4., 100, dtype=np.int32))
        wts = []
        for tr in range(n_trials):
            alg = self.algorithm_(X)
            for m, M in enumerate(Ms):
                alg.run(M)
                wts = alg.weights()
        core_set = wts > 0

        self.core_set_found_ = True

        if not np.any(core_set):
            self.core_set_found_ = False
            message = "Warning: no core sets found! Random samples will be picked from each class."
            print(message)

        core_set = _missing_class_fix(core_set, y)

        self.x_core_ = X.iloc[core_set]
        self.y_core_ = y[core_set]
        self.core_set_ = core_set

        return self

    def transform(self, X, **fit_params):
        return (self.x_core_, self.y_core_)


def _missing_class_fix(core_set, y):

    n_classes = len(np.unique(y))
    y_core = y[core_set]

    if n_classes != len(np.unique(y_core)):

        missing_classes = np.setdiff1d(np.unique(y), np.unique(y_core))

        for mc in missing_classes:
            indexes = np.argwhere(y == mc)
            core_set[indexes[0]] = 1

    return core_set
