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

import copy
from typing import Iterable, Tuple
import pandas as pd
from lazygrid.lazy_estimator import LazyPipeline
from sklearn.base import BaseEstimator


class CoresetPipeline(LazyPipeline):
    """
    CoresetPipeline class.
    """

    def __init__(self, steps, core_step, database: str = "./database/", verbose: bool = False):
        super().__init__(steps, verbose=verbose)
        self.database = database
        self.core_step = core_step

    def _fit(self, X: pd.DataFrame, y: Iterable = None, **fit_params):
        Xt = X
        yt = y
        ids = ()
        # fit or load intermediate steps
        for (step_idx, name, transformer) in self._iter(with_final=False, filter_passthrough=False):
            transformer, ids, Xt, yt = self._fit_step(transformer, ids, False, Xt, yt, **fit_params)
            self.steps[step_idx] = (name, copy.deepcopy(transformer))

        core_step = self.core_step[1]
        core_step, ids, Xt, yt = self._fit_step(core_step, ids, False, Xt, yt, **fit_params)
        self.core_step = (self.core_step[0], copy.deepcopy(core_step))

        # fit or load final step
        transformer, ids, Xt, yt = self._fit_step(self.steps[-1][1], ids, True, Xt, yt, **fit_params)
        self.steps[-1] = (self.steps[-1][0], copy.deepcopy(transformer))

        return self

    def _fit_step(self, transformer: BaseEstimator, ids: Tuple, is_final: bool,
                  X: pd.DataFrame, y: Iterable = None, **fit_params):
        # make transformer unique for each CV split
        transformer.train_ = tuple(X.index)
        transformer.features_ = tuple(X.columns)

        # load transformer from database
        transformer_loaded, ids_loaded = self._load(transformer, ids)
        is_loaded = False if transformer_loaded is None else True
        if is_loaded:
            transformer = transformer_loaded
            ids = ids_loaded

        # fit final step
        if is_final:
            if not is_loaded:
                transformer.fit(X, y, **fit_params)

        # fit intermediate steps
        else:
            if not is_loaded:
                transformer.fit(X, y, **fit_params)

            transformed_data = transformer.transform(X)

            if isinstance(transformed_data, Tuple):
                X, y = transformed_data

            else:
                Xnp = transformed_data

                # reshape input data
                if Xnp.shape != X.shape:
                    if isinstance(X, pd.DataFrame):
                        X = X.iloc[:, transformer.get_support()]

                else:
                    X = pd.DataFrame(Xnp)

        # save transformer
        if not is_loaded:
            ids = self._save(transformer, ids)

        return transformer, ids, X, y
