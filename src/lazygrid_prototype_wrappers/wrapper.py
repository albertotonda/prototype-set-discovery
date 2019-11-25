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

import copy
from typing import Any, Callable
import sklearn
from lazygrid.wrapper import corner_cases, Wrapper, SklearnWrapper
from evocore import EvoCore
from bayesian_core_set import BayesianPrototypes
import lazygrid as lg


def _parse_evocore_parameters(model):
    args = []
    for attribute_name in dir(model):
        if not attribute_name.startswith("_") and not attribute_name.endswith("_"):
            try:
                attribute = getattr(model, attribute_name)
                if isinstance(attribute, Callable):
                    attribute = attribute.__name__
                else:
                    attribute = ' '.join(str(attribute).split())
                if attribute_name != str(attribute):
                    if attribute_name not in ["core_set", "individuals", "x_core",
                                              "x_train", "y_core", "y_train", "n_classes",
                                              "max_points_in_core_set", "min_points_in_core_set"]:
                        args.append(attribute_name + ": " + str(attribute))
            except (sklearn.exceptions.NotFittedError, AttributeError):
                pass
    return ", ".join(args)


class PipelineCoreWrapper(lg.wrapper.PipelineWrapper):

    def __init__(self, model, dataset_id=None, dataset_name=None, db_name=None,
                 fit_params=None, predict_params=None, score_params=None, model_id=None,
                 cv_split=None, is_standalone=True):
        """
        Wrapper initialization.

        Parameters
        --------
        :param model: machine learning model
        :param dataset_id: data set identifier
        :param dataset_name: data set name
        :param db_name: database name
        :param fit_params: model's fit parameters
        :param predict_params: model's predict parameters
        :param score_params: model's score parameters
        :param model_id: model identifier
        :param cv_split: cross-validation split identifier
        :param is_standalone: True if model can be used independently from other models
        """
        Wrapper.__init__(self, model, dataset_id, dataset_name, db_name, fit_params,
                         predict_params, score_params, model_id, cv_split, is_standalone)
        self.models = []
        self.models_id = []
        for step in model.steps:
            pipeline_step = SklearnWrapper(model=step[1], cv_split=cv_split, dataset_id=dataset_id,
                                           dataset_name=dataset_name, db_name=db_name, is_standalone=False)
            if isinstance(step[1], EvoCore) or isinstance(step[1], BayesianPrototypes):
                pipeline_step.parameters = _parse_evocore_parameters(step[1])
            self.models.append(pipeline_step)
            self.models_id.append(pipeline_step.model_id)

        parameters = []
        for step in self.models:
            parameters.append(step.parameters)
        self.parameters = str(parameters)

    def fit(self, x_train, y_train, **kwargs):
        """
        Fit model with some samples.

        Parameters
        --------
        :param x_train: train data
        :param y_train: train labels
        :return: None
        """
        x_train_t = x_train
        y_train_t = y_train
        i = 0
        for pipeline_step, model in zip(self.model.steps, self.models):
            if not model.is_fitted:
                # print("NOT FITTED!")
                pipeline_step[1].fit(x_train_t, y_train_t)
                self.models[i] = lg.wrapper.SklearnWrapper(model=copy.deepcopy(pipeline_step[1]),
                                                           cv_split=self.cv_split,
                                                           dataset_id=self.dataset_id, dataset_name=self.dataset_name,
                                                           db_name=self.db_name, is_standalone=False)
                self.models[i].parameters = _parse_evocore_parameters(pipeline_step[1])
                self.models[i].is_fitted = True
            if isinstance(pipeline_step[1], EvoCore) or isinstance(pipeline_step[1], BayesianPrototypes):
                x_train_t, y_train_t = pipeline_step[1].transform(x_train_t)
            elif hasattr(pipeline_step[1], "transform"):
                x_train_t = pipeline_step[1].transform(x_train_t)
            i += 1
        self.is_fitted = True

    def score(self, x, y, **kwargs) -> Any:
        x_t = x
        score = None
        for pipeline_step, model in zip(self.model.steps, self.models):
            if isinstance(pipeline_step[1], EvoCore) or isinstance(pipeline_step[1], BayesianPrototypes):
                continue
            elif hasattr(pipeline_step[1], "transform"):
                x_t = pipeline_step[1].transform(x_t)
            elif hasattr(pipeline_step[1], "score"):
                score = pipeline_step[1].score(x_t, y)
        return score

    def predict(self, x, **kwargs) -> Any:
        x_t = x
        predictions = None
        for pipeline_step, model in zip(self.model.steps, self.models):
            if isinstance(pipeline_step[1], EvoCore) or isinstance(pipeline_step[1], BayesianPrototypes):
                continue
            elif hasattr(pipeline_step[1], "transform"):
                x_t = pipeline_step[1].transform(x_t)
            elif hasattr(pipeline_step[1], "score"):
                predictions = pipeline_step[1].predict(x_t)
        return predictions

    def set_random_seed(self, seed, split_index, random_model):
        """
        Set model random state if possible.

        Parameters
        --------
        :param seed: random seed
        :param split_index: cross-validation split identifier
        :param random_model: whether the model should have the same random state for each cross-validation split
        :return: None
        """
        if random_model:
            random_state = split_index
        else:
            random_state = seed
        for parameter in list(self.model.get_params().keys()):
            if "random_state" in parameter:
                self.model.set_params(**{parameter: random_state})
        self.cv_split = split_index
        # set random seed of pipeline steps
        for model in self.models:
            model.set_random_seed(seed, split_index, random_model)
            model.parameters = _parse_evocore_parameters(model.model)