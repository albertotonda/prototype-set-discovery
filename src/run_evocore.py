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

import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import load_digits
import lazygrid as lg
from scipy.stats import sem
from scipy import mean
from evocore import EvoCore
import bayesian_core_set as bc
from lazygrid_prototype_wrappers.wrapper import PipelineCoreWrapper


def main():

    seed = 42
    db_name = "evocore-test"

    logger = lg.logger.initialize_logging(log_name=db_name)

    dataset_id = 1
    dataset_name = "digits"
    # x, y, n_classes = lg.datasets.load_openml_dataset(dataset_name=dataset_name, logger=logger)
    x, y = load_digits(return_X_y=True)

    # generate list of estimators
    preprocessors = [RobustScaler()]
    core_set_alg = [
        EvoCore(estimator=RandomForestClassifier(random_state=seed),
                pop_size=100, max_generations=100, n_splits=10, seed=seed),
    ]
    for algorithm in bc.algorithms:
        core_set_alg.append(bc.BayesianPrototypes(algorithm_name=algorithm))
    classifiers = [
        RandomForestClassifier(random_state=seed),
        # SVC(random_state=seed),
        # LogisticRegression(random_state=seed)
    ]
    elements = [preprocessors, core_set_alg, classifiers]
    estimators = lg.grid.generate_grid(elements)

    score_list = []
    for estimator in estimators:
        model = PipelineCoreWrapper(estimator, dataset_id, dataset_name, db_name)
        scores, fitted_models, y_pred_list, y_list = lg.model_selection.cross_validation(model, x, y, logger=logger)
        score_list.append(scores["val_cv"])
        logger.info("Accuracy: %.4f +- %.4f" % (mean(scores["val_cv"]), sem(scores["val_cv"])))

    return


if __name__ == "__main__":
    sys.exit(main())
