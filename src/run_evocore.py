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
from evocore import EvoCore
from bayescore import BayesCore, bc_algorithms
from cross_validator import datasets, classifiers


def main():

    bc = BayesCore(datasets=datasets,
                   models=classifiers,
                   algorithms=bc_algorithms)
    bc.run()

    ec = EvoCore(datasets=datasets,
                 models=classifiers)
    ec.run()

    return


if __name__ == "__main__":
    sys.exit(main())
