# -*- coding: utf-8 -*-
#
##################
#                #
# Logging module #
#                #
##################
#
# Copyright [YEAR] [AUTHORS]
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

import datetime
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_DIR = "../log"


def initialize_logging(log_name: str, date: bool = True) -> logging.Logger:

    if date:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_name = os.path.join(LOG_DIR, date_str + "_" + log_name + ".log")

    # remove old logger if it exists
    if os.path.exists(log_name):
        os.remove(log_name)

    # create an additional logger
    logger = logging.getLogger(log_name)

    # format log file
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s",
                                  "%Y-%m-%d %H:%M:%S")

    # the 'RotatingFileHandler' object implements a log file
    # that is automatically limited in size
    fh = RotatingFileHandler(log_name,
                             mode='a',
                             maxBytes=100*1024*1024,
                             backupCount=2,
                             encoding=None,
                             delay=0)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Starting " + log_name + "!")

    return logger


def close_logging(logger: logging.Logger):

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def main():

    # unique name of the directory
    uniqueName = "unique-name"

    # initialize logging, using a logger that smartly manages disk occupation
    logger = initialize_logging(uniqueName)

    # start program
    logger.info("Hi, I am a program, starting now!")
    logger.info(type(logger))
    close_logging(logger)

    return


if __name__ == "__main__":
    sys.exit(main())
