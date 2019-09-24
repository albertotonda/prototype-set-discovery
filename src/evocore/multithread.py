# -*- coding: utf-8 -*-
#
######################
#                    #
# Multithread module #
#                    #
######################
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


import sys
from queue import Queue
from threading import Thread, Lock
import traceback


class Worker(Thread):
    """ Thread executing tasks from a given tasks queue """
    def __init__(self, tasks, thread_id):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.id = thread_id
        self.start()

    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
#            print("[Thread %d] Args retrieved: \"%s\"" % (self.id, args))
            new_args = []
#            print("[Thread %d] Length of args: %d" % (self.id, len(args)))
            for a in args[0]:
                new_args.append(a)
            new_args.append(self.id)
#            print("[Thread %d] Length of new_args: %d" % (self.id,
#                  len(new_args)))
            try:
                func(*new_args, **kargs)
            except Exception as e:
                # An exception happened in this thread
#                print(e)
                print(traceback.format_exc())
            finally:
                # Mark this task as done, whether an exception happened or not
#                print("[Thread %d] Task completed." % self.id)
                self.tasks.task_done()


class ThreadPool:
    """ Pool of threads consuming tasks from a queue """
    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for i in range(num_threads):
            Worker(self.tasks, i)

    def add_task(self, func, *args, **kargs):
        """ Add a task to the queue """
        self.tasks.put((func, args, kargs))

    def map(self, func, args_list):
        """ Add a list of tasks to the queue """
        for args in args_list:
            self.add_task(func, args)

    def wait_completion(self):
        """ Wait for completion of all the tasks in the queue """
        self.tasks.join()


def main():

    import numpy as np

    # Function to be executed in a thread with thread lock
    def wait_and_save_delay(my_data: dict, integer: int,
                            threadLock: Lock, thread_id: int):
        # acquire lock and write result in the appropriate spot
        threadLock.acquire()
        my_data["Positive"].append(integer)
        my_data["Negative"].append(-integer)
        threadLock.release()

    # Instantiate a thread pool with 5 worker threads
    pool = ThreadPool(20)
    threadLock = Lock()

    # Generate data
    my_data = {
            "Positive": [np.inf],
            "Negative": [-np.inf],
    }
    # arguments to pass
    arguments = []
    for i in range(0, 10):
        arguments.append((my_data, i, threadLock))

    # Add the jobs in bulk to the thread pool. Alternatively you could use
    # `pool.add_task` to add single jobs. The code will block here, which
    # makes it possible to cancel the thread pool with an exception when
    # the currently running batch of workers is finished.
    pool.map(wait_and_save_delay, arguments)
    pool.wait_completion()

    print(my_data)


if __name__ == "__main__":
    sys.exit(main())
