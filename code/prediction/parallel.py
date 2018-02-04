"""Classes and methods to run predictions in parallel."""
# stdlib
import os
import math
import time
from threading import Thread
from Queue import Queue as ThreadQueue
from multiprocessing import Process, Semaphore, cpu_count, Queue as ProcessQueue

# local
from .methods import Predictor
from ..microarraydata import MicroArrayData
from .. import config

#############################
#   Parallelize             #
#############################
class Pool(object):
    """Parallelize computing of Predictor tasks on one MicroArrayData dataset.
    
    Specify the number of cores to use and other computational parameters. Add 
    prediction.Predicor subclassed tasks by using the add_task or add_tasks methods.
    Finally, fit the data using fit(data) method.
    
    Attributes:
        folder (str, optional): Location of the folder to use for (possible) output of computations.
            If not given, use for each folder its own setting.
        print_fitting_time (int): Output a print statement every [print_fitting_time] seconds.
        processes (int, optional): Number of simultaneous threads to use for multiprocessing.
        randomize (bool): Randomize order of computation.
        save_all_tasks (TYPE): Description
        tasks (list): All prediction tasks that are to be computed. Add to this with the add_task or add_tasks methods.
        verbose (int): Level of detailed output.
            verbose == 0 :: no messages; 
            verbose >= 1 :: (default) messages of start and done methods and periodic small printouts;
            verbose >= 2 :: set verbose level for all output methods too, for debugging tasks;
            verbose >= 3 :: thread maintenance, for debugging this class.
    
    """
    print_fitting_time = 600 # print once every 10 minutes.
    # print_fitting_time = 10 # print once every 10 seconds.

    def __init__(self, processes=None, tasks=None, folder=None, save_all_tasks=False, randomize=True, verbose=1):
        """Pool instance, where tasks can be added to 
        
        Args:
            processes (None, optional): Description
            tasks (None, optional): Description
            folder (None, optional): Description
            save_all_tasks (bool, optional): Description
            randomize (bool, optional): Description
            verbose (int, optional): Description
        """
        self.processes = processes or config.ncores_total
        self.folder = folder if folder is not None else config.folder_results
        if not os.path.exists(self.folder):
            if verbose: print '[Parallel] creating output folder at ' + self.folder
            os.makedirs(self.folder)
        self.save_all_tasks = save_all_tasks
        self.randomize = randomize
        self.verbose = verbose or config.verbose
        self._semaphore = Semaphore(self.processes)
        self.tasks = list()
        if self.verbose:
            if folder is not None:
                print '[Parallel] overriding output folder of all tasks:', self.folder
            else:
                print '[Parallel] using default output folder for all tasks that have task.folder set to:', self.folder
        if self.verbose >= 2: print '[Parallel] all prediction tasks are set to verbose'
        if self.verbose >= 3: print '[Parallel] detailed thread output'
        if tasks: self.add_tasks(tasks)

    def add_task(self, task):
        """Add a task to the pool. 
        
        Args:
            task (TYPE):
        """
        # assert issubclass(type(task), Predictor) # not necessary..
        try:
            assert task.processes <= self.processes
            if self.folder: task.folder = self.folder # override with this  
            if self.save_all_tasks: task.save = True
            if self.verbose >= 2: task.verbose = True
            self.tasks.append(task)
            if self.verbose: print '[Parallel] added task', task.name, 'with', task.processes, 'proces{}.'.format('ses' if self.processes > 1 else '')
        except AttributeError:
            raise Exception('No processes found!')

    def add_tasks(self, iterator):
        for task in iter(iterator):
            self.add_task(task)

    def fit(self, data):
        """Main thread: adds task while semaphore free, else blocks. 
        Other thread is used to free up finished tasks. Quite simple to just 
        
        Args:
            data (MicroArrayData): data.
        """
        if self.verbose: print '[Parallel] fitting {} tasks with {} process{}...'.format(len(self.tasks), 
            self.processes, 'es' if self.processes > 1 else '')
        assert issubclass(type(data), MicroArrayData)

        start_time = time.time()

        # need to use two different kinds of queues, one thread-safe and one process-safe
        task_queue = ThreadQueue()              # Pipe tasks between threads
        result_queue = ProcessQueue()           # Pipe results back to self.tasks list

        # keep track of start time per task
        def wrap_fit(task, data, index):
            """Wrapper of fit method, keep track of index of in self.task
            list where the results will be put back to
            """
            result_queue.put((task.fit(data), index))

        # Thread - start processes and acquire semaphore
        def add_processes(task_queue):
            indices = range(len(self.tasks))
            if self.randomize: random.shuffle(indices)
            for index in indices:
                task = self.tasks[index]
                for _ in xrange(task.processes):
                    self._semaphore.acquire()
                if self.verbose >= 3:
                    time.sleep(0.1)
                    print '[thread-start] acquired', task.processes, 'process{} for'.format('ses' if task.processes > 1 else ''), task.name
                p = Process(target=wrap_fit, args=(task, data, index))
                # Need non-daemonic threads to use multiprocessed python processes.
                p.daemon = False 
                p.start()
                # Put tuple of process and associated task in queue.
                task_queue.put((p, task))
            task_queue.put(None) # send sentinal
        thread_add_processes = Thread(target=add_processes, args=(task_queue,))
        thread_add_processes.start()
        
        # Thread - maintain processes and release semaphore
        def handle_processes(task_queue):
            running_tasks = []
            finished = False
            print_count = 1
            while not finished or len(running_tasks) > 0:
                # check task_queue at intervals
                if not task_queue.empty():
                    next_task = task_queue.get(timeout=0.1)
                    # receive STOP sentinal, finish
                    if next_task is None: 
                        finished = True
                    else:
                        running_tasks.append(next_task)
                # maintain process list; 
                for proc, task in running_tasks[:]:
                    if not proc.is_alive():
                        if self.verbose >= 3: print '[thread-maintain] releasing', task.processes, 'process{} for'.format('ses' if task.processes > 1 else ''), task.name
                        for _ in xrange(task.processes):
                            self._semaphore.release()
                        proc.terminate()
                        running_tasks.remove((proc,task))
                        break # need when a process is found that is done!
                time.sleep(.5)
                # print currently running processes every once in a while.
                if int((time.time() - start_time) / self.print_fitting_time) > print_count and self.verbose >= 1:
                    print '[Parallel][{:02d}h{:02d}m] running:'.format(*divmod(print_count*10, 60)),
                    for _, task in running_tasks:
                        if task == running_tasks[-1][1]: # last task
                            print '{}'.format(task.name)
                        else:
                            print '{},'.format(task.name),
                        # print '[Parallel] {} ({:d}:{:2d})'.format(task.name, *divmod(int(start_time_task[task.name] - time.time()/60), 60))
                    print_count += 1
        thread_handle_processes = Thread(target=handle_processes, args=(task_queue,))
        thread_handle_processes.start()

        # Thread - catch results from result_queue and put back in self.task list
        def handle_results():
            processed_results = 0
            while processed_results < len(self.tasks):
                task, index = result_queue.get()
                if self.verbose >= 3: print '[thread-result] saving result for', task.name, 'to task list'
                self.tasks[index] = task
                processed_results += 1
                time.sleep(.1)
        thread_handle_results = Thread(target=handle_results, args=())
        thread_handle_results.start()

        # block main thread
        thread_add_processes.join()
        thread_handle_processes.join()
        thread_handle_results.join()

        assert all((i.done for i in self.tasks))

    @property
    def result(self):
        if not all((i.done for i in self.tasks)):
            return None
        return [i.result for i in self.tasks]

    @property
    def task_names(self):
        return [i.name for i in self.tasks]

    @property
    def task_dict(self):
        """Dict: set of names (str) --> set of tasks (prediction.Predictor subclass)"""
        return dict((task.name, task) for task in self.tasks)

    @property
    def result_dict(self):
        """Dict: set of names (str) --> set of results (causallib.CausalArray)"""
        return dict((task.name, task.result) for task in self.tasks)


#############################
#   Fit from file           #
#############################
def fit_from_files(task_file, data_file, verbose=True):
    """Fit a prediction task using previously saved task file and data file"""
    print_fitting_time = 1

    if verbose: print '[FitFromFile] Called with \n\ttask: {}\n\tdata: {}'.format(task_file, data_file)
    assert os.path.exists(task_file)
    assert os.path.exists(data_file)

    task = Predictor.pickle_load(task_file)
    data = MicroArrayData.load(data_file)

    start_time = time.time()
    if verbose: print '[FitFromFile] Fitting {task.name} on {data.name}..'.format(task=task, data=data)

    task.fit(data)

    print_count = 1
    if int((time.time() - start_time) / print_fitting_time) > print_count and verbose >= 1:
        print '[FitFromFile][{:02d}h{:02d}m] running:'.format(*divmod(print_count*10, 60)),
        print_count += 1