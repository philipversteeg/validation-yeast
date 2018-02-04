""" Package settings file"""
import os

## Settings 
_name = 'Prediction'                                            # printable name
verbose = 1                                                     # default global level of print output
server = os.sys.platform != 'darwin'                            # true if on linux or windows
save_task_threshold= 10.                                        # for each prediction task, a

## Package folders
_folder_package = os.path.abspath(os.path.dirname(__file__))    # package folder
_folder_prediction = _folder_package + '/prediction'
_folder_extern_R = _folder_prediction + '/extern/R'             # external R folder

## Project folders 
folder = '..'                                                   # base folder for computation
folder_test = folder + '/test'                                  # containts (unit)tests and output
folder_results = folder + '/results'                             # result base folder
# create folders if not exists
if not os.path.exists(folder_results): os.makedirs(folder_results)

## Parallel
ncores_total = os.sysconf('SC_NPROCESSORS_ONLN')                # number of cores in total, default is half of system threads
ncores_method = int(ncores_total / 2)                           # number of cores that each multiprocessed method can use
