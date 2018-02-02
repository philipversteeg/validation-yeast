""" Package settings

This acts as a singleton class when import config, as modules are only imported 
once if 'ncessary'. Underscored attributes should be remain fixed.
"""
import os

## Settings 
_name = 'Prediction'                                    # printable name
verbose = 1                                             # default global level of print output
server = os.sys.platform != 'darwin'                    # true if on linux or windows
save_task_threshold= 10.                                # for each prediction task, a

## Folders
_folder_package = os.path.dirname(__file__)             # package folder
_folder_extern_R = _folder_package + '/extern/R'        # external R folder
folder = os.path.curdir                                 # base folder for computation
folder_test = folder + '/test'                          # containts (unit)tests and output
folder_results = folder + '/results'                    # result base folder

# create folders if not exists
# if not os.path.exists(folder): os.makedirs(folder)
# if not os.path.exists(folder_test): os.makedirs(folder_test)
# if not os.path.exists(folder_results): os.makedirs(folder_results)

## Parallel
ncores_total = os.sysconf('SC_NPROCESSORS_ONLN')        # number of cores in total, default is half of system threads
ncores_method = int(ncores_total / 2)                   # number of cores that each multiprocessed method can use