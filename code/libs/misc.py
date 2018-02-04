"""Miscellaneous functions."""
import os.path
import json
import time
import datetime
import numpy as np 
import h5py
import math
from scipy import linalg
from scipy.stats import t
from scipy.sparse import csr_matrix
from copy import deepcopy

########################
#   MISC
########################
def is_iter(obj):
    return hasattr(obj, '__iter__')

def argsort_randomize_ties(a, kind='quicksort'):
    """Argsort on a 1dim array where the ties are randomly broken instead of picked deterministicly by numpy.
    
    * use random uniform noise for breaking ties
    * only 1 dimensional arrays
    """
    assert a.ndim == 1
    struc_array = np.array(
        zip(a.tolist(),np.random.rand(*a.shape).tolist()),
        dtype=[('var', int),('noise', float)])
    return np.argsort(struc_array, order=['var', 'noise'])


def get_function_arguments(func):
    return func.func_code.co_varnames[:func.func_code.co_argcount]

########################
#   I/O
########################
def timeit(fn):
    """ Timing decorator """
    def timed(*args, **kwawgs):
        ts = time.time()
        result = fn(*args, **kwawgs)
        te = time.time()
        print '[time] %r: %2.2f sec' % \
              (fn.__name__, te-ts)
        return result
    return timed

def format_size_printable(num, suffix='B'):
    """Source: """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return '%3.1f%s%s' % (num, unit, suffix)
        num /= 1024.0
    # return '%.1f%s%s' % (num, 'Yi', suffix)
    return '{:.1f}{}{}'.format(num, 'Yi', suffix)

def save_sparse_csr(filename, array):
    name, file_ext = os.path.splitext(filename)
    if file_ext == '.hdf5':
        with h5py.File('filename','w') as fid:
            fid.create_dataset('data',compression='gzip')
    if file_ext == '.npz':
        np.savez(filename, data = array.data ,indices=array.indices,
                 indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    name, file_ext = os.path.splitext(filename)
    if 'file_ext' == '.npz':
        loader = np.load(filename)
        return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
    if 'file_ext' == '.hdf5':
        pass

# IMPORTANT:
#     Python (hdf5) and R (rhdf5) use transposed conventions for storing 2D arrays 
#     to disk. The input / output buffer files will therefor need transposing.

#     We addopt the following convention here:
#         Input data:
#             Python observational data is saved p x N
#             R observational data is used as N x p
#             --> No transposing needed
#         Output data:
#             R scripts return (as saved hdf5 file) causes x effects shape.
#             --> NEED TO TRANSPOSE THE RESULT LOADED FROM A HDF5-FILE
#                 IN PYTHON WHEN LOADING!
#     This convention is hard-coded in the save_array and load_array functions if 
#     calling with load_from_r=True option.
def save_array(data, filename, data_name='data', verbose=False, compression='gzip'):
    if verbose: print 'Saving results at {}.'.format(filename)
    if os.path.exists(filename):
        if verbose: print 'Removing existing file {}...'.format(filename)
        os.remove(filename)
    if compression is None:
        with h5py.File(filename, 'w') as fid:
            fid.create_dataset(data_name, data=data)
    else:
        with h5py.File(filename, 'w') as fid:
            fid.create_dataset(data_name, data=data, compression=compression)

def load_array(filename, load_from_r=False, verbose=False):
    with h5py.File(filename, 'r') as fid:
        result = fid['data'][...]
    if verbose: print 'Loading existing result at {}.'.format(filename)
    if load_from_r: return result.T #NEED TO TRANSPOSE 
    return result

# def write_time_stamp(file, time_stamp=None):
#     time_stamp = time_stamp or time.gmtime()

#     # if h5py file:
#     if type(file) == h5py._hl.files.File:
#         file.create_dataset('/time', data=time.mktime(time_stamp))    
#     else:
#         with h5py.File(file, 'r+') as fid:
#             fid.create_dataset('/time', data=time.mktime(time_stamp))
    
# def read_time_stamp(file):
#     if type(file) == h5py._hl.files.File:
#         return file['/time'][...]
#     else:
#         with h5py.File(file, 'r') as fid:
#             file['/time'][...]

def write_datestamp_hdf5(file):
    # if h5py file:
    if type(file) == h5py._hl.files.File:
        file.create_dataset('/date', data=datetime.date.today().toordinal())
    else:
        with h5py.File(file, 'w') as fid:
            fid.create_dataset('/date', data=datetime.date.today().toordinal())
    
def read_datestamp_hdf5(file, larger_diff_days=None):
    if type(file) == h5py._hl.files.File:
        date = datetime.date.fromordinal(file['/date'][...])
    else:
        with h5py.File(file, 'r') as fid:
            date = datetime.date.fromordinal(fid['/date'][...])

    if larger_diff_days:
        return (datetime.date.today() - date).days >= larger_diff_days
    else: 
        return date

def write_datestamp(file):
    with open(file, 'w') as f:
        f.write(str(datetime.date.today().toordinal()) + '\n' + str(datetime.date.today()))

def read_datestamp(file, larger_diff_days=None):
    with open(file, 'r') as f:
        date = datetime.date.fromordinal(int(f.readline()))
    if larger_diff_days:
        return (datetime.date.today() - date).days >= larger_diff_days
    else: 
        return date

#     # if h5py file:
#     if type(file) == h5py._hl.files.File:
#         file.create_dataset('/date', data=datetime.date.today().toordinal())
#     else:
#         with h5py.File(file, 'r+') as fid:
#             fid.create_dataset('/date', data=datetime.date.today().toordinal())
    
# def read_datestamp(file, larger_diff_days=None):
#     if type(file) == h5py._hl.files.File:
#         date = datetime.date.fromordinal(file['/date'][...])
#     else:
#         with h5py.File(file, 'r') as fid:
#             date = datetime.date.fromordinal(fid['/date'][...])

#     if larger_diff_days:
#         return abs(date > datetime.date.today()) >= larger_diff_days
#     else: 
#         return date


########################
#   Stats and prob
########################
def pcorr_cond_z(corr, z, y=None, add_nans=False):
    """Compute partial correlation rho_{xy|z} for one condition variable z. 
    
    Args:
        corr (2d array): correlration matrix
        z (int): index of the condition variable
        y (int, optional): if given, only compute the slice of index y
        add_nans (bool, optional):  change "1.0"'s on the diagonal of x,y to np.nan

    Returns:
        rho_{xy|z} (2d array): for (x,y)
    """
    if add_nans: np.fill_diagonal(corr, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        if type(y) is int:
            return np.divide(corr[:,y] - corr[:,z] * corr[y,z],
                np.sqrt((1 - corr[:,z]**2) *  (1 - corr[y,z]**2)))
        else:
            return np.divide(corr  - np.outer(corr[:,z], corr[z,:]),\
                np.sqrt(np.outer(1 - corr[:,z]**2,1 - corr[z,:]**2)))


def pcorr_cond_z_solo(corr, y, z):
    print 'fnc start'
    # with np.errstate(invalid='ignore'):
    return np.divide(corr[:,y] - corr[:,z] * corr[y,z],
        np.sqrt((1 - corr[:,z]**2) *  (1 - corr[y,z]**2)))


def pcorr_cond_z_solo_ignore(corr, y, z):
    print 'fnc start'
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(corr[:,y] - corr[:,z] * corr[y,z],
            np.sqrt((1 - corr[:,z]**2) *  (1 - corr[y,z]**2)))


def pcorr_cond_all(corr, ids=None, add_nans=True, verbose=True):
    """Compute partial correlation rho_{xy|z} for all singleton condition sets. 
    
    Args:
        corr (2d array): correlration matrix
        ids (None, optional): indices of condition variables; default is all variables.
        add_nans (bool, optional): change "1.0"'s on the diagonal of x,y for each z to np.nan
    
    Returns:
        rho_{xy|z} (3d array) for (x,y,z)
    """
    start_time = time.time()
    if type(ids) == int: ids = (ids,) # make iterable
    ids = ids or range(len(corr)) # all indices if not specified
    tmp = []
    for i in ids:
        tmp.append(pcorr_cond_z(corr=corr, z=i, add_nans=add_nans))
        if verbose: '[misc] Computing pcorr for var {} out of {} in {:.2f}s'.format(i, len(ids), time.time() - start_time)
    return np.array(tmp).transpose(1,2,0) # dims z x y transposed to x y z
    # if add_nans:
    #     for j in xrange(pcorr.shape[2]):
    #         np.fill_diagonal(pcorr[:,:,j], np.nan)
    #     pcorr[np.array(pcorr.shape[2] * (~np.eye(len(corr), dtype=bool),)).transpose(2,0,1)] = np.nan
    # return pcorr

def cond_tests(corr, cth_indep=None, cth_dep=None, ids=None, add_nans=True, verbose=False):
    """Compute conditional independence and dependence efficiently for all given singleton condition sets. 
    
    Args:
        corr (2d array): correlration matrix
        ids (None, optional): indices of condition variables; default is all variables.
        add_nans (bool, optional): change "1.0"'s on the diagonal of x,y for each z to np.nan
    
    Returns:
        rho_{xy|z} (3d array) for (x,y,z)
    """
    start_time = time.time()
    if type(ids) == int: ids = (ids,) # make iterable
    ids = ids or range(len(corr)) # all indices if not specified

    # create result
    if cth_indep: result_indep = np.zeros((corr.shape[0], corr.shape[1], len(ids)), dtype=bool)
    if cth_dep: result_dep = np.zeros((corr.shape[0], corr.shape[1], len(ids)), dtype=bool)
    
    # loop over ids
    for z in ids:
        abs_pcorr = np.abs(pcorr_cond_z(corr=corr, z=z, add_nans=False))
        # with np.errstate(lesser='ignore', greater='ignore'):
        with np.errstate(invalid='ignore'):
            if cth_indep: 
                result_indep[:, :, z] = abs_pcorr < cth_indep
                # if add_nans: np.fill_diagonal(result_indep[:,:,z], 0) # fix?
            if cth_dep: 
                result_dep[:, :, z] = abs_pcorr > cth_dep
                # if add_nans: np.fill_diagonal(result_dep[:,:,z], 0) # fix?
            if verbose and ((z+1) % 10 == 0 or (z+1) == len(ids)):
                print '[misc] Computing cond_indep for var {} out of {} in {:.2f}s'.format(z, len(ids), time.time() - start_time)
    # returns
    if cth_indep and cth_dep: return result_indep, result_dep
    if cth_indep: return result_indep
    if cth_dep: return result_dep


### recall: p1 > p2 implies c1 < c2
def pth2Cth(pth, N, dz=1):
    """Convert threshold on 2-tailed p-value of to equivalent threshold of (partial) correlation
    
    Args:
        pth (TYPE): Description
        N (TYPE): Description
        dz (int, optional): Description
    """
    #dz = 1  # dimension of conditioning variable
    df = max(N - dz - 2,0)  # degrees of freedom
    y = -t.isf(1.0 - pth / 2.0, df, loc=0, scale=1) / math.sqrt(df) 
    Cth = abs(y / math.sqrt(1.0 + y ** 2))
    return Cth

def Cth2pth(Cth, N, dz=1):
    """Convert threshold on partial correlation to equivalent threshold on its 2-tailed p-value"""
    #dz = 1  # dimension of conditioning variable
    df = max(N - dz - 2,0)  # degrees of freedom
    tstat = 0 if Cth == 1.0 else Cth / math.sqrt(1.0 - Cth ** 2) # calculate t statistic 
    tstat = math.sqrt(df) * tstat
    pth = 2 * t.cdf(-abs(tstat), df, loc=0, scale=1)  # calculate p-value 2 tailed
    return pth
    

########################
#   Groundtruth functions.
########################
def count_transitive_edges(mat, print_results=False):
    mat = mat.astype('int')
    trans_matrix = np.dot(mat, mat)
    np.fill_diagonal(trans_matrix, False) # To be sure, should not be necessary
    transitive_edges = np.logical_and(trans_matrix, mat)
    total = transitive_edges.sum()
    if print_results: print 'number of transitive edges, i.e. A->B->C && A->C where A =/= C:', \
        total, '\nnumber of A->B->C where A =/= C:', trans_matrix.sum(),\
        '\nfraction:', total / float(mat.sum())
    return total


def count_cyclic_edges(mat, print_results=False):
    mat = mat.astype('int')
    total = np.dot(mat,mat).diagonal().sum()
    if print_results: print 'number of edges where cyclicity i.e. A-B && B->A holds (double counted):\n', \
        total, 'out of', \
        mat.sum(), 'edges; \nfrac:', \
        total / float(mat.sum()), '\n'
    return total


def convert_list_pairs_to_matrix(causes, effects, row_identifiers, col_identifiers):
    """Assumes indexes from identifier list.
    
    Args:
        causes (TYPE): Description
        effects (TYPE): Description
        row_identifiers (list): Description
        col_identifiers (list): Description
    
    Returns:
        TYPE: Description
    """
    if causes is None or effects is None or row_identifiers is None or col_identifiers is None:
        raise Exception('supply all parameters...')
    ids_1 = sgd_naming(row_identifiers)
    ids_2 = sgd_naming(col_identifiers)
    result = np.zeros((len(row_identifiers),len(col_identifiers)),dtype='bool')
    for id, ca in enumerate(causes):
        result[ids_1.index(ca), ids_2.index(effects[id])] = True
    return result


########################
#   Confusion Matrix
########################
def true_positives(est, gt):
    assert est.shape == gt.shape
    return np.logical_and(est.astype('bool'), gt.astype('bool')).sum()

def true_negatives(est, gt):
    assert est.shape == gt.shape
    return np.logical_and(~est.astype('bool'), ~gt.astype('bool')).sum()

def false_negatives(est, gt):
    assert est.shape == gt.shape
    return np.logical_and(~est.astype('bool'), gt.astype('bool')).sum()

def total_positives(est):
    return est.astype('bool').sum()

def precision(est, gt):
    assert est.shape == gt.shape
    try: 
        return true_positives(est,gt) / float(total_positives(est))
    except ZeroDivisionError:
        return float('NaN')

def recall(est, gt):
    assert est.shape == gt.shape
    try:
        return true_positives(est,gt) / float(gt.astype('bool').sum())
    except ZeroDivisionError:
        return float('NaN')

def risk_estimate(est, gt, loss_func='zero-one-loss'):
    """ est: cause x effect (indirect) boolean matrix"""
    if loss_func == 'zero-one-loss':
        return np.logical_xor(est.astype('bool'), gt.astype('bool')).sum()
    else:
        raise Exception('Specify correct loss function')

def compute_contingency_table(matrix1, matrix2):
    """
        Contingency table for two binary-like matrices.
        - input :   matrix1 (numpy array/matrix) 
                    matrix2 (numpy array/matrix)
        - output:   contingency table (row's ~ matrix 1, col's ~ matrix 2)
    """
    matrix1 = matrix1.astype('bool')
    matrix2 = matrix2.astype('bool')
    result_table = np.zeros((2,2),dtype='int')
    result_table[0,0] = np.sum(np.logical_and(np.logical_not(matrix1),\
                                              np.logical_not(matrix2)))
    result_table[1,1] = np.sum(np.logical_and(matrix1, \
                                              matrix2))
    result_table[0,1] = np.sum(np.logical_and(np.logical_not(matrix1),\
                                              matrix2))
    result_table[1,0] = np.sum(np.logical_and(matrix1,\
                                              np.logical_not(matrix2)))
    return result_table


def transitive_closure(a, print_output=False):
    """
    Implementation of Warshall's algorithm
    http://stackoverflow.com/questions/22519680/warshalls-algorithm-for-transitive-closurepython
    O(n^3)
    """
    start = time.time()

    # assert (len(row) == len(a) for row in a)
    n = len(a)
    for k in xrange(n):
        for i in xrange(n):
            for j in xrange(n):
                a[i][j] = a[i][j] or (a[i][k] and a[k][j])
                if print_output and (a[i][k] or a[k][j]): 
                    print 'k', k, 'i', i, 'j', j, 'a[i][j]', a[i][j], 'a[i][k]', a[i][k], 'a[k][j]', a[k][j]

    print 'Transitive closure in {} sec.'.format(round((time.time() - start), 2))
    return a

