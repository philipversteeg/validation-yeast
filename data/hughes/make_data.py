#
# Script to convert RData to HDF5
#
import numpy as np
import rpy2.robjects as robjects
import h5py
import json

FILENAME_IN = './Hughes.RData'
FILENAME_OUT = './hughes.hdf5'
FILENAME_SGD = '../sgd_naming/map_all_to_systematic.json'

def sgd_naming(some_list, verbose=False):
    with open(FILENAME_SGD) as f:
        sgd_mapping = json.load(f)
        # unicode mapping... convert to uppercase string?
        sgd_mapping = dict((i.encode('ascii').upper(),j.encode('ascii').upper()) for i,j in sgd_mapping.items()) 

    # make sure it is string type, otherwise convert
    if type(some_list[0]) is not str:
        some_list = [i.encode('ascii') for i in some_list]

    # get the result fast..
    result = [sgd_mapping[i.upper()] for i in some_list if i.upper() in sgd_mapping.keys()]
    
    # print not equal items
    if verbose:
        # not_equal_items = ((i, result[idx]) for (idx, i) in enumerate(some_list) if i.upper() is not result[idx])
        not_equal_items = ((i,j) for (i,j) in zip(some_list, result) if i.upper() != j)
        for i,j in not_equal_items:
            print 'RENAMED: {} --> {}'.format(i,j)
            
    return result

if __name__ =='__main__':
 
    # load data
    robjects.r['load'](FILENAME_IN)

    genes_total = int(robjects.r['data'][robjects.r['data'].names.index('p')][0])
    obser_total = int(robjects.r['data'][robjects.r['data'].names.index('nObs')][0])
    inter_total = int(robjects.r['data'][robjects.r['data'].names.index('nInt')][0])

    obser_data = np.transpose(np.array(robjects.r['data'][robjects.r['data'].names.index('obs')]))
    inter_data = np.transpose(np.array(robjects.r['data'][robjects.r['data'].names.index('int')]))
    # inter_position = np.transpose(np.array(robjects.r['data'][robjects.r['data'].names.index('intpos')],dtype='int') - 1)
    # x = list(robjects.r['data'][robjects.r['data'].names.index('genenames')])
    genes = sgd_naming(list(robjects.r['data'][robjects.r['data'].names.index('genenames')]))
    inter_position = list(robjects.r['data'][robjects.r['data'].names.index('intpos')])
    mutants = [genes[int(i) - 1] for i in inter_position]

    assert obser_data.shape[1] == obser_total     #obser_data = 5361 x 63
    assert obser_data.shape[0] == genes_total
    assert inter_data.shape[1] == inter_total     #inter_data = 5361 x 234
    assert inter_data.shape[0] == genes_total
    assert len(genes) == genes_total
    assert len(mutants) == inter_total

    print 'Hughes data:',robjects.r['object.size'](robjects.r['data'])
    print 'Genes:', genes_total
    print 'Observations:', obser_total
    print 'Interventions:', inter_total,'\n'

    # write file to hdf5
    file_out = h5py.File(FILENAME_OUT, 'w')
    file_out.create_dataset('obser/data', data = obser_data)
    file_out.create_dataset('inter/data', data = inter_data)
    # file_out.create_dataset('inter/position', data = inter_position) 
    file_out.create_dataset('ids/genes', data = genes)
    file_out.create_dataset('ids/mutants', data = mutants)
    file_out.close()

