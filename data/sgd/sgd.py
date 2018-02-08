"""Query SGD to create a groundtruth and a unique gene mapping using secondary identifiers. """
# std
import os
import time

# packages
import json
import numpy as np
from intermine.webservice import Service

# custom
os.sys.path.append('../../code')
from libs import CausalArray

folder_sgd = '.'

# files naming
sgd_mapping_file = folder_sgd + '/' + 'sgd_naming.json'
_filename_duplicates = folder_sgd + '/' + '.duplicates.json'

# files groundtruth
filename_groundtruth = folder_sgd + '/' + 'gt{}_{}.hdf5'
groundtruth_types = ['all', 'genetic', 'physical']

###############
# Groundtruth
###############
def _make_groundtruth(curated, round_size=None, verbose=True):
    """ Query SGD and compile groundtruth."""
    start_time = time.time()
    if verbose: print 'Querying yeastmine for %s' % 'manually curated' if curated else 'all' + 'interactions.'

    # Start intermine service
    yeastmine = Service('http://yeastmine.yeastgenome.org/yeastmine/service')

    # 1. Get a list of all genes in quick query. Note: these are unicode formated.
    query = yeastmine.new_query('Gene')
    query.add_view('secondaryIdentifier')
    query.add_sort_order('Gene.secondaryIdentifier', 'ASC')
    query.add_constraint('Gene', 'IN', 'ALL_Verified_Uncharacterized_Dubious_ORFs', code = 'A')

    # create genes list and map
    genes_list = []
    min_length = 100
    for row in query.rows():
        gene = str(row['secondaryIdentifier'])
        genes_list.append(str(gene))
        if len(gene) < min_length:
            min_length = len(gene)
    if verbose: print 'Gene char length min:', min_length
    genes_map = dict(zip(genes_list, range(len(genes_list))))

    if verbose: print 'Number of genes: {}'.format(len(genes_list)) # should be 6604 for ALL_Verified_Uncharacterized_Dubious_ORFs

    #  Query for interactions
    query = yeastmine.new_query('Gene')
    query.add_view('secondaryIdentifier', 'interactions.details.type', 'interactions.participant2.secondaryIdentifier', 'interactions.details.role1')
    #query.add_view('secondaryIdentifier', 'interactions.details.type', 'interactions.details.role1')
    query.add_sort_order('Gene.secondaryIdentifier', 'ASC')
    query.add_constraint('organism.shortName', '=', 'S. cerevisiae', code = 'B')
    query.add_constraint('Gene', 'IN', 'ALL_Verified_Uncharacterized_Dubious_ORFs', code = 'A')
    if curated:
        query.add_constraint('interactions.details.annotationType', '=', 'manually curated', code = 'C')

    # Matrices
    adjacency_matrix_genetic = np.zeros((len(genes_list),len(genes_list)), dtype = 'bool')
    adjacency_matrix_physical = np.zeros((len(genes_list),len(genes_list)), dtype = 'bool')

    # Add to correct adjacency matrix
    current_gene = None
    for row in query.rows():
        if (current_gene != str(row['secondaryIdentifier'])):
            current_gene = str(row['secondaryIdentifier'])
            print 'Querying {} - {}'.format(genes_map[current_gene], current_gene)

        # # Test genes, print all queried info
        # if (current_gene == u'YAL008W'): # or current_gene == u'Q0085' or current_gene == 'Q0105'):
        #     if row['interactions.details.role1'] == u'Bait':
        #         print row['secondaryIdentifier'], '<--' , row['interactions.details.type'], '---', row['interactions.participant2.secondaryIdentifier']
        #     else:
        #         print row['secondaryIdentifier'], '---' , row['interactions.details.type'], '-->', row['interactions.participant2.secondaryIdentifier']

        # Fill adjacency matrices for physical and genetic relationships
        if (str(row['interactions.participant2.secondaryIdentifier']) in genes_list):
            if (row['interactions.details.type'] == u'physical interactions'):
                if row['interactions.details.role1'] == u'Bait':
                    # Bait <--- Hit relationship found. 
                    # maybe check if the reverse already exists?
                    adjacency_matrix_physical[genes_map[row['interactions.participant2.secondaryIdentifier']],\
                        genes_map[str(row['secondaryIdentifier'])]] = True
            elif row['interactions.details.type'] == u'genetic interactions':
                if row['interactions.details.role1'] == u'Bait':
                    # Bait <--- Hit relationship found. 
                    adjacency_matrix_genetic[genes_map[row['interactions.participant2.secondaryIdentifier']],\
                        genes_map[str(row['secondaryIdentifier'])]] = True

    if verbose: 
        print 'Nuber of genetic interactions {}'.format(adjacency_matrix_genetic.sum())
        print 'Nuber of physical interactions {}'.format(adjacency_matrix_physical.sum())
    
    # save all data + write date in each.
    CausalArray(array=adjacency_matrix_genetic, causes=genes_list, name='sgd_gen{}'.format('_cur' if curated else '')).save(file=filename_groundtruth.format('_curated' if curated else '', 'genetic'))
    # write_datestamp_hdf5(filename_groundtruth.format('_curated' if curated else '', 'genetic'))
    CausalArray(array=adjacency_matrix_physical, causes=genes_list, name='sgd_phys{}'.format('_cur' if curated else '')).save(file=filename_groundtruth.format('_curated' if curated else '', 'physical'))
    # write_datestamp_hdf5(filename_groundtruth.format('_curated' if curated else '', 'physical'))
    CausalArray(array=np.add(adjacency_matrix_genetic, adjacency_matrix_physical), causes=genes_list, name='sgd_all{}'.format('_cur' if curated else '')).save(file=filename_groundtruth.format('_curated' if curated else '', 'all'))
    # write_datestamp_hdf5(filename_groundtruth.format('_curated' if curated else '', 'all'))
    if verbose: print 'Done in {:.2f}'.format(time.time() - start_time)

def make_all_groundtruths(round_size=None, verbose=True):
    for curated in (True, False):
        _make_groundtruth(curated=curated, round_size=round_size, verbose=verbose)

def groundtruth(curated, gt_type, verbose=True):
    # open gt files, compute if necessary
    filename = filename_groundtruth.format('_curated' if curated else '', gt_type)
    assert gt_type in groundtruth_types
    if not os.path.exists(filename):
        _make_groundtruth(curated=curated)
    return CausalArray.load(file=filename)

# quick access
def gt_curated_all():
    # return groundtruth(curated=True, gt_type='all')
    return CausalArray.load(file=filename_groundtruth.format('_curated', 'all'))
def gt_curated_physical():
    # return groundtruth(curated=True, gt_type='physical')
    return CausalArray.load(file=filename_groundtruth.format('_curated', 'physical'))
def gt_curated_genetic():
    # return groundtruth(curated=True, gt_type='genetic')
    return CausalArray.load(file=filename_groundtruth.format('_curated', 'genetic'))
def gt_inclusive_all():
    # return groundtruth(curated=False, gt_type='all')
    return CausalArray.load(file=filename_groundtruth.format('', 'all'))
def gt_inclusive_physical():
    # return groundtruth(curated=False, gt_type='physical')
    return CausalArray.load(file=filename_groundtruth.format('', 'physical'))
def gt_inclusive_genetic():
    # return groundtruth(curated=False, gt_type='genetic')
    return CausalArray.load(file=filename_groundtruth.format('', 'genetic'))

###############
# Gene naming
###############
def make_gene_mapping(add_self_maps=True, verbose=True):
    """Run macro to update json with two mappings:
        - All map from {systematic, standard, synonyms} --> systematic
        - Standard map from standard --> systematic
    
        Settings
        - systematic keeps original SGD lettercase, the rest is set uppercase
        - Add systematic and standard naming to the 'all' mapping, and only add those synonyms that are new and unique.
        - Keep the duplicates found.
        - The aliasses are not used as they contain the same information as synonyms, but less organized by whitespace separation.
        - The primaryID's are not used currently as identifiers in the data.
        - Name convention in SGD query, with decreasing importance:
          * Systematic = row['secondaryIdentifier']; i.e. YDR155C
          * Standard = row['symbol'] i.e. CPR1
          * Synonymes = row['synonyms.values'] i.e. CYP1, CPH1 (one row each)
          * (not used) Aliasses = row['sgd.alias'] i.e. CYP1 CPH1 (whitespace separated) 
          * (not used) Primary = row['primaryIdentifier'] ; i.e. S000002562
    
    Example output:
        Unique names in systematic: 7296; standard: 5350; synonym: 24237;
        Total map size: 25770
        Duplicates: 361 
    
    Args:
        add_self_maps (bool, optional): Description
        verbose (bool, optional): Description
    """
    start_time = time.time()

    gene_set = set()
    symbol_set = set()
    synonyms_set = set()

    map_all_to_systematic = {}        # one-to-one map from {systematic, standard, synonym} -> systematic.
    # map_from_systematic = {}        # one-to-many (list) map from systamic -> aliases.
    map_standard_to_systematic = {}   # one-to-one map from standard -> systematic.

    # sgd_service = Service('http://yeastmine.yeastgenome.org/yeastmine/service', token ='j1X4vbH5u0I547P60980')
    sgd_service = Service('http://yeastmine.yeastgenome.org/yeastmine/service')
    query = sgd_service.new_query('Gene')
    query.add_view('secondaryIdentifier', 'symbol') 
    query.add_sort_order('Gene.secondaryIdentifier', 'ASC')
    query.add_constraint('organism.shortName', '=', 'S. cerevisiae', code = 'A')
    # query.add_constraint('featureType', '=', 'ORF', code = 'B')
    # query.set_logic('A and B')

    ### Map STANDARD --> SYSTEMATIC ###
    count = 0
    for row in query.rows():
        count += 1
        gene_set.add(row['secondaryIdentifier'])
        if row['symbol']:
            symbol_set.add(row['symbol'])
            # print count, row['symbol']
            if row['secondaryIdentifier']:
                map_standard_to_systematic[row['symbol'].upper()] = row['secondaryIdentifier']

    # ### Map ALL --> SYSTEMATIC ###
    # # standard --> systematic
    # {systematic, standard} --> systematic
    if add_self_maps:
        for row in query.rows():
            if row['secondaryIdentifier']:
                map_all_to_systematic[row['secondaryIdentifier'].upper()] = row['secondaryIdentifier']
                if row['symbol']:
                    map_all_to_systematic[row['symbol'].upper()] = row['secondaryIdentifier']

    # synonym --> systematic
    synonym_duplicates = {}
    query = sgd_service.new_query('Gene')
    query.add_view('secondaryIdentifier', 'synonyms.value') 
    query.add_sort_order('Gene.secondaryIdentifier', 'ASC')
    query.add_constraint('organism.shortName', '=', 'S. cerevisiae', code = 'A')

    for row in query.rows():
        if row['synonyms.value'] and row['secondaryIdentifier']:
            syn, secId = row['synonyms.value'].upper(), row['secondaryIdentifier']
            synonyms_set.add(syn)
            syn_in_dup = syn in synonym_duplicates.keys()
            if syn in map_all_to_systematic.keys():
                if syn_in_dup:
                    synonym_duplicates[syn].add(secId)
                    # print 'duplicate synonym:', syn, '-->', synonym_duplicates[syn]
                    if verbose: print 'del duplicated', syn, '-->', map_all_to_systematic[syn], 'from original map'
                    del map_all_to_systematic[syn]
                else:
                    pass
            else: 
                if syn_in_dup:
                    synonym_duplicates[syn].add(secId)
                    # print 'duplicate synonym:', syn, '-->', synonym_duplicates[syn]
                else:
                    map_all_to_systematic[syn] = secId
                    synonym_duplicates[syn] = set([secId])

    # Remove all items that have correctly been added to the mapping from the synonym_duplicates map, what remains is real duplicates
    for item in map_all_to_systematic.keys():
        if item in synonym_duplicates.keys():
            del synonym_duplicates[item]

    if verbose: 
        print 'Unique names in systematic: {}; standard: {}; synonym: {};'.format(len(gene_set), len(symbol_set), len(synonyms_set))
        print 'Total map size:', len(set(map_all_to_systematic.keys()))
        print 'Duplicates:', len(set(synonym_duplicates.keys()))

    with open(sgd_mapping_file, 'w') as f:
        json.dump(map_all_to_systematic, f)
    with open(sgd_mapping_file, 'w') as f:
        json.dump(map_standard_to_systematic, f)
    
    temp_ = {}
    for key in synonym_duplicates.keys():
        temp_[key] = list(synonym_duplicates[key])
        if verbose: print 'duplicate:', key, '-->', temp_[key]
    synonym_duplicates = temp_
    with open(_filename_duplicates, 'w') as f:
        json.dump(synonym_duplicates, f)

    if verbose: print 'Time:', time.time() - start_time


def gene_mapping(gene_list=None, verbose=False):
    """Uniquely map from gene aliasses onto second identifier
    
    Args:
        gene_list (None, optional): Description
        verbose (bool, optional): Description
    
    Returns:
        Dict or list of renamed genes if gene_list is not None.
    
    Deleted Parameters:
        only_standard (bool, optional): Description
    """
    if not os.path.exists(sgd_mapping_file):
        make_gene_mapping()

    with open(sgd_mapping_file, 'r') as f:
        sgd_mapping = json.load(f)
        # unicode mapping... convert to uppercase string
        sgd_mapping = dict((i.encode('ascii').upper(),j.encode('ascii').upper()) for i,j in sgd_mapping.items()) 

    if gene_list is None:
        return sgd_mapping

    # make sure strings are ASCII
    if type(gene_list[0]) is not str:
        gene_list = [i.encode('ascii') for i in gene_list]

    # get the result fast..
    result = [sgd_mapping[i.upper()] for i in gene_list if i.upper() in sgd_mapping.keys()]
    
    # print not equal items
    if verbose:
        # not_equal_items = ((i, result[idx]) for (idx, i) in enumerate(gene_list) if i.upper() is not result[idx])
        not_equal_items = ((i,j) for (i,j) in zip(gene_list, result) if i.upper() != j)
        for i,j in not_equal_items:
            print 'RENAMED: {} --> {}'.format(i,j)
    return result

# quick acces dict
mapping = gene_mapping()

# if __name__ == '__main__':
    
    # make_all_groundtruths()
    
    # for type in ('genetic', 'physical', 'all'):
    #     for curated in [True, False]:
    #         print type, curated, groundtruth(curated=curated, type=type).array.sum()
    #         print '   fraction:', groundtruth(curated=curated, type=type).array.sum() / float(groundtruth(curated=curated, type=type).array.size)

    # # pass optional argument to script
    # if len(os. sys.argv) > 1:
    #     if (sys.argv[1] == '0' or sys.argv[1] == 'False' or sys.argv[1] == 'false'):
    #         curated = False


