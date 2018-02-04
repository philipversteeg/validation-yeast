"""Interventional scores computed on microarraydata."""
from ..microarraydata import MicroArrayData
from ..libs import CausalArray

def gt(data, name, successful_ints=False, change_strint_name=False):
    return data.gt(name, successful_ints=successful_ints, change_strint_name=change_strint_name)

def successful_ints(data):
    """Used also in SIE, give list of successfull ('strong') interventions"""
    result = []
    for mut in data.mutants:
        id_gen = data.map_genes[mut]        # get index in gene array
        id_mut = data.map_mutants[mut]      # gt index in mutant array
        ids_min_mut = range(data.nmutants)  # all indices minus this mutant
        ids_min_mut.remove(id_mut)
        if data.inter[id_gen, id_mut] <= min(np.min(data.obser[id_gen,]), np.min(data.inter[id_gen, ids_min_mut])):
            result.append(mut)
    return result

## GT Measures
def gt_sie(data, effect_percentile=None):
    """ SIE """
    if effect_percentile is None:
        min_per_gene = np.apply_along_axis(func1d=min, axis=1, arr=data.obser_inter_joint)  # obsmin 
        max_per_gene = np.apply_along_axis(func1d=max, axis=1, arr=data.obser_inter_joint)  # obsmax
    else:
        max_per_gene = np.apply_along_axis(np.percentile, 1, np.abs(data.obser_inter_joint), (100 - effect_percentile))
        min_per_gene = - max_per_gene

    result = np.zeros((data.nmutants, data.ngenes), dtype='bool')
    for mut in data.successful_ints:
        id_gen = data.map_genes[mut]       # get index in gene array
        id_mut = data.map_mutants[mut]     # gt index in mutant array
        # array of indices of strong effects  # i.e. either larger or smaller than the intervention effect
        effect_indices = np.argwhere(np.logical_or(data.inter[:,id_mut] <= min_per_gene, data.inter[:, id_mut] >= max_per_gene))
        for ef in effect_indices:
            result[id_mut, ef] = True
        return result

def gt_rel(data):
    """Returns gt.rel[causes,effects]"""
    return np.abs(data.ratio_x_ij).T

def gt_rel_norm(data):
    """Returns [causes,effects]"""
    return np.abs(data.ratio_z_ij).T

def gt_abs(data):
    """Returns [causes,effects]"""
    return np.abs(data.x_ij).T

def gt_abs_norm(data):
    """Returns [causes,effects]"""
    return np.abs(data.z_ij).T

def gt_abs_norm_robust(data):
    return np.abs(data.z_ij_robust).T


def gt_abs_norm_damped(data, damping_term):
    """Returns [causes,effects]"""
    return np.abs(np.divide(np.subtract(data.inter.T, data.obser.mean(axis=1)), data.obser.std(axis=1) + damping_term)).T


def gt_abs_cutoff(data, num_std_cutoff):
    # x_ij = gene x mut
    tmp_ = data.x_ij[[data.map_genes[i] for i in data.mutants],:].diagonal()\
    > num_std_cutoff * data.obser.std(axis=1)[[data.map_genes[i] for i in data.mutants]].astype('int')
    return np.multiply(np.abs(data.x_ij), tmp_).T

def gt_nature(data):
    """ Used in nature methods paper."""
    a = np.divide(data.inter.T, np.std(data.inter, axis=1))
    means = a.mean(axis=0)    
    numerator = a - means
    denominator = np.diag(numerator[:, [data.map_genes[i] for i in data.mutants]])
    return np.abs(np.divide(numerator.T, denominator)).T # transpose for broadcasting... for broadcasting reason

def gt_nature_methods(data):
    """Used in nature_methods paper; same as gt_rel_norm but applied on standardized data."""
    a = data.inter_std.T
    means = a.mean(axis=0)    
    numerator = a - np.multiply((1/float(len(means) - 1)),(len(means) * means - a))

    # denominator is the same, but just for diagonal of genes x mutatns
    denominator = np.diag(numerator[:, [data.map_genes[i] for i in data.mutants]])
    return np.abs(np.divide(numerator.T, denominator)).T # transpose for broadcasting... for broadcasting reason

def gt_nature_methods_slow(data): 
    """Used in nature_methods paper; same as gt_rel_norm but applied on standardized data.
    
    Note: slow non-vectorized version.
    """
    a = data.inter_std.T
    means = a.mean(axis=0)
    numerator = np.zeros(a.shape, dtype=float) # means matrix A_{-i, <j>} means over column j with i-th term removed
    for (i,j), _ in np.ndenumerate(a): # for each intervention
        # if j == 0: print 'gt.nature: processing {} / {} ...'.format(i, data.nmutants) 
        # indices = [x for x in xrange(a.shape[0]) if x != i] # all except one
        # mean = a[indices,j].mean()
        # numerator[i,j] = a[i,j] - mean
        numerator[i,j] = a[i,j] - (1/float(len(means) - 1)) * (len(means) * means[j] - a[i,j])
    
    # denominator is the same, but just for diagonal of genes x mutatns
    denominator = np.diag(numerator[:, [data.map_genes[i] for i in data.mutants]])
    return np.abs(np.divide(numerator.T, denominator)).T # transpose for broadcasting... for broadcasting reason