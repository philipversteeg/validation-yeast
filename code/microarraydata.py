"""Data classes to be called: Hughes, Kemmeren and Simulated microarray expression sets."""
import os
import random
import numpy as np
import h5py
from copy import deepcopy
from libs import CausalArray

# SERVER = os.sys.platform != 'darwin'
SERVER = os.sys.platform != 'darwin' # true if on linux or windows
if SERVER:
    DATAPATH = '/zfs/ivi/causality/data/yeast'
else:
    DATAPATH = os.path.abspath('{}/../data'.format(os.path.dirname(__file__) 
        if os.path.dirname(__file__) != '' else '.'))


class MicroArrayData(object):
    """Gene observational and interventional expression data from micro-arrays
    
    Attributes:
        file (TYPE): Descriptionn
        genes (list): p gene systematic identifiers
        inter (numpy 2darray): p genes x q interventional measurements 
        mutants (list): q mutant systematic identifiers
        name (TYPE): Description
        obser (numpy 2darray): p genes x n observational measurements 
        observations (TYPE): Description
        name (TYPE)
    """

    ### Constructor and I/O methods ###
    def __init__(self, obser, inter, genes, mutants,
                 observations=None, file=None, name='MicroArrayData'):
        # easy loading
        # unique entries for all
        assert len(set(genes)) == len(genes)
        assert len(set(mutants)) == len(mutants)
        if observations: assert len(set(observations)) == len(observations)
        # construct instance variables
        self.obser = obser
        self.inter = inter
        self.genes = genes
        self.mutants = mutants
        self.name = name
        self.file = file
        self.observations = observations or range(obser.shape[1])
        # check dimensions
        self.obser.shape == (len(self.genes),len(self.observations))
        self.inter.shape == (len(self.genes),len(self.mutants))

    def __print__(self):
        print self.name

    def __repr__(self):
        return ('[{self.name}] '
                # '\tObser:\t{self.obser.shape:10s}\n'
                # '\tInter:\t{self.inter.shape:10s}'
                'Obser:\t{self.ngenes} genes X {self.nobs} observations\n'
                '{space}Inter:\t{self.ngenes} genes X {self.nmutants} interventions'
                '{file}').format(self=self, space=(len(self.name) + 3) * ' ', 
                file=('' if self.file is None else '\n{space}File:\t{file_loc}'.format(
                    space=(len(self.name) + 3) * ' ', file_loc=os.path.relpath(self.file))))

    @classmethod
    def load(cls, file, name='MicroArrayData', verbose=True):
        """ Load from saved hdf5 file. This can be either a full data file or a 
        reduced settings file.
        """
        if os.path.splitext(file)[1] == '.hdf5': # hdf5 files
            with h5py.File(file, 'r') as fid:
                try: 
                    obser = fid['/obser/data'][...]
                    inter = fid['/inter/data'][...]
                    genes = list(fid['/ids/genes'][...])
                    mutants = list(fid['/ids/mutants'][...])
                    if verbose: print '[{0}] data at {1} loaded.'.format(name, 
                        file)
                    return cls(obser=obser, inter=inter, genes=genes, 
                        mutants=mutants, file=file, name=name)
                except KeyError:
                    # reduced data file
                    gen = list(fid['/genes'][...])
                    mut = list(fid['/mutants'][...])
                    obs = list(fid['/observations'][...])

                    # use stored class ifnormation to load correct class
                    orig_class_name = str(fid['/orig/class'][...]).split('.')
                    orig_class = getattr(os.sys.modules[__name__], 
                        orig_class_name[-1].split('\'')[0])
                    orig_data = orig_class.load(verbose=False)

                    # only load necessary ones!
                    if orig_data.ngenes == len(gen): gen = None
                    if orig_data.nmutants == len(mut): mut = None
                    if orig_data.nobs == len(obs): obs = None
                    result = orig_data.select(gen=gen, mut=mut, obs=obs)
                    result._file = file
                    if verbose: print ('[{0}] settings data at {1} loaded.'
                        .format(result.__class__.__name__, file))
                    return result
        else:
            raise IOError('{0} could not be loaded'.format(file))

    # alias for backwards compatibility
    @classmethod
    def loadfile(cls, file, name='MicroArrayData', verbose=True):
        return cls.load(file=file, name=name, verbose=verbose)

    def save(self, file, settings_only=True, compression='gzip', verbose=True):
        assert os.path.splitext(file)[1] == '.hdf5'
        self._file = file
        if settings_only:
            with h5py.File(file, 'w') as fid:
                fid.create_dataset('/genes', data=self.genes)
                fid.create_dataset('/mutants', data=self.mutants)
                fid.create_dataset('/observations', data=self.observations)
                # fid.create_dataset('/orig/class', data=str(type(self)))
                fid.create_dataset('/orig/class', data=str(self.__class__))
            if verbose: print '[{0}] settings saved at {1}.'.format(self.name, 
                os.path.relpath(file))
        else:
            with h5py.File(file, 'w') as fid:
                fid.create_dataset('/obser/data', data=self.obser, 
                    compression=compression)
                fid.create_dataset('/inter/data', data=self.inter, 
                    compression=compression)
                fid.create_dataset('/ids/genes', data=self.genes)
                fid.create_dataset('/ids/mutants', data=self.mutants)
            if verbose: print '[{0}] data saved at {1}.'.format(self.name, 
                os.path.relpath(file))

    # alias for backwards compatibility
    def savefile(self, file, verbose=True):
        self.save(file=file, verbose=verbose)

    @property
    def file(self):
        return self._file

    @property
    def folder(self):
        if self.file is None: 
            return None
        else:
            return os.path.split(self.file)[0]

    @file.setter
    def file(self, filename):
        assert filename is None or os.path.exists(filename)
        self._file = filename

    @property
    def default_folder(self):
        return ('{self.name}_{self.ngenes}gen_{self.nobs}obs_{self.nmutants}mut'
            .format(self=self).lower())

    # # For pickling the class (should not be necessary now...)
    # def __getstate__(self):
    #    print '[{}] Pickling... {}'.format(self.__class__.__name__, self.name)
    #    return self.__dict__
    # def __setstate__(self, state):
    #    self.__dict__ = state
    #    print '[{}] Unpickling... {}'.format(self.__class__.__name__, self.name)

    @property
    def ngenes(self):
        assert self.obser.shape[0] == len(self.genes)
        assert self.inter.shape == (len(self.genes),len(self.mutants))
        return len(self.genes)

    @property
    def nobs(self):
        return self.obser.shape[1]

    @property
    def nmutants(self):
        assert self.inter.shape == (len(self.genes),len(self.mutants))
        return len(self.mutants)

    @property
    def shape(self):
        return self.ngenes, self.nmutants, self.nobs

    @property
    def map_genes(self):
        """Dict that maps genes: systematic -> row i in obser"""
        return dict(zip(self.genes, range(len(self.genes))))

    @property
    def map_mutants(self):
        """Dict that maps mutants: systematic -> column j in obser"""
        return dict(zip(self.mutants, range(len(self.mutants))))

    def intpos(self, mutants=None):
        mutants = mutants or self.mutants
        return [self.map_genes[i] for i in mutants]

    def intpos_R(self, mutants=None):
        """Intervention position in R, different due to one-based arrays"""
        mutants = mutants or self.mutants
        return [self.map_genes[i] + 1 for i in mutants]

    @property
    def obser_inter_joint(self):
        """Joint array of N observations and M interventions.

        First M indices correspond to interventions, the remaining 
        to observations.
        
        Returns:
            ndarray: joint array
        """
        return np.hstack((self.inter, self.obser)) 

    @staticmethod
    def _standardize(data):
        # subtract mean and divide by std:
        return ((data.T - np.mean(data, axis=1))/ np.std(data, axis=1)).T

    @property
    def obser_std(self):
        return self._standardize(self.obser)

    @property
    def inter_std(self):
        return self._standardize(self.inter)

    @property
    def obser_inter_joint_std(self):
        return self._standardize(self.obser_inter_joint)

    def _sort_data(self, list_genes=None, list_mutants=None, by_name=False, 
                   return_index=False):
        """Return sorted obser and inter by list_genes nad list_mutants.
        """
        if by_name:
            if list_genes is not None:
                 list_genes = self.map_genes[list_genes]
            if list_mutants is not None:
                 list_mutants = self.map_genes[list_mutants]
        if list_genes is None:
            list_genes = range(len(self.genes))
        if list_mutants is None:
            list_mutants = range(len(self.mutants))
        if not type(list_genes) == list or not type(list_mutants) == list:
            raise ValueError('Only input list or None!')
        
        if return_index: 
            return (self.obser[list_genes,], self.inter[list_genes,:]
                [:,list_mutants], list_genes, list_mutants)
        return self.obser[list_genes,], self.inter[list_genes,:][:,list_mutants]

    ### data selection and sampling methods ###
    def select(self, gen=None, mut=None, obs=None, name=None,
               file=None, copy=True, seed=None):
        """ Return subsampled microarraydata according to parameters.

        Important: if arguments gen and mut are both passed, mutants are 
        selected within the gen pool.
        """
        if seed is not None: random.seed(seed)
        if gen is None and mut is None and obs is None:
            if copy:
                return deepcopy(self)
            else:
                return self

        # observations
        if type(obs) is int:
            obs = random.sample(self.observations, obs)
        # more complex case where we want sample size of genes without a mutant
        # of (gen - mut).
        if type(gen) is int:
            if mut is None:
                gen = random.sample(self.genes, gen)
                mut = [i for i in gen if i in self.mutants]
                print ('[{0}] WARNING: No mutant list given; randomly selected '
                    '{1} mutants within {2} genes...'.format(self.name, 
                        len(mut), len(gen)))
            else:
                if type(mut) is int:
                    mut = random.sample(self.mutants, mut)
                gen = mut + random.sample([i for i in self.genes if i not in
                self.mutants], gen - len(mut))
        if type(mut) is int:
            if gen is None:
                mut = random.sample(self.mutants, mut)
            else: 
                mut = random.sample([i for i in self.mutants if i in gen], mut)
                print ('[{0}] WARNING: No gene list given; randomly selected '
                    '{1} mutants within {2} genes...'.format(self.name, 
                        len(mut), len(gen)))
            # no need to check for type(gen) is int, it's covered above.

        assert len(gen) <= self.ngenes if gen is not None else True
        assert len(mut) <= self.nmutants if mut is not None else True
        assert len(obs) <= self.nobs if obs is not None else True
        assert len(mut) <= len(gen) if gen is not None and mut is not None else True

        # create return instance
        if copy:
            instance = deepcopy(self)
            if name is not None:
                instance.name = name
            instance.file = None
            # do mutants before genes, ordering important
            if obs is not None:
                instance.obser = instance.obser[:,obs]
                instance.observations = range(len(obs))
            if gen is not None:
                instance.obser = instance.obser[[self.map_genes[i] for i in gen],:]
                instance.inter = instance.inter[[self.map_genes[i] for i in gen],:]
                instance.genes = gen
            if mut is not None:
                instance.inter = instance.inter[:,[self.map_mutants[i] for i in mut
                if i in instance.mutants]]
                instance.mutants = mut
            if file: instance.save(file=file, settings_only=True)
            return instance
        else:
            if name is not None:
                self.name = name
            # do mutants before genes, ordering important
            if obs is not None:
                self.obser = self.obser[:,obs]
                self.observations = range(len(obs))
            if gen is not None:
                self.obser = self.obser[[self.map_genes[i] for i in gen],:]
                self.inter = self.inter[[self.map_genes[i] for i in gen],:]
                self.genes = gen
            if mut is not None:
                self.inter = self.inter[:,[self.map_mutants[i] for i in mut
                if i in self.mutants]]
                self.mutants = mut
            return self


    def split(self, mut=None, obs=None, name='New', name_remainder='Rest', 
              seed=None):
        """Split observational and interventional data.

        Args:
           mut: A list of mutants to be split or number of mutants to randomly 
           split.
           obs: A list of observation indices to be split or number of mutants 
           to randomly split.

        Returns:
           Two MicroArrayData instances class split along the parameters.
        """
        if seed is not None:
            random.seed(seed)
        # check input
        if mut is None and obs is None:
            raise 'No input given'
        if type(obs) is int:
            obs = random.sample(range(self.nobs), obs)
        if obs is None:
            obs = []
        if type(mut) is int:
            mut = random.sample(self.mutants, mut)
        if mut is None:
            mut = []
        assert len(mut) <= self.nmutants 
        assert len(obs) <= self.nobs
        return self.select(mut=mut, obs=obs, name=name), self.select(
            mut=[i for i in self.mutants if i not in mut], 
            obs=[i for i in range(self.nobs) if i not in obs], 
            name=name_remainder)


    ### interventional measures ###
    @property
    def x_ij(self):
        """x_ij = A_ij - <X_i.> """
        return np.subtract(self.inter.T, self.obser.mean(axis=1)).T

    @property
    def z_ij(self):
        """z_ij = (A_ij - <X_i.>)/ (\sigma_i.) """
        return np.divide(np.subtract(self.inter.T, self.obser.mean(axis=1)), 
            self.obser.std(axis=1)).T

    @property
    def ratio_x_ij(self):
        """r_{x_{ij}} = x_{ij} / x_jc(j) """
        return np.divide(np.subtract(self.inter.T, self.obser.mean(axis=1)).T,
                         self.x_ij[[self.map_genes[i] for i in 
                         self.mutants],:].diagonal())

    @property
    def ratio_z_ij(self):
        """ratio_z_{ij} = z_{ij} / z_{jc(j)} """
        # return np.divide( np.divide(np.subtract(self.inter.T, self.obser.mean(axis=1)),self.obser.std(axis=1)).T,
        # print 'Shape z_ij:', self.z_ij.shape
        # print 'Shape ratio_z_ij', np.divide(self.z_ij,
                         # self.z_ij[[self.map_genes[i] for i in self.mutants],:].diagonal()).shape
        # print 'Diagonal', self.z_ij[[self.map_genes[i] for i in self.mutants],:].diagonal().shape
        return np.divide(self.z_ij,
                         self.z_ij[[self.map_genes[i] for i in self.mutants],:].diagonal())

    @property
    def z_ij_robust(self):
        """ return subsampled z_ij_robust according to parameters median(X_i.>) / (iqr(x_i.)). """
        def iqr(row):
            return np.subtract(*np.percentile(row, [75, 25]))
        return np.divide(np.subtract(self.inter.T, np.median(self.obser,axis=1)),np.apply_along_axis(iqr, 1, self.obser)).T

    ### ground-truths ###
    # TODO: split of in separate class.
    def gt(self, name, successful_ints=False, change_strint_name=False):
        """Groundtruth measures as a CausalArray.
        Args:
            name (str): type of groundtruth
        
        Returns:
            CausalArray: groundtruth
        """
        causes = self.mutants
        effects = self.genes
        if name in ('gt_abs', 'gt.abs', 'abs'):
            array = self.gt_abs
            units = 'j'
        elif name in ('gt_rel', 'gt.rel' , 'rel'):
            array = self.gt_rel
            units = 'j/i'
        elif name in ('gt_abs_norm', 'gt.abs.norm', 'abs_norm', 'abs.norm'):
            array = self.gt_abs_norm
            units = '1'
        elif name in ('gt_abs_norm_robust', 'gt.abs.norm.robust', 'abs_norm_robust', 'abs.norm.robust'):
            array = self.gt_abs_norm_robust
            units = '1'
        elif name in ('gt_rel_norm', 'gt.rel.norm', 'rel_norm', 'rel.norm'):
            array = self.gt_rel_norm
            units = '1'
        elif name in ('gt_nature', 'gt.nature' , 'nature'):
            array = self.gt_nature
            units = '1'
        elif name in ('gt_sie', 'gt.sie' , 'sie'):
            array = self.gt_sie
            units = '0'
        else:
            raise Exception('Unknown groundtruth %s!' % name)
        ## only select successfull ints, put the others to zero OR smaller 
        # if successful_ints:
        #     array[[self.map_mutants[i] for i in list(set(self.mutants).difference(set(self.successful_ints)))],:] = 0
        if successful_ints:
            causes = self.successful_ints
            if len(causes) == 0:
                raise Exception('No successful interventions found!')
            array = array[[self.map_mutants[i] for i in causes],]
            if change_strint_name:
                name += '_strint'
        return CausalArray(array=array, causes=causes, effects=effects, units=units, name=name)
        
    @property
    def successful_ints(self):
        """Used also in SIE, give list of successfull ('strong') interventions"""
        result = []
        for mut in self.mutants:
            id_gen = self.map_genes[mut]        # get index in gene array
            id_mut = self.map_mutants[mut]      # gt index in mutant array
            ids_min_mut = range(self.nmutants)  # all indices minus this mutant
            ids_min_mut.remove(id_mut)
            if self.inter[id_gen, id_mut] <= min(np.min(self.obser[id_gen,]), np.min(self.inter[id_gen, ids_min_mut])):
                result.append(mut)
        return result
        # TODO: optimize
        # @property
        # def successful_ints(self):
        #     return [mut for mut in self.mutants if self.inter[self.map_genes[mut], id_mut = self.map_mutants[mut]] <= min(np.min(self.obser[self.map_genes[mut],]), np.min(self.inter[self.map_genes[mut], ids_min_mut]))]

    ## GT Measures
    def gt_strong_ints_effects(self, effect_percentile=None):
        """ SIE """
        if effect_percentile is None:
            min_per_gene = np.apply_along_axis(func1d=min, axis=1, arr=self.obser_inter_joint)  # obsmin 
            max_per_gene = np.apply_along_axis(func1d=max, axis=1, arr=self.obser_inter_joint)  # obsmax
        else:
            max_per_gene = np.apply_along_axis(np.percentile, 1, np.abs(self.obser_inter_joint), (100 - effect_percentile))
            min_per_gene = - max_per_gene

        result = np.zeros((self.nmutants, self.ngenes), dtype='bool')
        for mut in self.successful_ints:
            id_gen = self.map_genes[mut]       # get index in gene array
            id_mut = self.map_mutants[mut]     # gt index in mutant array
            # array of indices of strong effects  # i.e. either larger or smaller than the intervention effect
            effect_indices = np.argwhere(np.logical_or(self.inter[:,id_mut] <= min_per_gene, self.inter[:, id_mut] >= max_per_gene))
            for ef in effect_indices:
                result[id_mut, ef] = True
            return result

    @property
    def gt_sie(self):
        return self.gt_strong_ints_effects(effect_percentile=None)

    @property
    def gt_rel(self):
        """Returns gt.rel[causes,effects]"""
        return np.abs(self.ratio_x_ij).T

    @property
    def gt_rel_norm(self):
        """Returns [causes,effects]"""
        return np.abs(self.ratio_z_ij).T

    @property
    def gt_abs(self):
        """Returns [causes,effects]"""
        return np.abs(self.x_ij).T

    @property
    def gt_abs_norm(self):
        """Returns [causes,effects]"""
        return np.abs(self.z_ij).T

    @property
    def gt_abs_norm_robust(self):
        return np.abs(self.z_ij_robust).T


    def gt_abs_norm_damped(self, damping_term):
        """Returns [causes,effects]"""
        return np.abs(np.divide(np.subtract(self.inter.T, self.obser.mean(axis=1)), self.obser.std(axis=1) + damping_term)).T


    def gt_abs_cutoff(self, num_std_cutoff):
        # x_ij = gene x mut
        tmp_ = self.x_ij[[self.map_genes[i] for i in self.mutants],:].diagonal()\
        > num_std_cutoff * self.obser.std(axis=1)[[self.map_genes[i] for i in self.mutants]].astype('int')
        return np.multiply(np.abs(self.x_ij), tmp_).T

    @property
    def gt_nature(self):
        """ Used in nature methods paper."""
        a = np.divide(self.inter.T, np.std(self.inter, axis=1))
        means = a.mean(axis=0)    
        numerator = a - means
        denominator = np.diag(numerator[:, [self.map_genes[i] for i in self.mutants]])
        return np.abs(np.divide(numerator.T, denominator)).T # transpose for broadcasting... for broadcasting reason

    @property
    def gt_nature_methods(self): 
        """Used in nature_methods paper; same as gt_rel_norm but applied on standardized data."""
        a = self.inter_std.T  
        means = a.mean(axis=0)    
        numerator = a - np.multiply((1/float(len(means) - 1)),(len(means) * means - a))

        # denominator is the same, but just for diagonal of genes x mutatns
        denominator = np.diag(numerator[:, [self.map_genes[i] for i in self.mutants]])
        return np.abs(np.divide(numerator.T, denominator)).T # transpose for broadcasting... for broadcasting reason

    @property
    def gt_nature_methods_slow(self):
        """Used in nature_methods paper; same as gt_rel_norm but applied on standardized data.
             * slow non-vectorized version for checking...
        """
        a = self.inter_std.T
        means = a.mean(axis=0)
        numerator = np.zeros(a.shape, dtype=float) # means matrix A_{-i, <j>} means over column j with i-th term removed
        for (i,j), _ in np.ndenumerate(a): # for each intervention
            # if j == 0: print 'gt.nature: processing {} / {} ...'.format(i, self.nmutants) 
            # indices = [x for x in xrange(a.shape[0]) if x != i] # all except one
            # mean = a[indices,j].mean()
            # numerator[i,j] = a[i,j] - mean
            numerator[i,j] = a[i,j] - (1/float(len(means) - 1)) * (len(means) * means[j] - a[i,j])
        
        # denominator is the same, but just for diagonal of genes x mutatns
        denominator = np.diag(numerator[:, [self.map_genes[i] for i in self.mutants]])
        return np.abs(np.divide(numerator.T, denominator)).T # transpose for broadcasting... for broadcasting reason

class Kemmeren(MicroArrayData):
    """MicroArrayData from Kemmeren (2014).
    
    Attributes:
        extprot_inter (list): Extraction protocol for q interventions
        extprot_obser (list): Extraction protocol for n observations
        name (str): Object names
    """
    file_default = os.path.normpath('{}/kemmeren/kemmeren.hdf5'.format(DATAPATH))

    ### Constructor and factory methods ###
    def __init__(self, obser=None, inter=None, genes=None, mutants=None, file=None, observations=None, name='Kemmeren'):
        # for lazy loading.. Better way of copying all this?
        if obser is None and inter is None and genes is None and mutants is None and observations is None and file is None:
            self.obser = inst.obser
            self.inter = inst.inter
            self.genes = inst.genes
            self.genes = inst.genes
            self.mutants = inst.mutants
            self.file = inst.file
            self.observations = inst.observations
            self.name = inst.name
            self.extprot_obser = inst.extprot_obser
            self.extprot_inter = inst.extprot_inter
            self._extprot_available = inst._extprot_available
        else:
            super(Kemmeren, self).__init__(obser=obser, inter=inter, genes=genes, mutants=mutants, file=file, name=name)

    def __repr__(self):
        return  '{}\n{space}Extpr:\t{self.extprot_available}'.format(super(Kemmeren, self).__repr__(), 
            space=(len(self.name) + 3) * ' ', self=self)

    @classmethod
    def load(cls, file=None, name='Kemmeren', verbose=True):
        file = file or cls.file_default
        data = super(Kemmeren, cls).load(file=file, name=name, verbose=verbose)
        with h5py.File(file, 'r') as fid:
            try:
                data.extprot_obser = fid['/obser/extplate'][...]
                data.extprot_inter = fid['/inter/extplate'][...]
                if verbose: print '[{0}] extraction protocol loaded.'.format(name)
                data._extprot_available = True
            except KeyError:
                if verbose: print '[{0}] no extraction protocol found!'.format(name)
                data._extprot_available = False
        return data
    
    # alias for backwards compatibility
    @classmethod
    def loadfile(cls, file=None, name='Kemmeren', verbose=True):
        return cls.load(file=file, name=name, verbose=verbose)

    @classmethod
    def loaddata(cls, obser, inter, genes, mutants, extprot_obser=None, extprot_inter=None, name='Kemmeren'):
        cls._extprot_available = False
        if extprot_obser is not None:
            cls.extprot_obser = extprot_obser
            cls._extprot_available = True
        if extprot_inter is not None:
            cls.extprot_inter = extprot_inter
            cls._extprot_available = True
        return super(Kemmeren, cls).loaddata(obser, inter, genes, mutants, name)

    def save(self, file, **kwargs):
        super(Kemmeren, self).save(file=file, **kwargs)
        if self.extprot_available:
            with h5py.File(file, 'r+') as fid:
                 fid.create_dataset('/obser/extplate', data=self.extprot_obser)
                 fid.create_dataset('/inter/extplate', data=self.extprot_inter)

    # deprecated
    def savefile(self, file, verbose=True):
        self.save(file=file, verbose=verbose)

    @property
    def extprot_available(self):
        return self._extprot_available

    ### data selection and sampling methods ###
    def rem_mutants(self, to_remove, name=None, strict=True):
        _instance = super(Kemmeren, self).rem_mutants(to_remove, name, strict)
        if self.extprot_available: 
            _instance.extprot_inter = self.extprot_inter[[self.map_mutants[i] for i in _instance.mutants]]
        return _instance

    def sel_mutants(self, to_keep, name=None, strict=True):
        _instance = super(Kemmeren, self).sel_mutants(to_keep, name, strict)
        if self.extprot_available: 
            _instance.extprot_inter = self.extprot_inter[[self.map_mutants[i] for i in _instance.mutants]]
        return _instance

    def rem_observations(self, to_remove, name=None):
        if type(to_remove) is int:
            to_remove = random.sample(range(self.nobs), to_remove)
        _instance = super(Kemmeren, self).rem_observations(to_remove, name)
        if self.extprot_available:
            np.delete(_instance.extprot_obser, to_remove, axis=0)
        return _instance

    def sel_observations(self, to_keep, name=None):
        if type(to_keep) is int:
            to_keep = random.sample(range(self.nobs), to_keep)
        _instance = super(Kemmeren, self).sel_observations(to_keep, name)
        if self.extprot_available:
            _instance.extprot_obser = self.extprot_obser[to_keep]
        return _instance

    def rem_measurements(self, to_remove, name=None):
        self.rem_observations(to_remove, name=None)

    def sel_measurements(self, to_remove, name=None):
        self.sel_observations(to_remove, name=None)

class Simulated(MicroArrayData):
    """ Simulated MicroArrayData.
    
    Attributes:
        extprot_inter (list): Extraction protocol for q interventions
        extprot_obser (list): Extraction protocol for n observations
        name (str): Object name
    """
    file_default = os.path.normpath('{}/simulated/sim_kemmeren.hdf5'.format(DATAPATH))
    simulated = True

    @classmethod
    def load(cls, file=None, name='Simulated', verbose=True):
        file = file or cls.file_default
        data = super(Simulated, cls).load(file=file, name=name, verbose=verbose)
        with h5py.File(file, 'r') as fid:
            cls.b = fid['/sim/B'][...]
            cls.num_conf = fid['/sim/numconf'][...]
            cls.noise_cov = fid['/sim/noisecov'][...]
        return data

    def save(self, file, compression='gzip', **kwargs):
        super(Simulated, self).save(file=file, **kwargs)
        with h5py.File(file, 'w') as fid:
            fid.create_dataset('/sim/B', data = self.b, compression=compression)
            fid.create_dataset('/sim/numconf', data = self.num_conf, compression=compression)
            fid.create_dataset('/sim/noisecov', data = self.noise_cov, compression=compression)

    ### data selection and sampling methods ###
    # TODO: cleanup
    def rem_genes(self, to_remove, name=None, strict=True):
        _instance = deepcopy(self)
        # if name is None: name = '{0}_rem'.format(self.name)
        _instance.name = name
        try: 
            for it in to_remove:
                _instance.genes.remove(it)
            _instance.obser = np.delete(self.obser, [self.map_genes[i] for i in to_remove], axis=0)
            _instance.inter = np.delete(self.inter, [self.map_genes[i] for i in to_remove], axis=0)
            if self.simulated: _instance.b = np.delete(np.delete(self.b, [self.map_genes[i] for i in to_remove], axis=1), [self.map_genes[i] for i in to_remove], axis=0)
        except (KeyError, ValueError):
            if strict: raise Exception('MatchingError')
        return _instance

    def sel_genes(self, to_keep, name=None, strict=True):
        if type(to_keep) is int:
            to_keep = random.sample(self.genes, to_keep)
        _instance = deepcopy(self)
        # if name is None: name = '{0}_sel'.format(self.name)
        _instance.name = name
        try: 
            for it in self.genes:
                 if it not in to_keep:
                    _instance.genes.remove(it)
            _instance.obser = self.obser[[self.map_genes[i] for i in to_keep],:]
            _instance.inter = self.inter[[self.map_genes[i] for i in to_keep],:]
            _instance.b = self.b[[self.map_genes[i] for i in to_keep],:][:,[self.map_genes[i] for i in to_keep]]
        except (KeyError, ValueError) as e: 
            if strict: raise Exception('MatchingError: {}'.format(e))
        return _instance


class Hughes(MicroArrayData):
    """MicroArrayData from Hughes (2002).
    
    Attributes:
        file (str): Disk location of data (hdf5 format)
        name (TYPE): Object name
    """
    file_default = os.path.join(DATAPATH,'hughes/hughes.hdf5')

    def __init__(self, obser=None, inter=None, genes=None, mutants=None, observations=None, file=None, name='Hughes'):
        super(Hughes, self).__init__(obser=obser, inter=inter, genes=genes, mutants=mutants, file=file, name=name)

    @classmethod
    def load(cls, file=None, name='Hughes', verbose=True):
        file = file or cls.file_default
        return super(Hughes, cls).load(file=file, name=name, verbose=verbose)


if __name__ == '__main__':

    # KEMMEREN
    kem = Kemmeren.load()
    x_small = kem.select(gen=100, mut=50, obs=20)
    x_small.save('test_kem.hdf5')

    hug = Hughes.load()
    y_small = hug.select(gen=100, mut=50, obs=20)
    y_small.save('test_hug.hdf5')

    y_load = MicroArrayData.load('test_kem.hdf5')    
    print y_load
    # # print eval('')

    # x.select(gen=10, mut=10, obs=30, file_settings='test.hdf5')
    # print x

    # y = Hughes.load('test.hdf5')
    # print y
