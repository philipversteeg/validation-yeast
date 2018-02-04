# stdlib
import os
import time
import abc
from functools import wraps
import tempfile
from subprocess import check_call

# math stats ml
import numpy as np
from scipy.stats import kendalltau, spearmanr
from scipy.spatial.distance import correlation as distancecorr

# local
from ..microarraydata import MicroArrayData
from ..libs import CausalArray
from ..libs.misc import save_array, load_array
from .. import config

OBSERMETHOD_DATATYPES = ['obser', 'inter', 'obser_inter_joint']

###
### Abstract Base Class
###
class Predictor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, processes=None, folder=None, file=None, name=None, save=None, verbose=None):
        """Abstract base class where all prediction estimator methods inherit from. 
        
        Need to overwrite at least the abstract fit method
        
        Args:
            processes (int, optional): by default single process, inherit also from MultiProcess to use multiple cores
            folder (str, optional): disk location where results are stored
            file (str, optional): disk location of optionally saved file relative to self.folder
            name (str, optional): name of the method
            save (bool, optional): if true, always save output
            verbose (bool or int, optional): set verbose level; default is off
        """
        self.processes = processes or 1
        self._name = name
        self.folder = folder or config.folder
        self._file = file
        self.save = save
        self._opts = {}  # dict used for storing / overriding / deleting task settings.
        if verbose is None: 
            self.verbose = False # default option
        else:
            self.verbose = verbose
        self.result = None
        if not os.path.exists(self.folder): 
            if self.verbose: print '[{}] Creating folder {}'.format(self.name_method, self.folder)
            os.makedirs(self.folder)

    @abc.abstractmethod
    def fit(self, data, bootstrap=False):
        """Overwrite and return self"""
        pass


    @staticmethod
    def fit_decorator(fit_function):
        """ Decorator for 'fit' instance methods.

            Wrapper for fit(self, data) instance method that returns a CausalArray object
            1. if self.folder + self.filename exists
                --> load this result
            2. execute and time the method
            3. if the duration > threshold
                --> save result the self.folder + self.filename 
        """
        @wraps(fit_function)
        def wrapper(self, *args, **kwargs):
            file_name = self.folder + '/' + self.file

            # if self.folder + self.filename exists, return that
            if os.path.exists(file_name):
                if self.verbose: print '[{}] existing result loaded at {}.'.format(self.name_method, 
                    os.path.relpath(file_name))
                self.result = CausalArray.load(file_name)
                return self
            
            # compute and time that result
            start_time = time.time() # attach start_time to instance for easy printing
            if self.verbose: print '[{}] fitting data...'.format(self.name_method)
            result_instance = fit_function(self, *args, **kwargs)
            # try:
            #     result_instance = fit_function(*args, **kwargs)
            # except Exception as e:
            #     raise Exception('Error with method %s' % self.name)
            result_time = time.time() - start_time

            if self.verbose: print '[{}] ...done in {:.2f}s.'.format(self.name_method, result_time)
            # check for empty CausalArray
            if config.warning_empty_result: 
                if result_instance.result.nonzero == 0: print '[{}] WARNING: empty CausalArray returned!'.format(self.name_method)
            # save if necessary
            if result_time > config.save_task_threshold or self.save:
                result_instance.result.save(file_name)
            return result_instance
        return wrapper

    @property
    def name_method(self):
        """General name of the prediction method (non-unique!)"""
        return self.__class__.__name__

    @property
    def name(self):
        """Unique name for a given method based on all callable options.
        
        If self._name instance is defined, this is used instead.
        """
        if self._name:
            return self._name
        else:
            return self.__class__.__name__  + '_' + '_'.join(['{}={}'.format(i,j) for i,j in self.options.iteritems()])

    def pretty_name(self, print_options):
        """ A pretty printable name."""
        return self.name.replace('_', ' ')

    @property
    def file(self):
        if self._file:
            return self._file
        else:
            return self.name + '.hdf5'

    @property
    def done(self):
        return self.result is not None

    def pickle_dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f, -1) # highest protocol available

    @staticmethod
    def pickle_load(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    @property
    def options(self):
        """Return dict of all real parameters that are not purely for computation."""
        # options_set = set(self.__class__.__init__.__code__.co_varnames).difference(
        #     set(('self', 'args', 'kwargs')))
        # options_dict = dict((opt, getattr(self, opt)) for opt in options_set) if options_set else {}
        # if hasattr(self, '_options'): # add overriden options
        #     print self._options
        #     for k, v in self._options.iteritems():
        #         options_dict[k] = v
        return dict((k, v) for k, v in self._options.iteritems() if v is not None)

    @property
    def _options(self):
        return self._opts

    @_options.setter
    def _options(self, opts):
        assert type(opts) is dict
        self._opts.update(opts)

    def __repr__(self):
        result = '[{self.__class__.__name__}] {self.name}'.format(self=self)
        for k, v in self.options.iteritems():
            result += '\n{space}{k}: {v}'.format(space=(len(self.__class__.__name__) + 3) * ' ', k=k, v=v)
        return result + '\n'


class MultiProcess(Predictor):
    """Use multiple processes for computation, default is equal to DEFAULT_PROCESSES"""
    def __init__(self, processes=None, *args, **kwargs):
        super(MultiProcess, self).__init__(processes=processes or config.ncores_method, *args, **kwargs)


class Dummy(MultiProcess):
    """Dummy method used for testing multiprocesses"""
    def __init__(self, sleep, *args, **kwargs):
        super(Dummy, self).__init__(*args, **kwargs)
        self.sleep = sleep

    @Predictor.fit_decorator
    def fit(self, data):
        print '...dummy starting', self.name
        time.sleep(self.sleep)
        print '...dummy finished', self.name
        # return CausalArray(array=np.random.random((data.ngenes, data.ngenes)), 
        #     causes=data.genes, effects=data.genes, name='dummy')
        self.result = CausalArray(array=np.random.random((data.ngenes, data.ngenes)), 
            causes=data.genes, effects=data.genes, name='dummy')
        return self


class Random(Predictor):
    """Random (Bernoulli with p=0.5) predictions"""
    unitstring = '1'

    def __init__(self, *args, **kwargs):
        super(Random, self).__init__(*args, **kwargs)

    @Predictor.fit_decorator
    def fit(self, data):
        self.result = CausalArray(array=np.random.binomial(n=1, p=0.5, size=(data.ngenes,data.ngenes)), 
            causes=data.genes, effects=data.genes, units=self.unitstring, name=self.name)
        return self


#############################
#   Observational Methods   #
#############################
class ObservationalMethod(Predictor):
    """Facilitates observational methods by subclassing this.
    
    Either override fit(data) or define a _fit_simple(x) instance method.
    
    Attributes:
        datatype ('obser', 'inter' or 'obser_inter_joint'): Description
        result (TYPE): Description
        standardized (Bool): Description
    """
    def __init__(self, datatype, standardized=False, *args, **kwargs):
        assert datatype in OBSERMETHOD_DATATYPES
        self.datatype = datatype # add this to options
        self.standardized = standardized
        super(ObservationalMethod, self).__init__(*args, **kwargs)
        self._options = {'datatype':self.datatype, 'standardized':self.standardized}

    @Predictor.fit_decorator
    def fit(self, data):
        assert issubclass(type(data), MicroArrayData)
        if self.standardized:
            x = getattr(data, self.datatype + '_std')
        else:
            x = getattr(data, self.datatype)
        # need to add units from return value if applicable
        simple_result = self._fit_simple(x)
        if type(simple_result) is tuple:
            array, unitstring = simple_result
        else:
            array = simple_result
            unitstring = '0' # units are unknown by default
        # build result
        self.result = CausalArray(array=array, causes=data.genes, 
            effects=data.genes, name=self.name, units=unitstring) # default units are unknown
        return self

    @property
    def name(self):
        """Unique name for a given method based on all callable options.
        
        If self._name instance is defined, this is used instead. Overriden from Predictor.
        """
        if self._name:
            return self._name
        else:
            name = self.__class__.__name__
            # make sure the datatype is printed first
            options_keys = self.options.keys()
            options_keys.remove('datatype')
            name = name + '_' + self.options['datatype']
            # if options are remaining, print the reset
            if options_keys:
                name = name + '_' + '_'.join(['{}={}'.format(i,j) for i,j in self.options.iteritems() if i is not 'datatype'])
            return name


class PearsonCor(ObservationalMethod):
    unitstring = '1'
    """Pearson's product-moment coefficient"""

    def _fit_simple(self, x):
        return np.abs(np.corrcoef(x)), self.unitstring # add units


class SpearmanCor(ObservationalMethod):
    """Spearman's rank correlation coefficient"""
    def _fit_simple(self, x):
        return np.abs(spearmanr(x, axis=1)[0])


class KendallCor(ObservationalMethod):
    """Kendall's rank correlation coefficient"""
    def _fit_simple(self, x):
        dim = x.shape[0]
        result = np.zeros((dim, dim), dtype=float)
        # upper triangular, as it is symmetric
        for i in xrange(dim):
            for j in xrange(i, dim):
                result[i,j], _ = kendalltau(x[i], x[j])
        # symmetrize result
        return np.abs(result + result.T - np.diag(result.diagonal()))


class DistanceCor(ObservationalMethod):
    """Distance correlation"""
    def _fit_simple(self, x):
        dim = x.shape[0]
        result = np.zeros((dim, dim), dtype=float)
        # upper triangular, as it is symmetric
        for i in xrange(dim):
            for j in xrange(i, dim):
                result[i,j] = distancecorr(x[i,:], x[j,:]) # non-negative
        return result + result.T - np.diag(result.diagonal()) 


class VarianceCauseInv(ObservationalMethod):
    """Inverse of the noisy sample standard-deviation of the cause as causal effect estimator.

    Note that we add the noise directly to the estimate of te standard-deviation before inverting the result.
    
    Attributes:
        noise (float, optional): ammount of standard normal noise added
        unitstring (str): Description
    """
    unitstring = '1/i'

    def __init__(self, noise=None, *args, **kwargs):
        super(VarianceCauseInv, self).__init__(*args, **kwargs)
        self.noise = noise
        if self.noise: self._options = {'noise': True}

    def _fit_simple(self, x):
        dim = x.shape[0] # ngenes
        # np.array( dim * list) stacks lists row-wise, i.e. [[row], [row], [row]] 
        # in the general case, the first dimension is added.
        array = np.reciprocal(np.array(dim * [x.std(axis=1),]).T)
        if self.noise is not None:
            array = array + np.reciprocal(self.noise * np.random.randn(*array.shape))
        return array, self.unitstring


class VarianceEffect(ObservationalMethod):
    """Sample standard-deviation of the effects"""
    unitstring = 'j' 

    def __init__(self, noise=None, *args, **kwargs):
        super(VarianceEffect, self).__init__(*args, **kwargs)
        self.noise = noise
        if self.noise: self._options = {'noise': True}

    def _fit_simple(self, x):
        dim = x.shape[0] # ngenes
        array = np.array(dim * [x.std(axis=1),])
        if self.noise is not None:
            array = array + self.noise * np.random.randn(*array.shape)
        return array, self.unitstring


class VarianceCombined(ObservationalMethod):
    """Sample standard-deviation of the cause and effects combined"""
    unitstring = 'j/i'

    def __init__(self, noise=None, *args, **kwargs):
        super(VarianceCombined, self).__init__(*args, **kwargs)
        self.noise = noise
        if self.noise: self._options = {'noise': True}

    def _fit_simple(self, x):
        dim = x.shape[0] # ngenes
        array = np.array(dim * [np.reciprocal(x.std(axis=1)),]).T * np.array(dim * [x.std(axis=1),])
        if self.noise is not None:
            array = array + (self.noise * np.random.randn(*array.shape)) * np.reciprocal((self.noise * np.random.randn(*array.shape)))
        return array, self.unitstring


class NonnormalCause(ObservationalMethod):
    """Score i --> j with -log p-value of KS test with a normal distribution"""
    def __init__(self, *args, **kwargs):
        super(NonnormalCause, self).__init__(standardized=True, *args, **kwargs)

    def _fit_simple(self, x):
        dim = x.shape[0]
        return np.array(dim * [compute_nonnormality(data=x, scale=False, alpha=None),]).T


class NonnormalEffect(ObservationalMethod):

    def __init__(self, *args, **kwargs):
        super(NonnormalEffect, self).__init__(standardized=True, *args, **kwargs)

    def _fit_simple(self, x):
        dim = x.shape[0]
        return np.array(dim * [compute_nonnormality(data=x, scale=False, alpha=None),])


class ANM(ObservationalMethod):
    pass


class NCC(ObservationalMethod, MultiProcess):
    def __init__(self, select_causes=None, graph_rdata_name=None, *args, **kwargs):
        super(NCC, self).__init__(*args, **kwargs)

###
### R methods
###
class CallRMethod(Predictor):
    def run_rscript(self, x, script, pars, keep_temp_files=False):
        """Call r script in subprocess with commandline arguments. 
        
        IMPORTANT:
            Python (hdf5) and R (rhdf5) use transposed conventions for storing 2D arrays 
            to disk. The input / output buffer files will therefor need transposing.

            We addopt the following convention here:
                Input data:
                    Python observational data is saved p x N
                    R observational data is used as N x p
                    --> No transposing needed
                Output data:
                    R scripts return (as saved hdf5 file) causes x effects shape.
                    --> NEED TO TRANSPOSE THE RESULT LOADED FROM A HDF5-FILE
                        IN PYTHON WHEN LOADING!
            This convention is hard-coded in the save_array and load_array functions if 
            calling with load_from_r=True option.

        Args:
            x (TYPE): Description
            script (TYPE): Description
            pars (TYPE): Description
            keep_temp_files (bool, optional): Description

        Returns:
            CausalArray wrapper of return script.
        """
        # setup files. Require that self.name is unique among all methods!
        input_buffer_file = os.path.abspath('{}/__input__{}.hdf5'.format(self.folder, self.name))
        output_buffer_file = os.path.abspath('{}/__output__{}.hdf5'.format(self.folder, self.name))
        if os.path.exists(input_buffer_file): os.remove(input_buffer_file)
        if os.path.exists(output_buffer_file): os.remove(output_buffer_file)
        print input_buffer_file
        print output_buffer_file
        print self.folder
        save_array(filename=input_buffer_file, data=x)
        input_parameters = ['input=%s' % input_buffer_file, 'input.dataset=data', 
        'output=%s' % output_buffer_file]

        # use concurrency if applicable
        if (self.processes > 1):
            input_parameters = input_parameters + ['processes=%s' % self.processes] 

        # start script
        start_time = time.time()
        # fancy_time = time.strftime('%Y-%m-%d %H:%M')
        fid_stderr = tempfile.TemporaryFile(mode='w+') 
        with open('{}/__stdout__{}__.txt'.format(self.folder, self.name), 'w+') as fid_stdout:
            try:
                print ' '.join(['Rscript', config._folder_extern_R + '/' + script] + input_parameters + pars)
                check_call(['Rscript', config._folder_extern_R + '/' + script] + input_parameters + pars, stdout=fid_stdout, stderr=fid_stderr)
                fid_stdout.seek(0,0)
                fid_stdout.write('***\n*** DONE in {:.2f}s\n***\n\n'.format(time.time() - start_time))
            except CalledProcessError as cpe:
                print 'CalledProcessError: {}.'.format(cpe)
                fid_stderr.seek(0)
                print '********\nSTDERR {}\n********\n{}********'.format(self.name, fid_stderr.read())
                with open('{}/__stderr__{}__.txt'.format(self.folder, self.name), 'w+') as fid_stderr_write:
                    fid_stderr.seek(0)
                    fid_stderr_write.write('STDERR for {}\n\n'.format(script))
                    fid_stderr_write.write(fid_stderr.read())
            finally:
                print 'file exists?', os.path.exists(output_buffer_file)
                result = load_array(output_buffer_file, load_from_r=True)
                if keep_temp_files: 
                    os.rename(input_buffer_file, '{}/input_{}.hdf5'.format(self.folder, self.name))
                    os.rename(output_buffer_file, '{}/output_{}.hdf5'.format(self.folder, self.name))
                else: # perhaps already removed previously!
                    if os.path.exists(input_buffer_file): os.remove(input_buffer_file)
                    if os.path.exists(output_buffer_file): os.remove(output_buffer_file)
                return result


class DummyR(ObservationalMethod, CallRMethod):
    def _fit_simple(self, x):
        script = 'ComputeDummy.R'
        pars = ['verbose=TRUE']
        if self.verbose: print 'input shape:', x.shape
        result = self.run_rscript(x=x, script=script, pars=pars)
        if self.verbose:
            print 'output shape', result.shape
            print '--> ALL ENTRIES ARE ' + ('' if np.all(np.equal(x, result)) else 'NOT ') + 'EQUAL.'
        return np.random.random((x.shape[0],x.shape[0]))
 

class Glasso(ObservationalMethod, CallRMethod):
    """ Compute GLASSO using standardized data, standardized implies dimensionless quantity."""
    unitstring = '1'

    def __init__(self, mbapprox, *args, **kwargs):
        super(Glasso, self).__init__(standardized=True, *args, **kwargs)
        self.mbapprox = mbapprox
        if self.mbapprox: self._options = {'mbapprox': True}
                    
    def _fit_simple(self, x):
        script = 'ComputeGlasso.R'
        pars = ['method={}'.format('mb' if self.mbapprox else 'glasso')]
        return np.abs(self.run_rscript(x=x, script=script, pars=pars)), self.unitstring


### old ridge class..
# class Ridge(ObservationalMethod):

    # def __init__(self, fix_lambda, *args, **kwargs):
    #     super(Ridge, self).__init__(*args, **kwargs)
    #     self.fix_lambda = fix_lambda
                    
    # def _fit_simple(self, x):
    #     script = 'ComputeRidgeCV.R'
    #     pars = ['fix.lambda.after={}'.format(self.fix_lambda)]
    #     return np.abs(self.run_rscript(x=x, script=script, pars=pars))


class ElasticNet(ObservationalMethod, MultiProcess, CallRMethod):
    """Elastic Net method (Zou, 2005)

    Note on select_causes:
        The problem with running the constrained regression on a subset of selected causes,
        and that as such the set of regressors X_i is restricted that it will influence which beta_ij are 
        found as predictive for Y_j
    """
    unitstring = '1' # as the data is standardized by default

    def __init__(self, alpha, nfolds=None, select_causes=None,  *args, **kwargs):
        super(ElasticNet, self).__init__(standardized=True, *args, **kwargs)
        self.alpha = alpha
        self.nfolds = nfolds or 10 # default of 10 folds for CV
        self.select_causes = select_causes
        # override self.options
        self._options = {'alpha':self.alpha, 'select_causes':True if self.select_causes is not None else None}

    @Predictor.fit_decorator
    def fit(self, data):
        assert issubclass(type(data), MicroArrayData)
        x = getattr(data, self.datatype + '_std')
        script = 'ComputeGLMNet.R'
        if self.select_causes is not None: # need the not None here, otherwise ambiguous truth-value!
            select_causes = ', '.join([str(i) for i in data.intpos_R(self.select_causes)]) # Note: python zero-based arrays, while R one-based arrays!
        else:
            select_causes = 'NULL'
        pars = ['alpha=%s' % self.alpha, 'nfolds=%s' % self.nfolds, 'selectCauses=%s' % select_causes]
        self.result = CausalArray(array=np.abs(self.run_rscript(x=x, script=script, pars=pars)),
            causes=self.select_causes if self.select_causes is not None else data.genes, effects=data.genes, units=self.unitstring, name=self.name)
        return self


class Ridge(ElasticNet):
    """Run elastic net with alpha = 0 """
    def __init__(self, *args, **kwargs):
        super(Ridge, self).__init__(alpha=0, *args, **kwargs)
        self._options = {'alpha':None} # remove alpha option!


class Lasso(ElasticNet):
    """Run elastic net with alpha = 1 """
    def __init__(self, *args, **kwargs):
        super(Lasso, self).__init__(alpha=1, *args, **kwargs)
        self._options = {'alpha':None} # remove alpha option!


class IDA(ObservationalMethod, MultiProcess, CallRMethod):
    """ 
    Note:
        As we give a scaled variant as the input to IDA, the output coefficient is a (physical) dimensionless quantity
        and directly corresonds with normalized groundtruths.
    """
    unitstring = '1'

    def __init__(self, method='pcstable', alpha=0.01, select_causes=None, graph_rdata_name=None, *args, **kwargs):
        super(IDA, self).__init__(standardized=True, *args, **kwargs)
        self.method = method
        assert method in ('pc', 'pcstable', 'pcstablefast', 'empty')
        self.alpha = alpha
        self.select_causes = select_causes
        self.graph_rdata_name = graph_rdata_name or 'ida_' + self.method + '_graph_alpha' + str(self.alpha) + '_' + self.datatype + '.RData'
        # override self.options
        self._options = {'method':self.method, 'alpha':self.alpha, 'select_causes':True if self.select_causes is not None else None}

    @Predictor.fit_decorator
    def fit(self, data):
        assert issubclass(type(data), MicroArrayData)
        x = getattr(data, self.datatype + '_std')
        script = 'ComputeIDA.R'
        if self.select_causes is not None: # need the not None here, otherwise ambiguous truth-value!
            select_causes = ', '.join([str(i) for i in data.intpos_R(self.select_causes)]) # Note: python zero-based arrays, while R one-based arrays!
        else:
            select_causes = 'NULL'
        pars = ['method=%s' % self.method, 'alpha=%s' % self.alpha, 'pcgraphfile=%s' % self.folder + '/' + self.graph_rdata_name,
        'selectCauses=%s' % select_causes, 'processes=%s' % self.processes]
        self.result = CausalArray(array=self.run_rscript(x=x, script=script, pars=pars),
            causes=self.select_causes if self.select_causes is not None else data.genes, effects=data.genes, 
            units=self.unitstring, name=self.name)
        return self


class ICP(MultiProcess):
    """Invariant Causal Prediction with stability sampled ranked predictions.
    
    Attributes:
        alpha (float): Confidence level for the statistical test
        bootstraps (int): Number of bootstraps to perform for stability sampling
        bootstrap_fraction (double): fraction of the interventional data to be subsampled for stability
        result (CausalArray): Resulting causal array if done, else None
    """
    def __init__(self, alpha=0.05, bootstraps=1, bootstrap_fraction=None, *args, **kwargs):
        super(ICP, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.bootstraps = bootstraps
        self.bootstrap_fraction = bootstrap_fraction or .5
        # self.select_causes = select_causes
        # if self.prescreening: print '{} WARNING: make sure prescreening CausalArray file {} exists!'.format(self.name_method, self.prescreening)
        # self._options = {'select_causes':select_causes is not None}
        self._options = {'alpha':self.alpha, 'bootstraps':self.bootstraps, 'bootstrap_fraction':self.bootstrap_fraction}

    @Predictor.fit_decorator
    def fit(self, data):
        """Fit function, modified from the "rscript" functions in the ObservationalMethod base class to 
            use the full dataset including interventions as input. 
        
        Note: require the full dataset to be completely saved instead of just a "settings_only version", as 
            the R scripts cannot read those out. 
        """
        keep_temp_files = False # do not keep temp io files
        
        # Set names etc
        input_buffer_file = '{}/__input__{}.hdf5'.format(self.folder, self.name)
        output_buffer_file = '{}/__output__{}.hdf5'.format(self.folder, self.name)
        if os.path.exists(input_buffer_file): os.remove(input_buffer_file)
        if os.path.exists(output_buffer_file): os.remove(output_buffer_file)

        # make sure to save the complete file and not only settings!
        data.save(file=input_buffer_file, settings_only=False, verbose=False)

        # if self.select_causes is not None: # need the not None here, otherwise ambiguous truth-value!
        #     select_causes = ', '.join([str(i) for i in data.intpos_R(self.select_causes)]) # Note: python zero-based arrays, while R one-based arrays!
        # else:
        #     select_causes = 'NULL'

        # Setup script and input parameters
        script = 'ComputeICP.R'
        input_parameters = [
            'input=%s' % input_buffer_file,
            'output=%s' % output_buffer_file,
            'alpha%s' % self.alpha,
            'bootstraps=%s' % self.bootstraps,
            'bootstrapFraction=%s' % self.bootstrap_fraction,
            # 'selectCauses=%s' % select_causes
        ]

        # use concurrency if applicable
        if (self.processes > 1):
            input_parameters += ['processes=%s' % self.processes] 

        # start script
        start_time = time.time()
        # fancy_time = time.strftime('%Y-%m-%d %H:%M')
        fid_stderr = tempfile.TemporaryFile(mode='w+') 
        with open('{}/__stdout__{}__.txt'.format(self.folder, self.name), 'w+') as fid_stdout:
            try:
                check_call(['Rscript', config._folder_extern_R + '/' + script] + input_parameters, stdout=fid_stdout, stderr=fid_stderr)
                fid_stdout.seek(0,0)
                fid_stdout.write('***\n*** DONE in {:.2f}s\n***\n\n'.format(time.time() - start_time))
            except CalledProcessError as cpe:
                print 'CalledProcessError: {}.'.format(cpe)
                fid_stderr.seek(0)
                print '********\nSTDERR {}\n********\n{}********'.format(self.name, fid_stderr.read())
                with open('{}/__stderr__{}__.txt'.format(self.folder, self.name), 'w+') as fid_stderr_write:
                    fid_stderr.seek(0)
                    fid_stderr_write.write('STDERR for {}\n\n'.format(script))
                    fid_stderr_write.write(fid_stderr.read())
            finally:
                result = load_array(output_buffer_file, load_from_r=True)
                if keep_temp_files: 
                    os.rename(input_buffer_file, '{}/input_{}.hdf5'.format(self.folder, self.name))
                    os.rename(output_buffer_file, '{}/output_{}.hdf5'.format(self.folder, self.name))
                else:
                    if os.path.exists(input_buffer_file): os.remove(input_buffer_file)
                    if os.path.exists(output_buffer_file): os.remove(output_buffer_file)
                self.result = CausalArray(array=result.astype('int'), causes=data.genes, effects=data.genes, name=self.name)
                return self


###
### Combined methods
###
class LCD(Predictor):
    def __init__(self, numeric_score=False, alpha=None, beta=None, gt_method=None, 
        gt_threshold=None, conservative=False, select_causes=None, reduce_memory=False, *args, **kwargs):
        super(LCD, self).__init__(*args, **kwargs)
        # self.numeric_score = numeric_score
        self.alpha = alpha
        self.beta = beta
        self.conservative = conservative
        self.gt_method = gt_method
        self.gt_threshold = gt_threshold
        self.select_causes = select_causes
        self.reduce_memory = reduce_memory
        # self._options = {'select_causes':select_causes is not None}
        self._options = {'alpha':self.alpha, 'beta':self.beta, 'conservative':self.conservative,
            'gt_method':self.gt_method, 'gt_threshold':self.gt_threshold, 'select_causes':True if self.select_causes is not None else None,
            'reduce_memory':self.reduce_memory}
    
    @Predictor.fit_decorator
    def fit(self, data):
        self.result = compute_lcd(data=data, alpha=self.alpha, beta=self.beta, gt_method=self.gt_method, 
            gt_threshold=self.gt_threshold, conservative=self.conservative, select_causes=self.select_causes,
            reduce_memory=self.reduce_memory, full_result_folder=None, verbose=self.verbose)
        return self


class LCDScore(LCD, MultiProcess):
    """LCD with stability sampled ranked predictions.
    
    By default, sample 1/2 of the number of datapoints in both regimes for the bootstrap.
    
    Attributes:
        bootstraps (int): Number of bootstraps to perform for stability sampling
        bootstraps_nobs (int): Number of observational datapoints to be sampled in bootstrap
        bootstraps_nobs (int): Number of intervational datapoints to be sampled in bootstrap
        result (CausalArray): Resulting causal array if done, else None
    """
    def __init__(self, bootstraps=10, bootstrap_obs=None, bootstrap_ints=None, alpha=None, beta=None, gt_method=None, 
        gt_threshold=None, conservative=False, select_causes=None, reduce_memory=False, **kwargs):
        # if 'numeric_score' in kwargs: del kwargs['numeric_score']
        self.bootstraps = bootstraps
        self.bootstrap_obs = bootstrap_obs
        self.bootstrap_ints = bootstrap_ints
        super(LCDScore, self).__init__(alpha=alpha, beta=beta, gt_method=gt_method, 
        gt_threshold=gt_threshold, conservative=conservative, select_causes=select_causes, 
        reduce_memory=reduce_memory, **kwargs)
        self._options = {'bootstraps':self.bootstraps, 'bootstrap_obs':self.bootstrap_obs,'bootstrap_ints':self.bootstrap_ints}

    @Predictor.fit_decorator
    def fit(self, data):
        self.result = compute_bagged_lcd(data=data, bootstraps=self.bootstraps, processes=self.processes, 
            alpha=self.alpha, beta=self.beta, gt_method=self.gt_method, gt_threshold=self.gt_threshold, 
            conservative=self.conservative, select_causes=self.select_causes, reduce_memory=self.reduce_memory,
            full_result_folder=None, verbose=self.verbose)
        return self


class LCDScoreSlow(LCD, MultiProcess):
    """LCD with stability sampled ranked predictions.
    
    By default, sample 1/2 of the number of datapoints in both regimes for the bootstrap.
    
    Attributes:
        bootstraps (int): Number of bootstraps to perform for stability sampling
        bootstraps_nobs (int): Number of observational datapoints to be sampled in bootstrap
        bootstraps_nobs (int): Number of intervational datapoints to be sampled in bootstrap
        result (CausalArray): Resulting causal array if done, else None
    """
    def __init__(self, bootstraps=10, bootstrap_obs=None, bootstrap_ints=None, alpha=None, beta=None, gt_method=None, 
        gt_threshold=None, conservative=False, select_causes=None, reduce_memory=False, **kwargs):
        # if 'numeric_score' in kwargs: del kwargs['numeric_score']
        self.bootstraps = bootstraps
        self.bootstrap_obs = bootstrap_obs
        self.bootstrap_ints = bootstrap_ints
        super(LCDScoreSlow, self).__init__(alpha=alpha, beta=beta, gt_method=gt_method, 
        gt_threshold=gt_threshold, conservative=conservative, select_causes=select_causes, 
        reduce_memory=reduce_memory, **kwargs)
        self._options = {'bootstraps':self.bootstraps, 'bootstrap_obs':self.bootstrap_obs,'bootstrap_ints':self.bootstrap_ints}

    @Predictor.fit_decorator
    def fit(self, data):
        self.result = compute_bagged_lcd_slow(data=data, bootstraps=self.bootstraps, processes=self.processes, 
            alpha=self.alpha, beta=self.beta, gt_method=self.gt_method, gt_threshold=self.gt_threshold, 
            conservative=self.conservative, select_causes=self.select_causes, reduce_memory=self.reduce_memory,
            full_result_folder=None, verbose=self.verbose)
        return self


class GIES(MultiProcess):
    """Greedy interventional equivalent search with stability sampled ranked predictions.
    
    Score-based algorithm to predict a CPDAG (or essential graph) that modifies greedy
    search used in GES with interventional data.
    
    Attributes:
        bootstraps (int): Number of bootstraps to perform for stability sampling
        bootstrap_fraction (double): fraction of the interventional data to be subsampled for stability
        result (CausalArray): Resulting causal array if done, else None
    """
    def __init__(self, max_degree=0, bootstraps=1, bootstrap_fraction=None, 
        select_causes=None, prescreening=None, *args, **kwargs):
        super(GIES, self).__init__(*args, **kwargs)
        self.max_degree = max_degree
        self.bootstraps = bootstraps
        self.bootstrap_fraction = bootstrap_fraction or .5
        self.select_causes = select_causes
        self.prescreening = prescreening
        if self.prescreening: print '{} WARNING: make sure prescreening CausalArray file {} exists!'.format(self.name_method, self.prescreening)


    @Predictor.fit_decorator
    def fit(self, data):
        """Fit function, modified from the "rscript" functions in the ObservationalMethod base class to 
        use the full dataset including interventions as input. 
        
        Note: require the full dataset to be completely saved instead of just a "settings_only version", as 
        the R scripts cannot read those out.    
        """
        keep_temp_files = False # do not keep temp io files

        # Set names etc
        input_buffer_file = '{}/__input__{}.hdf5'.format(self.folder, self.name)
        output_buffer_file = '{}/__output__{}.hdf5'.format(self.folder, self.name)
        if os.path.exists(input_buffer_file): os.remove(input_buffer_file)
        if os.path.exists(output_buffer_file): os.remove(output_buffer_file)

        # make sure to save the complete file and not only settings!
        data.save(file=input_buffer_file, settings_only=False, verbose=False)

        if self.select_causes is not None: # need the not None here, otherwise ambiguous truth-value!
            select_causes = ', '.join([str(i) for i in data.intpos_R(self.select_causes)]) # Note: python zero-based arrays, while R one-based arrays!
        else:
            select_causes = 'NULL'

        # Setup script and input parameters
        script = 'ComputeGIES.R'
        input_parameters = [
            'input=%s' % input_buffer_file,
            'output=%s' % output_buffer_file,
            # 'maxDegree%s' % self.max_degree,
            'bootstraps=%s' % self.bootstraps,
            'bootstrapFraction=%s' % self.bootstrap_fraction,
            'selectCauses=%s' % select_causes
        ]

        # use prescreening file if given
        if self.prescreening:
            if os.path.exists(self.prescreening):
                input_parameters += ['prescreeningFile=%s' % self.prescreening]
            else:
                print '[{}] WARNING: prescreening file {} does not exists!'.format(self.name_method, self.prescreening)
                return self

        # use concurrency if applicable
        if (self.processes > 1):
            input_parameters += ['processes=%s' % self.processes] 

        # start script
        start_time = time.time()
        # fancy_time = time.strftime('%Y-%m-%d %H:%M')
        fid_stderr = tempfile.TemporaryFile(mode='w+') 
        with open('{}/__stdout__{}__.txt'.format(self.folder, self.name), 'w+') as fid_stdout:
            try:
                check_call(['Rscript', script] + input_parameters, stdout=fid_stdout, stderr=fid_stderr)
                fid_stdout.seek(0,0)
                fid_stdout.write('***\n*** DONE in {:.2f}s\n***\n\n'.format(time.time() - start_time))
            except CalledProcessError as cpe:
                print 'CalledProcessError: {}.'.format(cpe)
                fid_stderr.seek(0)
                print '********\nSTDERR {}\n********\n{}********'.format(self.name, fid_stderr.read())
                with open('{}/__stderr__{}__.txt'.format(self.folder, self.name), 'w+') as fid_stderr_write:
                    fid_stderr.seek(0)
                    fid_stderr_write.write('STDERR for {}\n\n'.format(script))
                    fid_stderr_write.write(fid_stderr.read())
            finally:
                result = load_array(output_buffer_file, load_from_r=True)
                if keep_temp_files: 
                    os.rename(input_buffer_file, '{}/input_{}.hdf5'.format(self.folder, self.name))
                    os.rename(output_buffer_file, '{}/output_{}.hdf5'.format(self.folder, self.name))
                else:
                    if os.path.exists(input_buffer_file): os.remove(input_buffer_file)
                    if os.path.exists(output_buffer_file): os.remove(output_buffer_file)
                self.result = CausalArray(array=result.astype('int'), causes=self.select_causes if self.select_causes is not None else data.genes, 
                    effects=data.genes, name=self.name)
                return self

# directly get the list of methods available from globals
# METHODS_DICT = dict((k,v) for k,v in (globals().copy()).iteritems() if issubclass(type(v), type(Predictor)) and k[0] is not '_')
# METHODS_AVAILABLE = METHODS_DICT.keys()
