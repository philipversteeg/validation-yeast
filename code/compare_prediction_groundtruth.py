"""Class for efficiently comparing multiple prediction sets against multiple 
groundtruth sets and computing summary statistics and plots.

Instanciate ComparePredictionGroundtruth and use its methods.

Author: Philip Versteeg (pjjpversteeg@gmail.com)
"""
# stdlib
import os
from copy import deepcopy
import itertools
from functools import wraps, partial
import time
from multiprocessing import Pool, Queue
import h5py

# ml / stats
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

# plot
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import ticker

# local
import config
from microarraydata import Kemmeren # for testing only
from libs.causallib import CausalArray, ZeroCausalArray,  convert_to_binary, UnitsOfMeasure, intersect_causalarrays
from libs.misc import risk_estimate, precision, recall, argsort_randomize_ties, under_scores_to_camel_case, memory_usage_resource
from libs.ranking_pauc import roc_auc_score # includes the patch for the partial AUC.

COMPARE_SETTINGS = ['intersection', 'pairwise']
SERVER = os.sys.platform != 'darwin' # true if on linux or windows
SEED = 42
np.random.seed = SEED

### BETTER DEFAULT PLOT SETTINGS 
# MatPlotLib 2.0 colors, except grey which is reserved for random.
MARKER_CYCLE = ['x', 'o', '.', ',', 's', '^', 'v', 'P', '*']
MARKER_NUMBER = 20 # number of markers per plot
COLOR_CYCLE = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728',
              '#9467bd', u'#8c564b', u'#e377c2', #u'#7f7f7f',
              '#bcbd22', u'#17becf']
LINE_CYCLE_RANDOM = ['dotted', 'dashed', 'dashdot', 'solid'] # for multiple random plots
FIGSIZE=(5.5, 5.5)
# DPI = 200
DPI = 300

# import matplotlib 2.1 default parameters to older version
if int(matplotlib.__version__.split('.')[0]) < 1: 
    for k, v in config.plt_rcParams.iteritems():
        plt.rcParams[k] = v

plt.rcParams['font.size'] = 9.
plt.rcParams['legend.fontsize'] = 7.
plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markeredgewidth'] = .5
plt.rcParams['lines.markersize'] = 3.

###
### Helper functions
###
def compute_auc_bootstrap(p, g, nbootstraps, max_fpr=None, nprocesses=None):
    """ Compute AUC bootstrapped."""
    nprocesses = nprocesses or 36 if SERVER else 2
    # array_bootstraps = Queue() # processafe
    pool = Pool(nprocesses)
    process_list = [pool.apply_async(func=compute_one_bootstrap, kwds={'p':p, 'g':g, 'max_fpr':max_fpr}) for i in range(nbootstraps)]
    pool.close()
    pool.join()
    array_bootstraps = np.array([i.get() for i in process_list])
    return array_bootstraps

def compute_one_bootstrap(p, g, max_fpr):
    _start_bootstrap = time.time()
    score = roc_auc_score(
        np.random.choice(g, size=g.size, replace=True),
        np.random.choice(p, size=p.size, replace=True),
        max_fpr=max_fpr)
    # print 'Done in {:.2f}s.'.format(time.time() - _start_bootstrap) 
    return score


class ComparePredictionGroundtruth(object):
    """Class used to compare multiple predictions against multiple groundtruths.
    
    Workflow:
        0.  Create instance with desired copmarison properties.
        1.  Add predictions / groundtruths, which are saved in self._predictions and self._groundtruth dicts 
            that are NOT mutated throughout computation.  
        2.  These are acessed only through the self.predictions and self.groundtruth read-only dicts through the identifiers (keys) that are created by generators. 
        3.  Compare shapes, compute masks and save. Change this to only compute the intersection depending on what you need.)
        4.  Provide the desired output statistics:
            a.  ROC curves (pairwise vs gt-wise) 
            b.  AUC and AUC boxplots
            c.  Confusion matrix, need both binary.
    
    Implementation notes:
        Uses dict of dicts for storing mask for each {prediction, groundtruth} pair. 
        The following naming conventions are used:
        * p, pred are [unique identifier], [CausalArray] pairs
        * ps, preds are iterables of these pairs
        * similar convention holds for g, gt and gs, gts
        Thresholds are not computed a priori but applied on the fly through generator functions
        to save memory and computations. This also allows for later modification of the parameter
        without computing (part of) the mask dictionaries.
    
    Attributes:
        default_max_fpr_list (list): default list of maximum FPR to plot
        default_max_tpr_list (list): default list of maximum TPR to plot
        default_thresholds (list): default list of thresholds
        force_units_unknown (bool): option to force all 'unknown' units from predictions and threat them as a) the specified UnitsOfMeasure or unitstring, i.e. '1' or 'j' or b) 'match' to the compared unit
        linestyle (str): linestyle of plots
        number_random_groundtruths (int): number of randomly sampled groundtruths added
        number_random_predictions (int): number of randomly sampled predictions added
        remove_self_loops (bool): remove self loops from the comparison, i.e. A -> A excluded
        scale_units (dict): if not none, should be a dict containing scale factors for all causes and effects
        setting (str): comparison settings: 'pairwise' (default) takes the subset of causes and effects in each (prediction, groundtruth) as the comparison
        settings_available (list): available comparison settings
        set; 'intersection' takes the intersection of all. The latter is faster and less memory. 
        expensive.
        symmetrize (bool): Take intersection of causes and effects
        thresholds (list): list of top-percentiles that are taken as positives for computing binary groundtruths
        verbose (bool): verbose flag
    """
    # default_thresholds = [5., 1.]
    default_thresholds = [90., 95.] # CHANGED THRESHOLD TO PERCENTILE
    default_max_fpr_list = [0.05, 0.01, 0.002]
    default_max_tpr_list = [0.5, 0.05, 0.005, 0.0005]

    settings_available = ['intersection', 'pairwise']

    # plot options
    linestyle_tpfp = '-' # default solid line

    def __init__(self, 
                 pred=None,                         # (iterable of) prediction(s), calls self.add_pred()
                 gt=None,                           # (iterable of) groundtruth(s), calls self.add_gt()
                 remove_self_loops=True,            # remove self loops from the comparison, i.e. A -> A excluded
                 symmetrize=False,                  # reduce the comparison shape to only common causes and effects
                 thresholds=None,                   # set of top-percentiles that are taken as positives for computing binary groundtruths
                 setting='pairwise',                # default = 'pairwise'
                 number_random_predictions=5,       # number of randomly sampled predictions added
                 number_random_groundtruths=5,      # number of randomly sampled groundtruths added
                 scale_units=None,                  # if not none, should be a dict containing scale factors for all causes and effects
                 force_units_unknown=None,          # extra option to force all 'unknown' units from predictions and threat them as a) the specified UnitsOfMeasure or unitstring, i.e. '1' or 'j' or b) 'match' to the compared unit.
                 verbose=True                       # verbose flag
                 ):
        super(ComparePredictionGroundtruth, self).__init__()
        self._predictions = {}
        self.add_pred(pred)
        self._groundtruths = {}
        self.add_gt(gt)
        self.remove_self_loops = remove_self_loops
        self.symmetrize = symmetrize
        self.setting = setting
        assert self.setting in self.settings_available
        if self.setting == 'intersection': raise Exception('Not implemented for now!')
        self.thresholds = [float(i) for i in thresholds] if thresholds else self.default_thresholds
        self.number_random_predictions = number_random_predictions
        self.number_random_groundtruths = number_random_groundtruths # usually just one...
        self.force_units_unknown = force_units_unknown
        self.scale_units = scale_units
        self.verbose = verbose
        self._changed_dicts = True # flag that keeps track of changed dicts; need to do set of computations again if this is True.
        self._noscale_these = []    # temp list of pairs of identifiers to force a not scale.
        # assertions and exceptions
        assert scale_units is None or type(scale_units) is dict
        if self.setting not in COMPARE_SETTINGS: 
            raise Exception('setting %s not implemented!' % setting)
        # print self

    def __repr__(self):
        return 'Predcitions:\n{}\n\nGroundtruths:\n{}'.format('\t'.join(self._predictions.keys()), '\t'.join(self._groundtruths.keys()))

    def add_gt(self, gt):
        """Add a prediction CausalArray or dictionary of CausalArrays."""
        if issubclass(type(gt), dict):
            for i in gt.values():
                assert issubclass(type(i), CausalArray)
            self._groundtruths.update(gt)
        elif issubclass(type(gt), CausalArray):
            self._groundtruths.update({gt.name: gt})
        elif hasattr(gt, '__iter__'): # iterable
            for i in gt: self.add_gt(i)
                # assert issubclass(type(i), CausalArray)
                # self._groundtruths[i.name] = i
        self._changed_dicts = True

    def add_pred(self, pred):
        """Add a prediction CausalArray or dictionary of CausalArrays."""
        if issubclass(type(pred), dict):
            for i in pred.values():
                assert issubclass(type(i), CausalArray)
            self._predictions.update(pred)
        elif issubclass(type(pred), CausalArray):
            self._predictions.update({pred.name: pred})
        elif hasattr(pred, '__iter__'): # iterable
            for i in pred: self.add_pred(i)
                # assert issubclass(type(i), CausalArray)
                # self._predictions[i.name] = i
        self._changed_dicts = True

    def _compute_masks(self):
        """Internal method called to compute all masks 

        Note: This should be the only method that can set self._changed_dicts = False!

        Raises:
            ZeroCausalArray: Exception raised when the set of either predictions or groundtruths is empty.
        """
        # first check that the start i
        start_time = time.time()
        if len(self._predictions) == 0:
            raise ZeroCausalArray('No predictions added!')
        if len(self._groundtruths) == 0:
            raise ZeroCausalArray('No groundtruths added!')

        ## Process random ground truth and predictions.
        ## Intersection 
        self._intersection_causes, self._intersection_effects = intersect_causalarrays(*(self._groundtruths.values() + self._predictions.values()))
        self._intersection_shape = len(self._intersection_causes), len(self._intersection_effects)

        ## Union 
        self._union_causes = sorted(set.union(*[set(v.causes) for v in self._groundtruths.values() + self._predictions.values()]))
        self._union_effects = sorted(set.union(*[set(v.effects) for v in self._groundtruths.values() + self._predictions.values()]))
        self._union_shape = len(self._union_causes), len(self._union_effects)
 
        ## Create random predictions over the joint union of causes and effects
        self.random_predictions = dict(('random_%s' % i, CausalArray(array=np.random.random(size=self._union_shape).astype('float16'), # 32 or even 16 bit is sufficiently large
            causes=self._union_causes, effects=self._union_effects, name='random_%s' % i)) for i in xrange(self.number_random_predictions))
        self.random_groundtruths = dict(('random_%s' % i, CausalArray(array=np.random.random(size=self._union_shape).astype('float16'),  # 32 or even 16 bit is sufficiently large
            causes=self._union_causes, effects=self._union_effects, name='random_%s' % i)) for i in xrange(self.number_random_groundtruths))

        ## For each prediction-type, a dict points to each groundtruth type, where a dict points to the comparison mask.
        self._pred_gt_masks = {}

        ## Pairwise version
        if self.setting == 'pairwise':
            for p, pred in dict(self._predictions, **{'_random':self.random_predictions['random_0']}).iteritems() if self.number_random_predictions > 0 else self._predictions.iteritems():

                ## Force units of unknown prediction if required
                # if not pred.has_units and self.force_units_unknown is not None:
                #     pred.units = self.force_units_unknown

                self._pred_gt_masks[p] = {}
                for g, gt in dict(self._groundtruths, **{'_random':self.random_groundtruths['random_0']}).iteritems() if self.number_random_groundtruths > 0 else self._groundtruths.iteritems():

                    ## Force units of unknown groundtruth estimate if required
                    # if not gt.has_units and self.force_units_unknown is not None:
                    #     gt.units = self.force_units_unknown                    

                    # ## Skip the comparison for any pair when either one one has no units and if both are not random predictions.
                    if self.scale_units and self.force_units_unknown is None and (not gt.has_units or not pred.has_units) and not (p == '_random' or g == '_random'):
                        self._pred_gt_masks[p][g] = None
                        continue

                    ## Create masks
                    p_mask = np.zeros(pred.shape, dtype=bool)
                    g_mask = np.zeros(gt.shape, dtype=bool)
                    
                    ## Mark predictions missing
                    p_mask[pred.index_causes(set.difference(set(pred.causes), set(gt.causes))),:] = True
                    p_mask[:,pred.index_effects(set.difference(set(pred.effects), set(gt.effects)))] = True

                    ## Mark gts missing
                    g_mask[gt.index_causes(set.difference(set(gt.causes), set(pred.causes))),:] = True
                    g_mask[:,gt.index_effects(set.difference(set(gt.effects), set(pred.effects)))] = True

                    ## Remove self loops if applicable
                    if self.remove_self_loops:
                        common_ca_ef = set.intersection(set(pred.causes), set(pred.effects))
                        p_mask[[pred.causes.index(i) for i in common_ca_ef], [pred.effects.index(i) for i in common_ca_ef]] = True

                        common_ca_ef = set.intersection(set(gt.causes), set(gt.effects))
                        g_mask[[gt.causes.index(i) for i in common_ca_ef], [gt.effects.index(i) for i in common_ca_ef]] = True

                    if self.symmetrize:
                        if pred.ncauses > pred.neffects:
                            p_mask[pred.index_causes(set.difference(set(pred.causes), set(pred.effects))),:] = True
                        else:
                            p_mask[:,pred.index_effects(set.difference(set(pred.effects), set(pred.causes)))] = True

                        if gt.ncauses > gt.neffects:
                            g_mask[gt.index_causes(set.difference(set(gt.causes), set(gt.effects))),:] = True
                        else:
                            g_mask[:,gt.index_effects(set.difference(set(gt.effects), set(gt.causes)))] = True

                    # print p_mask
                    # print g_mask

                    ## Need to do some post-processing to check that there still exists overlap in pred and gt masks!
                    if np.all(p_mask) or np.all(g_mask):
                        self._pred_gt_masks[p][g] = None
                    else:
                        self._pred_gt_masks[p][g] = p_mask, g_mask


        ## Intersection need just one mask? No.
        if self.setting == 'intersection':
            raise Exception('Not implemented!')

        ## Done
        if self.verbose: print '[CompPredGt] Computed comparisons in {:.2f}s.'.format(time.time() - start_time)
        self._changed_dicts = False


    ###
    ###     Helper decorators
    ###
    def update_mask(fn):
        """ Decorator for methods that call objects that might need to be computed."""
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            if self._changed_dicts: self._compute_masks()
            return fn(self, *args, **kwargs)
        return wrapper

    ###
    ###     Properties that return compatible predictions and groundtruths.
    ###
    @property
    @update_mask
    def predictions(self):
        """Except the random"""
        return self._predictions

    @property
    @update_mask
    def groundtruths(self):
        return self._groundtruths

    @property
    def predictions_available(self):
        return sorted(self.predictions.keys())

    @property
    def groundtruths_available(self):
        return sorted(self.groundtruths.keys())

    @property
    def predictions_continuous_available(self):
        return [p for p in self.predictions_available if not self.predictions[p].binary]

    @property
    def predictions_binary_available(self):
        """All predictions can be made binary"""
        if len(self.thresholds) != 0:
            return self.predictions_available 
        else:
            return [p for p in self.predictions_available if self.predictions[p].binary]

    @property
    def groundtruths_continuous_available(self):
        return [g for g in self.groundtruths_available if not self.groundtruths[g].binary]

    @property
    def groundtruths_binary_available(self):
        """All groundtruths can be made binary"""
        if len(self.thresholds) != 0:
            return self.groundtruths_available 
        else:
            return [g for g in self.groundtruths_available if self.groundtruths[g].binary]


    ###
    ###     Generators to call (reducing memory pressure) when looping over predictions and groundtruths. 
    ###
    def predictions_binary(self, ps=None):
        """Generator: maps predictions identifier --> preds"""
        ps = ps or self.predictions_available
        for p in ps:
            pred = self.predictions[p]
            if pred.binary:
                yield p, pred
            else:
                for thr in self.thresholds:
                    yield p, pred.make_binary(percentile=float(thr), strict=True)

    def predictions_continuous(self, ps=None):
        ps = ps or self.predictions_available
        for p in ps:
            pred = self.predictions[p]
            if not pred.binary:
                yield p, pred

    def groundtruths_binary(self, gs=None):
        gs = gs or self.groundtruths_available
        for g in gs:
            gt = self.groundtruths[g]
            if gt.binary:
                yield g, gt
            else:
                for thr in self.thresholds:
                    yield g, gt.make_binary(percentile=float(thr), strict=True)

    def groundtruths_continuous(self, gs=None):
        gs = gs or self.groundtruths_available
        for g in gs:
            gt = self.groundtruths[g]
            if not gs.binary:
                yield g, gs

    def number_predictions_continuous(self, ps=None):
        """Number of continuous predictions in total or in 'ps'"""
        ps = ps or self.predictions_available
        return len([1 for p in ps if not self.predictions[p].binary])

    def number_predictions_binary(self, ps=None):
        ps = ps or self.predictions_available
        return len([1 for p in ps if self.predictions[p].binary]) + len(self.thresholds) * self.number_predictions_continuous(ps)

    def number_groundtruths_continuous(self, gs=None):
        """Number of continuous groundtruths in total or in 'gs'"""
        gs = gs or self.groundtruths_available
        return len([1 for g in gs if not self.groundtruths[g].binary])

    def number_groundtruths_binary(self, gs=None):
        gs = gs or self.groundtruths_available
        return len([1 for g in gs if self.groundtruths[g].binary]) + len(self.thresholds) * self.number_groundtruths_continuous(gs)

    ## intersection
    @property
    @update_mask
    def intersection_causes(self):
        return self._intersection_causes

    @property
    @update_mask
    def intersection_effects(self):
        return self._intersection_effects

    @property
    @update_mask
    def intersection_shape(self):
        return self._intersection_shape

    ## union
    @property
    @update_mask
    def union_causes(self):
        return self._union_causes

    @property
    @update_mask
    def union_effects(self):
        return self._union_effects

    @property
    @update_mask
    def union_shape(self):
        return self._union_shape

    ###
    ###     Comparing functions
    ###
    @update_mask
    def comparison_exists(self, p, g):
        """Check if the result is not empty"""
        return self._pred_gt_masks[p][g] is not None

    @update_mask
    def compare_pred_gt(self, p, g, pred=None, gt=None, flatten=True):
        """Return 1d arrays of the matched prediction and groundtruth and normalize / marginalize if necessary."""
        assert self.comparison_exists(p, g) # should have been checked
        # get the 
        if gt is None: gt = self.groundtruths[g]
        if pred is None: pred = self.predictions[p]
        # check if the pred or gt is random 
        if p.startswith('random_'): p = '_random' 
        if g.startswith('random_'): g = '_random'
        p_mask, g_mask = self._pred_gt_masks[p][g]
        if self.scale_units:
            # make an exception for the random predicitons and groundtruths, do not apply normalization if either one (or both) is random 
            if p == '_random' or g == '_random':
                p_array = pred.array
            elif (p, g) in self._noscale_these:
                p_array = pred.array
            else:
                # case 1. both have units
                if pred.has_units and gt.has_units:
                    p_array = pred.normalize_units_array(compare_with=gt, scale_units=self.scale_units)
                # depending on the setting "force units unknown", we need to match these when normalizing
                else:
                    # if we match the units, just skip thistermin
                    if self.force_units_unknown == 'match':
                        p_array = pred.array
                    elif type(self.force_units_unknown) is str:
                        # case 2. an exception for the case that both p and gt have no units, then we apply no scaling
                        if not pred.has_units and gt.has_units:
                            p_array = pred.array
                        # case 3. pred has no units, but gt has
                        elif not pred.has_units:
                            _tmp_units = pred.units
                            pred.units = self.force_units_unknown
                            p_array = pred.normalize_units_array(compare_with=gt, scale_units=self.scale_units)
                            pred.units = _tmp_units
                        # case 4. gt has no units but pred has
                        elif not gt.has_units:
                            _tmp_units = gt.units
                            gt.units = self.force_units_unknown
                            p_array = pred.normalize_units_array(compare_with=gt, scale_units=self.scale_units)
                            gt.units = _tmp_units
                    else:
                        raise Exception('Unknown parameter setting, force units is not None and not a unitstring!')
        else:
            p_array = pred.array
        g_array = gt.array
        # construct masked versions
        p_ma, g_ma = np.ma.MaskedArray(p_array, mask=p_mask), np.ma.MaskedArray(g_array, mask=g_mask)
        
        if flatten:
            return p_ma.compressed(), g_ma.compressed()
        else:
            return p_ma, g_ma

    @update_mask
    def compute_fp_tp(self, p, g, pred, gt, *args, **kwargs):
        """Compute (fp, tp) for two CausalArrays."""
        fpr, tpr, _ = roc_curve(*self.compare_pred_gt(p, g, pred, gt, *args, **kwargs)[::-1], drop_intermediate=False) # much faster with drop_intermediate=True?
        con_pos = gt.num_nonzero
        con_neg = gt.num_zero
        return fpr * con_neg, tpr * con_pos

    ###
    ###   Confusion Matrix
    ###    
    def print_confusion(self, ps=None, gs=None):
        """
        Args:
            ps (Iterator, optional): Prediction identifiers
            gs (ITerator, optional): Groundtruths identifiers
        """
        print '\n'
        for g, gt in self.groundtruths_binary(gs=gs):
            print '*{}\n\t\t{:16s}{}\t{}\t{}'.format(gt.pretty_name, ' ','0,1-loss', 'precision', 'recall')
            for p, pred in self.predictions_binary(ps=ps):
                p_flat, g_flat = self.compare_pred_gt(p, g, pred, gt)
                print '\t{:16s}\t{:.2e}\t{:.2e}\t{:.2e}'.format(
                    pred.pretty_name,
                    risk_estimate(est=p_flat, gt=g_flat, loss_func='zero-one-loss'),
                    precision(est=p_flat, gt=g_flat),
                    recall(est=p_flat, gt=g_flat))
            print'\n'

    ### 
    #  AUC
    ###
    def auc(self,
            max_fpr=None,
            ps=None, 
            gs=None,
            folder=None,
            save_auc=False,      # if not none, save auc and bootstraps
            load_auc=False,      # if true, load from available auc file if available
            nbootstraps=None,
            print_out=False):
        """Compute (partial) AUCs for each prediction and groundtruth
        
        If the comparison between a particular pred and gt yields either
            a.) no overlap in causes or effects,
            b.) one groundtruth class,
        then the resulting value in the AUC array will be a NaN.
        
        Args:
            max_fpr (None, optional): Compute partial auc up to this maximum false positive rate
            ps (None, optional): Description
            gs (None, optional): Description
            nbootstraps (bool, optional): number of nbootstrapss for percentile method (~1000 is sufficient)
            print_out (bool, optional): Print the result
        
        Returns:
            nbootstraps == None (default) 
                (2darray, list, list): Tuple of AUC (row: groundtruths x cols: predictions), predictions, groundtruths
            nbootstraps > 0
                (2darray, 2darray, list, list): Tuple of AUC (row: groundtruths x cols: predictions), predictions, groundtruths     
        """
        if save_auc or load_auc:
            if folder is None:
                raise Exception('Specify a folder to save to!')
        save_auc_folder = 'auc_buffer'
        if save_auc:
            if not os.path.exists(folder + '/' + save_auc_folder):
                os.makedirs(folder + '/' + save_auc_folder)

        auc_array = np.empty(shape=(self.number_predictions_continuous(ps=ps), self.number_groundtruths_binary(gs=gs)))
        print '[CompPredGt][{}] Computing AUC with {} predictions and {} groundtruths'.format(
            under_scores_to_camel_case(os.sys._getframe().f_code.co_name), auc_array.shape[0], auc_array.shape[1])
        auc_array[:] = np.nan
        g_ids = [] # g unique identifiers
        p_ids = [] # p unique identifiers
        # fill array
        for g_index, (g, gt) in enumerate(self.groundtruths_binary(gs=gs)):
            g_ids.append(g) # to preserve order
            for p_index, (p, pred) in enumerate(self.predictions_continuous(ps=ps)):
                if g_index == 0: # only fill names once!
                    p_ids.append(p)
                if self.comparison_exists(p, g):
                    compute_this_pair = True
                    if load_auc:
                        p_g_folder = folder + '/' + save_auc_folder + '/' + gt.binary_name + '/' + p
                        if os.path.exists(p_g_folder + '/auc.hdf5'): # only load if the path exists
                            with h5py.File(p_g_folder + '/auc.hdf5', 'r') as fid:
                                auc_array[p_index, g_index] = fid['auc'][...]
                            compute_this_pair = False # no need to compute the pair and save it again
                    if compute_this_pair:
                        auc_array[p_index, g_index] = roc_auc_score(*self.compare_pred_gt(p, g, pred, gt)[::-1], max_fpr=max_fpr)
                        if save_auc or load_auc:
                            p_g_folder = folder + '/' + save_auc_folder + '/' + gt.binary_name + '/' + p
                            if not os.path.exists(p_g_folder):
                                os.makedirs(p_g_folder)
                            with h5py.File(p_g_folder + '/auc.hdf5', 'w') as fid:
                                fid.create_dataset('auc', data=auc_array[p_index, g_index])
        # bootstrap, perform them afterwards and seperately
        if nbootstraps > 0:
            auc_array_bootstraps = np.empty(shape=(self.number_predictions_continuous(ps=ps), self.number_groundtruths_binary(gs=gs), nbootstraps))
            conf_ints =  np.empty(shape=(self.number_predictions_continuous(ps=ps), self.number_groundtruths_binary(gs=gs)))
            for g_index, (g, gt) in enumerate(self.groundtruths_binary(gs=gs)):
                g_ids.append(g) # to preserve order
                for p_index, (p, pred) in enumerate(self.predictions_continuous(ps=ps)):
                    if g_index == 0: # only fill names once!
                        p_ids.append(p)
                    if self.comparison_exists(p, g):
                        compute_this_pair = True
                        if load_auc:
                            p_g_folder = folder + '/' + save_auc_folder + '/' + gt.binary_name + '/' + p
                            if os.path.exists(p_g_folder + '/auc_bootstraps.hdf5'): # only load if the path exists
                                with h5py.File(p_g_folder + '/auc_bootstraps.hdf5', 'r') as fid:
                                    auc_array_bootstraps[p_index, g_index, :] = fid['auc_bootstraps'][...]
                                print '[CompPredGt][{}] Bootstrapped AUC for {} vs {} loaded from file.'.format(under_scores_to_camel_case(os.sys._getframe().f_code.co_name), gt.binary_name, p)
                                compute_this_pair = False # no need to compute the pair and save it again
                        if compute_this_pair:
                            p_flat, g_flat = self.compare_pred_gt(p, g, pred, gt)
                            _start_bootstrap = time.time()
                            auc_array_bootstraps[p_index, g_index, :] = compute_auc_bootstrap(p=p_flat, g=g_flat, nbootstraps=nbootstraps, max_fpr=max_fpr)
                            print '[CompPredGt][{}] Bootstrapped AUC for {} vs {} done. ({:.2f}sec.)'.format(under_scores_to_camel_case(os.sys._getframe().f_code.co_name), gt.binary_name, p, time.time() - _start_bootstrap)
                            if save_auc or load_auc:
                                p_g_folder = folder + '/' + save_auc_folder + '/' + gt.binary_name + '/' + p
                                if not os.path.exists(p_g_folder):
                                    os.makedirs(p_g_folder)
                                with h5py.File(p_g_folder + '/auc_bootstraps.hdf5', 'w') as fid:
                                    fid.create_dataset('auc_bootstraps', data=auc_array_bootstraps[p_index, g_index,:])
                        auc_array[p_index, g_index] = roc_auc_score(*self.compare_pred_gt(p, g, pred, gt)[::-1], max_fpr=max_fpr)
                    # if self.comparison_exists(p, g):
                    #     p_flat, g_flat = self.compare_pred_gt(p, g, pred, gt)
                    #     _start_bootstrap = time.time()
                    #     auc_array_bootstraps[p_index, g_index, :] = compute_auc_bootstrap(p=p_flat, g=g_flat, nbootstraps=nbootstraps, max_fpr=max_fpr)
                    #     print '[CompPredGt][{}] Bootstrapped AUC for {} vs {} done. ({:.2f}sec.)'.format(under_scores_to_camel_case(os.sys._getframe().f_code.co_name), gt.binary_name, p, time.time() - _start_bootstrap)
                    #     if save_auc:
                    #         p_g_folder = folder + '/' + save_auc_folder + '/' + gt.binary_name + '/' + p
                    #         with h5py.File(p_g_folder + '/auc_bootstraps.hdf5', 'w') as fid:
                    #             fid.create_dataset('auc_bootstraps', data=auc_array_bootstraps[p_index, g_index,:])
                    #     # compute the 2 SD's from the estimate above:
                    #     conf_ints[p_index, g_index] = 2 * np.sqrt(np.sum((auc_array_bootstraps[p_index, g_index, :] - auc_array[p_index, g_index])**2) / float(nbootstraps))
        if print_out:
            print '\n\tAUC\t{:12s}\t{}'.format(' ' if max_fpr is None else 'm={}'.format(max_fpr),''.join(['{:14s}'.format(g_name) for g_name in g_ids]))
            for p_index, p_name in enumerate(p_ids):
                print '\t{:16s}\t{}'.format(p_name,''.join(['{:.2f}{:10s}'.format(auc_array[p_index, g_id], '') for g_id in range(len(g_ids))]))
        # save auc and bootstraps
        # for g_index, (g, gt) in enumerate(self.groundtruths_binary(gs=gs)):
        #     for p_index, (p, pred) in enumerate(self.predictions_continuous(ps=ps)):
        #         if save_auc:
        #             p_g_folder = folder + '/' + save_auc_folder + '/' + g + '/' + p
        #             if not os.path.exists(p_g_folder):
        #                 os.makedirs(p_g_folder)
        #             with h5py.File(p_g_folder + '/auc.hdf5', 'w') as fid:
        #                 fid.create_dataset('auc', data=auc_array[p_index, g_index])
        #             if nbootstraps > 0:
        #                 with h5py.File(p_g_folder + '/auc_bootstraps.hdf5', 'w') as fid:
        #                     fid.create_dataset('auc_bootstraps', data=auc_array_bootstraps[p_index, g_index,:])
        if nbootstraps:
            return auc_array, conf_ints, ps or self.predictions_continuous_available, gs or self.groundtruths_binary_available
        else:
            return auc_array, ps or self.predictions_continuous_available, gs or self.groundtruths_binary_available

    def get_auc(self, *args, **kwargs):
        return self.auc(*args, **kwargs)

    def get_auc_for_method(self, 
                           pred, 
                           gt, 
                           max_fpr=None):
        auc_tuple = self.auc(max_fpr=max_fpr)
        auc_array = auc_tuple[0]
        pred_index = auc_tuple[1].index(pred)
        gt_index = auc_tuple[2].index(gt)
        return auc_array[pred_index, gt_index]

    def print_auc(self, max_fpr=None):
        self.auc(max_fpr=max_fpr, print_out=True)

    def plot_auc(self,
                 file_prefix=None,
                 file_postfix=None,
                 folder=None,
                 title=None,
                 plt_clim=True,             # colourbar limits
                 ps=None,                   # list of pred identifiers to plot in the desired order
                 gs=None,                   # list of gt identifiers to plot in the desired order
                 p_labels=None,             # either a) list of labels for pred identifiers in the same ordering or b) dict from pred identifiers mapping onto labels
                 g_labels=None,             # either a) list of labels for gt identifiers in the same ordering or b) dict from gt identifiers mapping onto labels
                 max_fpr=None,
                 show_auc_text=True,        # plot AUC text in each box.
                 figsize=None,
                 show_title=False,          # show titles
                 highlight_these=None,      # list of pairs of pred.names and gt.names to handle.
                 **kwargs):                 # kwargs go to auc()
        filename = '{}{}{}.pdf'.format(file_prefix or 'auc', file_postfix or '','_{}'.format(max_fpr) if max_fpr else '')
        folder = folder or os.path.curdir
        if not os.path.exists(folder):
            print '[{0}] creating AUC folder {1}'.format('CompPredGt', folder)
            os.makedirs(folder)
        title = title or 'AUC overview{}'.format('' if max_fpr is None else ' max fpr: ' + str(max_fpr))

        auc, ps, gs = self.auc(print_out=False, ps=ps, gs=gs, max_fpr=max_fpr, folder=folder, **kwargs)

        # Transform the AUC plot and the pred, gt labels to the desired order AND labels.
        if not p_labels:
            p_labels = [self.predictions[p].pretty_name for p in ps]
        if not g_labels:
            g_labels = [self.groundtruths[g].pretty_name for g in gs]

        # Need to process NaN's in auc array!
        if np.any(np.isnan(auc)):
            auc = np.ma.masked_array(auc, mask=np.isnan(auc))

        if figsize is None:
            # set figsize by heuristic.
            len_pred, len_gt = auc.T.shape
            figsize = ((len_gt + 6)/2., (len_pred + 3)/2.)
            if self.verbose: print '[CompPredGt][{}] setting figsize dynamically:'.format(
                under_scores_to_camel_case(os.sys._getframe().f_code.co_name)), figsize
        fig, ax = plt.subplots(figsize=figsize)
            
        # plot figure
        if plt_clim is None:
            cax = ax.matshow(auc.T, cmap='bwr') # looks better transposed; i.e. preds horizontally
        else:
            if hasattr(plt_clim, '__iter__'): # iterable
                cax = ax.matshow(auc.T, vmin=plt_clim[0], vmax=plt_clim[1], cmap='bwr')
                # cax = ax.matshow(auc.T, vmin=plt_clim[0], vmax=plt_clim[1])
            else:
                # set clims from auc, first find the largest value from 0.5
                lim = max(auc.max() - 0.5, 0.5 - auc.min())
                cax = ax.matshow(auc.T, vmin=0.5 - lim, vmax=0.5 + lim, cmap='bwr') 

        # fix axes and title
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(range(len(p_labels)))
        ax.set_yticks(range(len(g_labels)))
        ax.set_xticklabels(p_labels, rotation='vertical')
        ax.set_yticklabels(g_labels)  
        if show_title: plt.ylabel('ground-truth')
        if show_title: plt.title(title)
        plt.grid(linestyle='dotted', alpha=0.5 if show_auc_text else 1.)

        # highlight requested coordinates
        def highlight_box(x, y, axes, color='black'):
            axes.add_patch(
                mpatches.Rectangle(
                    xy=(x-0.5,y-0.5),
                    width=1,
                    height=1,
                    fill=False,
                    edgecolor=color,
                    linewidth=2,
                    clip_on=True,
                    # linestyle='dotted',
                    # hatch='/'
                    )
                )
            # ax.text(x,y, 'x', va='center', ha='center')
        if highlight_these:
            for coor in highlight_these:
                if type(coor[0]) is not int:
                    x = ps.index(coor[0])
                    y = gs.index(coor[1])
                else:
                    x,y, = coor
                highlight_box(x=x, y=y, axes=ax)

        # plot text
        if show_auc_text:
            for x, y in itertools.product(xrange(auc.T.shape[0]), xrange(auc.T.shape[1])): # recall; auc is plotted transposed
                if auc.T[x,y] is np.ma.masked: 
                    if auc.T.mask[x,y]:
                        text = 'X'
                    else:
                        text = '{:.2f}'.format(auc.T[x,y])
                else:
                    text = '{:.2f}'.format(auc.T[x,y])
                # print 'text', text
                ax.text(y, x, text, va='center', ha='center')

        # fig.colorbar(cax, format='%0.2f')
        cb = fig.colorbar(cax)
        cb.locator = ticker.MaxNLocator(nbins=8)
        cb.update_ticks()
        with plt.warnings.catch_warnings(): # Surpress mac warning..
            plt.warnings.simplefilter('ignore')
            try: #TODO fix this. Has to do with large labels?
                plt.tight_layout() 
            except ValueError:
                print 'WARNING: SKIPPING TIHGT LAYOUTS DUE TO LARGE LABELS'
            fig.savefig('{}/{}'.format(folder, filename))

    ###
    ### Generalized plots
    ###
    def plot_pairs(self, comparison_type, pairs, **kwargs):
        assert comparison_type in ['tpfp', 'roc', 'pr']
        print '[CompPredGt] plotting {}...'.format(comparison_type)
        if comparison_type == 'roc': return self.plot_roc_pairs(pairs, **kwargs)
        if comparison_type == 'tpfp': return self.plot_tpfp_pairs(pairs, **kwargs)
        if comparison_type == 'pr': return self.plot_pr_pairs(pairs, **kwargs)


    def plot_roc_pairs(self, pairs, lims=None, file_prefix=None, **kwargs):
        _start_time = time.time()

        # set defaults
        lims = lims or [(1,1)]
        file_prefix = file_prefix or 'roc'
        xlabel = 'False positive rate'
        ylabel = 'True positive rate'

        def function(p, g, pred, gt):
            fpr, tpr, _ = roc_curve(*self.compare_pred_gt(p, g, pred, gt)[::-1])
            return fpr, tpr

        self._plot_base_pairs(pairs, file_prefix=file_prefix, function=function, 
            lims=lims, xlabel=xlabel, ylabel=ylabel, **kwargs)
        if self.verbose > 1: print '[CompPredGt][{}] ROC pairwise plotted in {:.2f}s'.format(
            under_scores_to_camel_case(os.sys._getframe().f_code.co_name), time.time() - _start_time)

    def plot_tpfp_pairs(self, pairs, file_prefix=None, **kwargs):
        _start_time = time.time()

        # set defaults
        file_prefix = file_prefix or 'tpfp'
        xlabel = 'False positives'
        ylabel = 'True positives'

        self._plot_base_pairs(pairs, file_prefix=file_prefix, function=self.compute_fp_tp, 
            xlabel=xlabel, ylabel=ylabel, **kwargs)
        if self.verbose > 1: print '[CompPredGt][{}] TP FP pairwise plotted in {:.2f}s'.format(
            under_scores_to_camel_case(os.sys._getframe().f_code.co_name), time.time() - _start_time)

    def plot_pr_pairs(self, pairs, lims=None, file_prefix=None, file_extension='png', **kwargs):
        """Summary
        
        Args:
            pairs (TYPE): Description
            lims (None, optional): Description
            file_prefix (None, optional): Description
            file_extension (str, optional): Set extension to png here as the plots are absolutely huge in number of points
            **kwargs: Description
        """
        _start_time = time.time()

        # set defaults
        lims = lims or [(1,1)]
        file_prefix = file_prefix or 'pr'
        xlabel = 'Recall'
        ylabel = 'Precision'

        def function(p, g, pred, gt):
            pre, rec, _ = precision_recall_curve(*self.compare_pred_gt(p, g, pred, gt)[::-1])
            return rec, pre

        self._plot_base_pairs(pairs, file_prefix=file_prefix, file_extension=file_extension, 
            function=function, lims=lims, xlabel=xlabel, ylabel=ylabel, **kwargs)
        if self.verbose > 1: print '[CompPredGt][{}] PR pairwise plotted in {:.2f}s'.format(
            under_scores_to_camel_case(os.sys._getframe().f_code.co_name), time.time() - _start_time)

    def _plot_base_pairs(self, 
                         pairs, 
                         function,              # function over arguments (p, pred, g, gt) that returns x, y arrays
                         file_prefix,
                         file_postfix=None,
                         file_extension='pdf',
                         threshold=None,
                         figure=None,
                         folder=None,
                         lims=None,             # list of limits. If singleton, xlims, else xlim, ylim
                         plot_random=True,
                         legend_labels=None, 
                         group_plots_by=None,   # either 'gt', 'pred'
                         return_fig=False,
                         xlabel=None,
                         ylabel=None,
                         ):
        """Base plot function where multiple combinations of prediciton, groundtruth pairs are plotted in the same figure.
        
        Args:
            pairs (TYPE): Description
            file_prefix (None, optional): Description
            folder (None, optional): Description
            max_tp_fp (None, optional): Description
            plot_random (bool, optional): Description
            gts_labels (None, optional): Description
            legend_labels (None, optional): Description
            return_fig (bool, optional): Description
            threshold (None, optional): Description
        
        Raises:
            Exception: Description
        
        Returns:
            TYPE: Description
        """
        _start_time = time.time()
        folder = folder or os.path.curdir
        if not os.path.exists(folder):
            print '[{0}] creating folder {1}.'.format('CompPredGt', folder)
            os.makedirs(folder)
        file_postfix = ('_' + file_postfix) if file_postfix else ''

        if legend_labels: assert len(pairs) == len(legend_labels)

        # check that all pairs are available
        for p, g in pairs:
            if p not in self.predictions_continuous_available:
                raise Exception('Prediction %s not available; choose from: %s' % (p, ', '.join(self.predictions_continuous_available)))
            if g not in self.groundtruths_binary_available:
                raise Exception('Groundtruth %s not available; choose from: %s' % (g, ', '.join(self.groundtruths_binary_available)))
        if not threshold:
            threshold = self.thresholds[0]
            print '[CompPredGt] Warning: using {} as threshold'.format(self.thresholds[0])

        color_cycle = itertools.cycle(COLOR_CYCLE)
        # if group_plots_by: 
        def generate_pairs_colors_markers(group_plots_by=group_plots_by):
            """Generate the correct sorted iteration of pairs, colors and markers and cycle through it"""
            if group_plots_by == 'gt' or group_plots_by == 'groundtruth':
                unique_gts = [] # this preserves order of the color to assigned, cant use list(set(unique_gts))...
                for i in pairs:
                    if i[1] not in unique_gts:
                        unique_gts.append(i[1])
                gts_color_marker = dict(zip(unique_gts, [(color_cycle.next(), itertools.cycle(MARKER_CYCLE)) for i in unique_gts]))
                for i_pair, (p, g) in enumerate(pairs):
                    yield i_pair, p, g, gts_color_marker[g][0], gts_color_marker[g][1].next()
            elif group_plots_by == 'pred' or group_plots_by == 'prediction':
                unique_preds = []
                for i in pairs:
                    if i[0] not in unique_preds:
                        unique_preds.append(i[1])
                pred_color_marker = dict(zip(unique_preds, [(color_cycle.next(), itertools.cycle(MARKER_CYCLE)) for i in unique_preds]))
                for i_pair, (p, g) in enumerate(pairs):
                    color = pred_color_marker[p][0]
                    marker = pred_color_marker[p][1].next()
                    yield i_pair, p, g, color, marker
            else: # None setting
                for i_pair, (p, g) in enumerate(pairs):
                    marker_cycle = itertools.cycle(itertools.chain([None,], MARKER_CYCLE)) # not sure of this works?
                    if i_pair % len(COLOR_CYCLE) == 0:
                        marker = marker_cycle.next()
                    yield i_pair, p, g, color_cycle.next(), marker

         # create figure 
        fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
        plt_lines = [] # list of lines that are plotted, for easier manipulation

        # loop over tuple (pair index, prediction, groundtruth, color, marker)
        for i_pair, p, g, color, marker in generate_pairs_colors_markers(group_plots_by):
            if self.comparison_exists(p, g):
                pred = self.predictions[p] # get the generated prediction
                gt = self.groundtruths[g].make_binary(percentile=threshold, strict=False)
                x, y = function(p, g, pred, gt)
                # clip the values of x, y according to x-limits to VASTLY reduce plotting time
                if lims:
                    xlim_max = max([xlim[0] if type(xlim) is tuple else xlim for xlim in lims])
                    index = np.argmax(x > xlim_max)
                    if index:
                        x = x[:index + 1]
                        y = y[:index + 1]
                plt_lines +=  plt.plot(x, y,
                    color=color,
                    linestyle=self.linestyle_tpfp,
                    marker=marker,
                    alpha=0.8,
                    label=r'{}'.format(legend_labels[i_pair] if legend_labels is not None else '{} - {}'.format(pred.pretty_name, gt.pretty_name)))
        if plot_random:
            # single random plot as there is one threshold here!
            line_cycle = itertools.cycle(LINE_CYCLE_RANDOM)
            linestyle = line_cycle.next()
            for i, g in enumerate(self.random_groundtruths.keys()):
                # comparison should always exists here!
                gt = self.random_groundtruths[g].make_binary(percentile=threshold)
                x, y = function(p, '_random', pred, gt)
                # clip the values of x, y according to x-limits to VASTLY reduce plotting time
                if lims:
                    xlim_max = max([xlim[0] if type(xlim) is tuple else xlim for xlim in lims])
                    index = np.argmax(x > xlim_max)
                    if index:
                        x = x[:index + 1]
                        y = y[:index + 1]
                if i == 0:
                    plt_lines += plt.plot(x, y, label='random {:.2f}'.format(threshold), alpha=0.5, linestyle='dotted', color='{}'.format(0.1 + i * 0.8 / float(len(self.random_predictions) -1 )))
                else:
                    plt_lines += plt.plot(x, y, alpha=0.5, linestyle=linestyle, color='{}'.format(0.1 + i * 0.8 / float(len(self.random_predictions) -1 )))
            if False:
                line_cycle = itertools.cycle(LINE_CYCLE_RANDOM)
                for thr in self.thresholds:
                    linestyle = line_cycle.next()
                    for i, g in enumerate(self.random_groundtruths.keys()):
                        # comparison should always exists here!
                        gt = self.random_groundtruths[g].make_binary(percentile=thr)
                        x, y = function(p, '_random', pred, gt)
                        # clip the values of x, y according to x-limits to VASTLY reduce plotting time
                        if lims:
                            xlim_max = max([xlim[0] if type(xlim) is tuple else xlim for xlim in lims])
                            index = np.argmax(x > xlim_max)
                            if index:
                                x = x[:index + 1]
                                y = y[:index + 1]
                        if i == 0:
                            plt_lines += plt.plot(x, y, label='random {:.2f}'.format(thr), alpha=0.5, linestyle='dotted', color='{}'.format(0.1 + i * 0.8 / float(len(self.random_predictions) -1 )))
                        else:
                            plt_lines += plt.plot(x, y, alpha=0.5, linestyle=linestyle, color='{}'.format(0.1 + i * 0.8 / float(len(self.random_predictions) -1 )))
        plt.legend(loc='best', fancybox=True) #, ncol=1 if len(pairs) < 6 else 2, labelspacing=0.1)
        if xlabel is not None: plt.xlabel(xlabel)
        if ylabel is not None: plt.ylabel(ylabel)
        # set plot limits
        if lims: 
            for l in lims:
                fig_name = '{}/{}'.format(folder, file_prefix)
                _start_time = time.time()
                if type(l) is tuple:    # l = (xlim, ylim)
                    xlim, ylim = l
                    plt.xlim(0, xlim)
                    plt.ylim(0, ylim)
                    fig_name += '_xlim{}_ylim{}'.format(xlim, ylim)
                else:                   # l = xlim
                    xlim = l
                    plt.xlim(0, xlim)
                    fig_name += '_xlim{}'.format(xlim)
                    # set ylim dynamically
                    top = 0
                    # for line in plt.gca().lines:
                    for line in plt_lines:
                        xd = line.get_xdata()
                        yd = line.get_ydata()
                        y_displayed = yd[((xd < xlim))]
                        new_top = np.max(y_displayed) * 1.05
                        if new_top > top: top = new_top
                    if top > 0:
                        plt.ylim(0,top)
                # after the xlimits are set, compute the number of markers
                for line in plt.gca().lines:
                    # n_markers = int(np.argmax(line.get_xdata() > xlim) / MARKER_NUMBER) # number of datapoints 
                    try:
                        markevery = int(np.argwhere(line.get_xdata() > xlim)[0][0] / MARKER_NUMBER)
                        if markevery > 0:
                            line.set_markevery(markevery)
                    except IndexError: # out of bounds means that we get no results
                        pass
                    # else:
                    #     line.set_markevery(1)
                # save limited figure
                fig.savefig('{}_perc{:.2f}{}.{}'.format(fig_name, threshold, file_postfix, file_extension), bbox_inches='tight', dpi=DPI)
                # print 'LIMS PLOTTED IN {:.2f}sec'.format(time.time() - _start_time)
        else:
            # save overall figure
            fig.savefig('{}/{}_perc{:.2f}{}.{}'.format(folder, file_prefix, threshold, file_postfix, file_extension), bbox_inches='tight', dpi=DPI)
        if return_fig:
            return fig
        else:
            plt.close(fig)
            del fig
