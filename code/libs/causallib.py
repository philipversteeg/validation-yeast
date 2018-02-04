"""Asymmetric objects with distinct 'cause' and 'effect' dimensions.

* CausalArray
* UnitsOfMeasure
"""
import numpy as np
import h5py
import os
import copy

class CausalArray(object):
    """Derived numpy.ndarray with distinct 'cause' and 'effect' dimensions.
    
    Attributes:
        array (numpy.ndarray): Array of cause x effect dimensions
        causes (list): List of unique causes corresponding to array
        effects (list): List of unique effects corresponding to array
        file (str): Location of saved version on disk
        name (str): Name
        pretty_name (str): Printable name
        threshold (int): Cutoff value used for thresholding a continuous array
        units (UnitsOfMeasure): UnitsOfMeasure instance
    """
    _default_name = 'causalarray'

    def __init__(self, array, causes, effects=None, mask=None, remove_self_loops=None, units=None, file=None, name=None, sort=False):
        """Derived numpy.ndarray with distinct 'cause' and 'effect' dimensions.
        
        Args:
            array (numpy.ndarray): Array of cause x effect dimensions
            causes (list): List of unique causes corresponding to array
            effects (list, optional): List of unique effects corresponding to array, if None copy the causes
            mask (numpy.ndarray, optional): Array used to mask invalued entries of array
            remove_self_loops (bool, optional): Mask away the diagonal entries
            units (None, optional): UnitsOfMeasure object or create one by UnitsOfMeasure(units), default of None corresponds to 'unset'.
            file (str, optional): Location of saved version on disk
            name (str, optional): Name
        """
        assert issubclass(type(array), np.ndarray)
        assert issubclass(type(causes), list) # set won't do, unordered... Enforce unique list!
        self.array = array
        self.causes = causes
        self.effects = effects or causes # default to effects = causes if not specified
        assert len(self.causes), len(self.effects) == array.shape
        assert len(set(self.causes)) == len(self.causes) # uniqueness
        assert len(set(self.effects)) == len(self.effects)
        if mask: self.mask(mask=mask)
        if remove_self_loops: self.remove_self_loops()
        self.units = units
        self.file = file
        self.name = self._default_name if name is None else name
        self.pretty_name = None
        if file is not None and not os.path.exists(file): self.save(file)
        # if file is not None: self.save(file)
        if sort:
            self._sort_self()


    @classmethod
    def from_dict(cls, dictionary, sort=True, name=None):
        """Alternative constructor from dictionary.
        
        Args:
            dictionary (dict): Dict containing as keys the causes, with each value an iterable of effects.
            sort (bool, optional): Description
            name (None, optional): Name of CausalArray
        """
        causes = dictionary.keys()
        effects = list(set([i for j in dictionary.values() for i in j]))
        if sort: 
            causes.sort()
            effects.sort()

        array = np.zeros(shape=(len(causes),len(effects)), dtype=bool)
        for i, c in enumerate(causes):
            # print c, len(dictionary[c])
            for e in dictionary[c]:
                # print i, effects.index(e)
                array[i, effects.index(e)] = True
        return cls(array=array, causes=causes, effects=effects, name=name)
    
    @classmethod
    def random(cls, ncauses=None, neffects=None, binary=False):
        """Alternative constructor, random array"""
        if ncauses is None: ncauses = np.random.randint(1,10)
        if neffects is None: neffects = np.random.randint(1,10)
        if binary:
            return cls(array=np.random.randint(low=0, high=2, size=(ncauses, neffects)).astype(bool),
                causes=range(ncauses), effects=range(neffects), 
                name='random_' + str(ncauses) + '_' + str(neffects))
        else:
            return cls(array=np.random.random((ncauses, neffects)), 
                causes=range(ncauses), effects=range(neffects), 
                name='random_' + str(ncauses) + '_' + str(neffects))

    def __repr__(self):
        return ('\n[CausalArray] name:       {self.name}\n'
                '              causes:     {self.ncauses}\n'
                '              effects:    {self.neffects}\n'
                '              binary:     {self.binary}'
                '{mask}{file}{units}{threshold}\n').format(self=self, 
                    mask='' if not self.masked else '\n              masked:     True',
                    file='' if self.file is None else '\n              file:       {}'.format(os.path.realpath(self.file)),
                    units='' if not self.units.exists else '\n              units:      {}'.format(self.units),
                    threshold='' if not 'threshold' in vars(self).keys() else '\n              threshold:  {}'.format(self.threshold))

    def __getitem__(self, args):
        return self.array.__getitem__(args)

    # ## forward all numpy methods to array
    # def __getattr__(self, attr):
    #     if attr in dir(self.array):
    #         return getattr(self.array, attr)

    def _sort_self(self):
        """ Sort causes and effects and the array."""
        sorted_ca, sorted_ef = sorted(self.causes), sorted(self.effects)
        if sorted_ca == self.causes: sorted_ca = None
        if sorted_ef == self.effects: sorted_ef = None
        self.sort_in_place(ca=sorted_ca, ef=sorted_ef)

    @property
    def pretty_name(self):
        if self._pretty_name is not None:
            return self._pretty_name
        else:
            name = self.name
            if 'threshold' in vars(self):
                name += ' ({:.2f})'.format(self.threshold)
            if self.has_units:
                name += ' [{}]'.format(self.units)
            return name

    @pretty_name.setter
    def pretty_name(self, pn=None):
        self._pretty_name = pn

    @property
    def binary_name(self):
        name = self.name
        if 'threshold' in vars(self):
            name += '_{:.2f}'.format(self.threshold)
        return name

    def index_causes(self, ca):
        if hasattr(ca, '__iter__'):
            return [self.causes.index(i) for i in ca]
        else:
            return self.causes.index(ca)

    def index_effects(self, ef):
        if hasattr(ef, '__iter__'):
            return [self.effects.index(i) for i in ef]
        else:
            return self.effects.index(ef)

    def sort_in_place(self, ca=None, ef=None):
        if ca:
            assert len(ca) == self.ncauses
            assert all((i in self.causes for i in ca))
            self.array = self.array[self.index_causes(ca=ca) ,:]
            self.causes=ca
        if ef:
            assert len(ef) == self.neffects
            assert all((i in self.effects for i in ef))
            self.array = self.array[:, self.index_effects(ef=ef)]
            self.effects=ef

    def reduce(self, causalarray=None, causes=None, effects=None, name=None, deepcopy=False):
        """Summary
        
        Args:
            causalarray (None, optional): Description
            causes (None, optional): Description
            effects (None, optional): Description
            name (None, optional): Description
            deepcopy (bool, optional): Description
        
        Returns:
            TYPE: Description
        """
        if deepcopy:
            _instance = copy.deepcopy(self)
        else:
            _instance = self
        if causes is not None: # as 0 can be a single cause identifier.
            if not hasattr(causes, '__iter__'):
                causes = (causes,)
            _instance.array = _instance.array[_instance.index_causes(causes), :]
            _instance.causes = causes
        if effects is not None:
            if not hasattr(effects, '__iter__'):
                effects = (effects,)
            _instance.array = _instance.array[:, _instance.index_effects(effects)]
            _instance.effects = effects
        if name: _instance.name = name
        return _instance

    def add_empty_vars(self, causes=None, effects=None):
        """Add cause(s) or effect(s) and fill the array at the appropriate indices with empty elements
        
        Args:
            causes (None, optional): Description
            effects (None, optional): Description
        """
        if causes:
            causes = causes if hasattr(causes, '__iter__') else (causes,) # make sure iterable
            causes = [i for i in causes if not i in self.causes] # only include causes that are not already in self.causes
            if len(causes) > 0: # make sure there are any causes left
                self.array = np.insert(arr=self.array, obj=[self.ncauses for i in range(len(causes))], values=0, axis=0)
                self.causes.extend(causes)
        if effects:
            effects = effects if hasattr(effects, '__iter__') else (effects,) # make sure iterable
            effects = [i for i in effects if not i in self.effects] # only include effects that are not already in self.effects
            if len(effects) > 0: # make sure there are any effects left
                self.array = np.insert(arr=self.array, obj=[self.neffects for i in range(len(effects))], values=0, axis=1)
                self.effects.extend(effects)

    ###
    ### Operations returning a masked version.
    ###
    def mask(self, mask):
        self.array = np.ma.masked_array(self.array, mask=mask)
        return self

    @property
    def num_masked(self, mask):
        return self.array.mask.sum()

    @property
    def flatten(self):
        if self.masked:
            return self.array.compressed()
        else:
            return self.array.flatten()

    @property
    def compressed(self):
        return self.flatten

    @property
    def masked(self):
        return type(self.array) is np.ma.MaskedArray

    def remove_self_loops(self):
        diagonal_mask = np.zeros(self.shape, dtype='bool')
        common_ca_ef = set.intersection(set(self.causes), set(self.effects))
        diagonal_mask[[self.causes.index(i) for i in common_ca_ef], [self.effects.index(i) for i in common_ca_ef]] = True
        self.mask(mask=diagonal_mask)
        return self

    def remove_cause_indices(self, indices, strict=True):
        assert hasattr(indices, '__iter__')
        keep_list = [i for i in range(self.ncauses) if i not in indices]
        self.causes = [self.causes[i] for i in keep_list]
        self.array = self.array[keep_list,:]
        return self        

    # @property
    # def map_causes(self):
    #     return dict(zip(self.causes, range(len(self.causes))))

    # @property
    # def map_effects(self):
    #     return dict(zip(self.effects, range(len(self.effects))))

    @property
    def ncauses(self):
        return len(self.causes)

    @property
    def neffects(self):
        return len(self.effects)
    
    @property
    def ncausaleffects(self):
        if self.masked:
            return self.array.size - self.array.mask.sum()
        else:
            return self.array.size

    @property
    def num_nonzero(self):
        """Number of nonzero edges in (masked) array.

        Note: np.sum correctly skips masked values in np.ma.ma object.
        """
        if self.binary:
            return np.sum(self.array)
        else:
            return np.sum(self.array != 0)

    @property
    def nonzero(self):
        return self.num_nonzero

    @property
    def num_zero(self):
        if self.binary:
            return self.array.size - np.sum(self.array)
        else:
            return self.array.size - np.sum(self.array != 0)

    @property
    def zero(self):
        return self.num_zero

    @property
    def binary(self):
        return self.array.dtype is bool or self.array.dtype == np.bool_ # needed as some GT's loaded from disk are a different boolean type.
    
    @property
    def is_binary(self):
        return self.binary

    @property
    def shape(self):
        return self.array.shape

    @property
    def size(self):
        if self.masked:
            return np.logical_not(self.array.mask).sum()
        else:
            return self.array.size

    @property
    def sorted(self):
        return sorted(self.causes) == self.causes and sorted(self.effects) == self.effects

    @property
    def has_units(self):
        return self.units.exists

    @property
    def has_units_not_none(self):
        return self.units.is_not_none

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, obj):
        if type(obj).__name__ == UnitsOfMeasure.__name__: self._units = obj
        else: self._units = UnitsOfMeasure(obj)

    def normalize_units_array(self, compare_with, scale_units):
        """Return a numpy array that is rescaled according to compare_causalarray using scale_units
        
        Args:
            compare_with (CausalArray, UnitsOfMeasure or str): units to compare with
            scale_units (Dict): dictionary containing the scale parameters for the units
        
        Returns:
            TYPE: Description
        """
        std_ca = np.array([scale_units[i] for i in self.causes])
        std_ef = np.array([scale_units[i] for i in self.effects])
        # rescale the array
        num_ca, num_ef = diff_unit_of_measure(compare_with, self)
        # faster implementation..
        if num_ca != 0 and num_ef != 0:
            return self.array * np.power(np.column_stack(self.neffects * [std_ca,]), float(num_ca)) * np.power(np.row_stack(self.ncauses * [std_ef,]), float(num_ef))
        elif num_ca == 0 and num_ef != 0:
            return self.array * np.power(np.row_stack(self.ncauses * [std_ef,]), float(num_ef))
        elif num_ca != 0 and num_ef == 0:
            return self.array * np.power(np.column_stack(self.neffects * [std_ca,]), float(num_ca))
        else:
            return self.array

    def normalize_units(self, compare_with, scale_units, deepcopy=False):
        """Normalize units of array and copy to return a new instance.
        
        Args:
            compare_with (CausalArray, UnitsOfMeasure or str): Units to compare with
            scale_units (Dict): Dictionary containing the scale parameters for the units
            deepcopy (bool, optional): Return a new copy of the instance, with the transformed units
        
        Returns:
            TYPE: Self object
        """
        if deepcopy:
            _instance = copy.deepcopy(self)
        else:
            _instance = self
        _instance.array = _instance.normalize_units_array(compare_with=compare_with, scale_units=scale_units)
        if type(compare_with).__name__ is CausalArray.__name__:
            _instance.units = UnitsOfMeasure(compare_with._units)
        else:
            _instance.units = copy.deepcopy(compare_with)
        return _instance

    def make_binary(self, percentile, strict=True):
        # print 'calling', self.name, 'make binary with value:', percentile 
        self.threshold = percentile
        if self.binary:
            if strict:
                raise Exception('Already binary!')
            else:
                return self
        else:
            # _instance = copy.deepcopy(self)
            with np.errstate(invalid='ignore'): 
                # _instance.array = _instance.array >= np.percentile(_instance.flatten, float(percentile)) 
                array = np.nan_to_num(self.array) >= np.percentile(np.nan_to_num(self.flatten), float(percentile)) 
                # note that >= is needed, as some may have np.inf value, e.g. rel.norm.
            _instance = CausalArray(array=array, name=self.name, causes=self.causes, effects=self.effects, units=self.units)
            _instance.threshold = percentile
            return _instance
    
    def save(self, file=None, overwrite=True, compression='gzip', verbose=False):
        if file is None:
            if self.name is None or self.name == self._default_name:
                raise ValueError('No file argument given and no name attribute specified, can\'t save to default file.')
            else:
                # raise Warning('No file argument given, saving to %s.hdf5' % self.name)
                print '[warning] No file argument given, saving to %s.hdf5' % self.name
                file = '%s.hdf5' % self.name
        if os.path.exists(file) and overwrite:
            os.remove(file)
            if verbose: print 'Overwriting CausalArray at {}.'.format(os.path.relpath(file))
        else:
            if verbose: print 'Saving CausalArray at {}.'.format(os.path.relpath(file))
        assert compression in ['gzip', None]
        self.file = file
        with h5py.File(file, 'w') as fid:
            fid.create_dataset('/array', data=self.array, compression=compression)
            fid.create_dataset('/causes', data=self.causes, compression=compression)
            fid.create_dataset('/effects', data=self.effects, compression=compression)
            if self.units.is_not_none:
                fid.create_dataset('/units', data=self.units._unitstring)
            fid.create_dataset('/name', data=self.name)
        # with h5py.File(file, 'w') as fid:
        #     fid.create_dataset('/array', data=self.array, compression=compression)
        #     fid.create_dataset('/causes', data=self.causes, compression=compression)
        #     fid.create_dataset('/effects', data=self.effects, compression=compression)
        #     fid.create_dataset('/name', data=self.name)
        # perhapse the problem is single-writer-multiple-reader mode? See http://docs.h5py.org/en/latest/swmr.html.
        # with h5py.File(file, 'w', libver='latest') as fid:
        #     fid.create_dataset('/array', data=self.array, compression=compression)
        #     fid.create_dataset('/causes', data=self.causes, compression=compression)
        #     fid.create_dataset('/effects', data=self.effects, compression=compression)
        #     fid.create_dataset('/name', data=self.name)
        #     fid.swmr_mode = True

    @classmethod
    def load(cls, file, verbose=False):
        if verbose: print 'Loading CausalArray at {}.'.format(os.path.relpath(file))
        if os.path.splitext(file)[1] == '.hdf5':
            # more safe?
            try:
                with h5py.File(file, 'r') as fid:
                    array = fid['/array'][...]
                    causes = list(fid['/causes'][...])
                    effects = list(fid['/effects'][...])
                    try: # Fix for stuff written from R with rhdf5:
                        name = str(fid['/name'][...][0])
                    except IndexError:
                        name = str(fid['/name'][...])
                    try:
                        unitstring = str(fid['/units'][...])
                    except KeyError as ke:
                        unitstring = None
                # fid.close()
            except IOError as e:
                print 'Error when opening file %s' % file
                raise IOError(e)
        else:
            raise IOError('{0} could not be loaded'.format(file))
        return cls(array=array, causes=causes, effects=effects, file=file, units=unitstring, name=name)

    def intersection(self, causalarray, sorted=True):
        """Compute intersection of causes and effects with another causal array.
        
        Args:
            causalarray (TYPE): Description
            order (bool, optional): If true (default), sort by 

        Returns:
            (list, list): Tuple of intersecting causes and effects.
        """
        if sorted:
            return sorted(set.intersection(set(self.causes), set(causalarray.causes)), set.intersection(set(self.effects)), set(causalarray.effects))
        else:
            return [i for i in self.causes if i in causalarray.causes], [i for i in self.effects if i in causalarray.effects]

def intersect_causalarrays(*arrays):
    """For a given set of arrays, give the intersection of common causes and effects."""
    assert all((issubclass(type(i), CausalArray) for i in arrays))

    causes = sorted(set.intersection(*[set(i.causes) for i in arrays]))
    effects = sorted(set.intersection(*[set(i.effects) for i in arrays]))

    return causes, effects

def union_causalarrays(*arrays):
    """For a given set of arrays, give the union of common causes and effects."""
    assert all((issubclass(type(i), CausalArray) for i in arrays))

    causes = sorted(set.union(*[set(i.causes) for i in arrays]))
    effects = sorted(set.union(*[set(i.effects) for i in arrays]))

    return causes, effects

def reduce_causalarrays(*arrays):
    ca, ef = intersect_causalarrays(*arrays)
    for i in arrays: i.reduce(causes=ca, effects=ef) 

def convert_to_binary(causalarrays, thresholds, add_current_binary_objects=True):
    """Convert dict of causalarrays to binary using thresholds
    
    Args:
        causalarrays (TYPE): Description
        thresholds (TYPE): Description
        add_current_binary_objects (bool, optional): Description
    """
    assert type(causalarrays) is dict

    result = {}
    for key in causalarrays:
        ca_obj = causalarrays[key]
        if ca_obj.binary:
            if add_current_binary_objects:
                result[key] = ca_obj
        else:
            for t in thresholds:
                result['{}_{}'.format(key, int(t))] = ca_obj.make_binary(percentile=float(t), strict=True)
    assert all([result[i].binary for i in result])
    return result

def test_array(ncauses=None, neffects=None):
    if ncauses is None: ncauses = np.random.randint(1,10)
    if neffects is None: neffects = np.random.randint(1,10)
    return CausalArray(
        array=np.random.random((ncauses, neffects)),
        causes=range(ncauses),
        effects=range(neffects),
        name='testarray_' + str(ncauses) + '_' + str(neffects))


class UnitsOfMeasure(object):
    """Holds units of measure for a causalarray in factors of [cause] and [effect].
    
    Can have one of several different values:
        0 :: unknown
        1 :: (physical) unitless
        i^a :: 'a' multiples of the unit of measure of the cause variable 'i'
        j^b :: 'b' multiples of the unit of measure of the cause variable 'j'
        None :: not set
    """
    def __init__(self, unitstring=None):
        self.unitstring = unitstring

    def __repr__(self):
        if self._unitstring == '0': return 'unknown'
        else: return self._unitstring

    @property
    def unitstring(self):
        if self._unitstring is None or self._unitstring == '0':
            raise UnknownUnits('Unknown units found!')
        return self._unitstring

    @unitstring.setter
    def unitstring(self, value):
        """Construct unit string and set the i and j variables for convenience."""
        if value is None or value.lower() == 'none': self._unitstring = None
        if value is 0 or value == '0': self._unitstring = '0'
        elif value is 1 or value == '1': 
            self._unitstring = '1'
            self._i = 0
            self._j = 0
        elif type(value) is str:
            split = value.split('/')
            self._i = split[0].count('i')
            self._j = split[0].count('j')
            if len(split) == 2:
                self._i -= split[1].count('i')
                self._j -= split[1].count('j')
            # compose the unitstring
            self._unitstring = ''
            if self._i > 0:
                self._unitstring += self._i * 'i'
            if self._j > 0:
                self._unitstring += self._j * 'j'
            if self._i <= 0 and self._j <= 0:
                self._unitstring += '1'
            if self._i < 0 or self._j < 0:
                self._unitstring += '/'
                if self._i < 0:
                    self._unitstring += -self._i * 'i'
                if self._j < 0:
                    self._unitstring += -self._j * 'j'

    @property
    def exists(self):
        return not (self._unitstring is None or self._unitstring == '0')

    @property
    def is_not_none(self):
        return self._unitstring is not None

    def __div__(self, other):
        return diff_unit_of_measure(self, other)

    def __truediv__(self, other):
        return diff_unit_of_measure(self, other)


class UnknownUnits(Exception):
    """Exception for unknown units when attempting a unit conversion."""

class ZeroCausalArray(Exception):
    """Exception for a zero set of predictions or groundtruths."""
    pass

def diff_unit_of_measure(x, y):
    """Return the units of measure difference [x] / [y] = ['i']^a ['j']^b as the tuple (a,b)
    where 'i' are the cause variable and 'j' the effect variable.
    
    Args:
        x (TYPE): Description
        y (TYPE): Description
    
    Raises:
        UnknownUnits: Description
    
    Returns:
        (int, int): Tuple of a,b, difference in units.
    """
    try:
        xunits = x.units
    except AttributeError:
        if type(x) is str:
            xunits = UnitsOfMeasure(x)
        else:
            xunits = x
    try:
        yunits = y.units
    except AttributeError:
        if type(y) is str:
            yunits = UnitsOfMeasure(y)
        else:
            yunits = y
    # print type(x)
    # assert issubclass(type(x), UnitsOfMeasure)
    # assert issubclass(type(y), UnitsOfMeasure)
    if xunits._unitstring is None or xunits._unitstring == '0':
        if issubclass(type(x), CausalArray):
            raise UnknownUnits('Unknown unit found for %s' % x.name)
        else:
            raise UnknownUnits('Unknown unit found')
    if yunits._unitstring is None or yunits._unitstring == '0':
        if issubclass(type(x), CausalArray):
            raise UnknownUnits('Unknown unit found for %s' % y.name)
        else:
            raise UnknownUnits('Unknown unit found')
    return xunits._i - yunits._i, xunits._j - yunits._j

