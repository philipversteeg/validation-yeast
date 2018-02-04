## methods
import methods
from methods import *       # quick access

methods_dict = dict((k, v) for k, v in methods.__dict__.iteritems() if issubclass(type(v), type(Predictor)) and v is not 'Predictor')
methods_available = methods_dict.keys()

import parallel

## todo
# __all__ = ['' ,'' ]