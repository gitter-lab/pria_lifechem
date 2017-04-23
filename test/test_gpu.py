from theano import function, config, shared, tensor
import numpy
import time
import rpy2.robjects
import rpy2.robjects.packages
from croc import BEDROC, ScoredData
import virtual_screening
# import virtual_screening.data_preparation
# import virtual_screening.function
# import virtual_screening.evaluation
# import virtual_screening.models.CallBacks
# import virtual_screening.models.deep_classification
# import virtual_screening.models.deep_regression
# import virtual_screening.models.vanilla_lstm


vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
                      ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
