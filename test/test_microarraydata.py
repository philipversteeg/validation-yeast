from ..microarraydata import Hughes
import numpy as np

def test_hughes():
    """Test Hughes"""
    x = Hughes.load()

    x_small = x.select(gen=100, mut=50, obs=20)

    assert x_small.ngenes == 100
    assert x_small.nmutants == 50
    assert x_small.nobs == 20

    x_small.save('tests/tmp.hdf5')
    x_small2 = Hughes.load('tests/tmp.hdf5')

    # assert np.all(x_small.obser == x_small2.obser)
