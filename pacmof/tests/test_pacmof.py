"""
Unit and regression test for the pacmof package.

"""
# Uses the pytest fixtures decorator to take the energy grid and use it for a series of tests including
# values, writing, isotherms and etc
# Import package, test suite, and other packages as needed

import pacmof
import pytest
import sys

#%%
def test_pyIsoP_imported():
        """Sample test, will always pass so long as import statement worked"""
        assert "pacmof" in sys.modules

def test_serial_calc():
    import os
    import pacmof as pac

    path_to_cif = os.path.join(os.path.dirname(pacmof.__file__),'data','FIFFUX_clean_DDEC_ML_chg.cif')
    path_to_data = os.path.join(os.path.dirname(pacmof.__file__),'data','FIFFUX_clean_featurespace_ML_charge.dat')
    import pandas as pd
    import numpy as np
    df = pd.read_csv(path_to_data, delim_whitespace=True)
    charges = df.ML_charge.values

    data = pac.get_charges_single_serial(path_to_cif)
    predicted = data.info['_atom_site_charge']
    print("Testing the serial calculation")
    assert np.max(np.abs(predicted-charges)) <= 1E-1
#
def test_single_large():
    import os
    import pacmof as pac

    path_to_cif = os.path.join(os.path.dirname(pacmof.__file__),'data','FIFFUX_clean_DDEC_ML_chg.cif')
    path_to_data = os.path.join(os.path.dirname(pacmof.__file__),'data','FIFFUX_clean_featurespace_ML_charge.dat')
    import pandas as pd
    import numpy as np
    df = pd.read_csv(path_to_data, delim_whitespace=True)
    charges = df.ML_charge.values

    data = pac.get_charges_single_large(path_to_cif)
    predicted = data.info['_atom_site_charge']
    print("Testing the single large calculation")
    assert np.max(np.abs(predicted-charges)) <= 1E-1


def test_multiple_parallel():
    import os
    import pacmof as pac

    path_to_cif = os.path.join(os.path.dirname(pacmof.__file__),'data','FIFFUX_clean_DDEC_ML_chg.cif')
    path_to_data = os.path.join(os.path.dirname(pacmof.__file__),'data','FIFFUX_clean_featurespace_ML_charge.dat')
    import pandas as pd
    import numpy as np
    df = pd.read_csv(path_to_data, delim_whitespace=True)
    charges = df.ML_charge.values

    data_all = pac.get_charges_multiple_parallel([path_to_cif])
    predicted = data_all[0].info['_atom_site_charge']
    print("Testing the multiple parallel calculation")
    assert np.max(np.abs(predicted-charges)) <= 1E-1

def test_multiple_onebyone():
    import os
    import pacmof as pac

    path_to_cif = os.path.join(os.path.dirname(pacmof.__file__),'data','FIFFUX_clean_DDEC_ML_chg.cif')
    path_to_data = os.path.join(os.path.dirname(pacmof.__file__),'data','FIFFUX_clean_featurespace_ML_charge.dat')
    import pandas as pd
    import numpy as np
    df = pd.read_csv(path_to_data, delim_whitespace=True)
    charges = df.ML_charge.values

    data_all = pac.get_charges_multiple_onebyone([path_to_cif])
    predicted = data_all[0].info['_atom_site_charge']
    print("Testing the multiple one by one calculation")
    assert np.max(np.abs(predicted-charges)) <= 1E-1


# def test_single_large_calc():
#     import os
#     import pacmof as pac
#     path_to_cif = os.path.dirname(pacmof.__file__)+'/data/FIFUX_clean_DDEC_ML_chg.cif.cif'
#     path_to_data = os.path.dirname(pacmof.__file__)+'/data/FIFUX_clean_featurespace_ML_charge.cif.cif'
#     import pandas as pd
#     import numpy as np
#     df = pd.read_csv(path_to_data, delim_whitespace=True)
#     charges = df.ML_charge.values
#
#     data = pac.get_charges_single_large(path_to_cif)
#     predicted = data.info['_atom_site_charge']
#     print("Testing the single large calculation")
#     assert np.max(np.abs(predicted-charges)) <= 1E-4
#
#
# def test_multiple_parallel_calc():
#     import os
#     import pacmof as pac
#     path_to_cif = os.path.dirname(pacmof.__file__)+'/data/FIFUX_clean_DDEC_ML_chg.cif.cif'
#     path_to_data = os.path.dirname(pacmof.__file__)+'/data/FIFUX_clean_featurespace_ML_charge.cif.cif'
#     import pandas as pd
#     import numpy as np
#     df = pd.read_csv(path_to_data, delim_whitespace=True)
#     charges = df.ML_charge.values
#
#     data_all = pac.get_charges_single_large([path_to_cif])
#     predicted = data_all[0].info['_atom_site_charge']
#     print("Testing the multiple parallel calculation")
#     assert np.max(np.abs(predicted-charges)) <= 1E-4

