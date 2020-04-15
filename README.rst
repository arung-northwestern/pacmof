

Partial Atomic Charges in Metal-Organic Frameworks (PACMOF) from Machine Learning 
**********************************************************************************

PACMOF is a small and easy to use python library that uses machine Learning to quickly estimate partial atomic charges in 
metal-organic frameworks. The pre-trained Random Forest model (Scikit-learn) in PACMOF generates high-quality charges of the same accuracy as that of
Density Derived Electrostatic and Chemical (DDEC), but without needing hours of periodic-DFT calculations. PACMOF is made with high-throughput screening
in mind, where you can get charges on a large number of CIF files in paralle using a Dask_ backend with options to write the output charges into new CIF files and to 
use a user-trained machine learning model instead of the pre-trained one (included). The inspiration for this work came from this recent paper_ published in the Physics Archives, 
where each atom in a MOF is described by a list of the elemental and the local environmental features. However, we chose a similar set of features to train our ML model namely,

- Electronegativity (elemental)
- First ionization energy (elemental)
- Number of atoms in the first coordination shell (local environment).
- The average electronegativity of the first coordination shell (local environment). 
- The average coordination distance of the coordinated atoms (local environment) .
- The average ionization energy of the first coordination shell (local environment).

Among these features, we found that the elemental electronegativity and the average electronegativity of the 
first coordination shell are by far the most important features when it comes to predicting partial charges.

.. image:: ./docs/images/Feature_importance_final.jpg
   :width: 200

About the pre-trained Random Forest model in PACMOF (Coming soon!)
****************************************************


.. figure:: ./docs/images/DDEC_vs_RF_final.jpg
   :width: 200



Installing PACMOF
***********************

PyIsoP will deployed on PyPI_ soon, after which we can install it easily using pip_ 

.. code-block:: bash

    pip install pacmof
    
.. _pip: https://pypi.org/project/pip/
.. _PyPI: https://pypi.org/

..    conda install -c conda-forge pyisop 

.. Tip: Use "--override-channel" option for faster environment resolution.

As of now, please clone it from github_

.. code-block:: bash

    git clone git@github.com:arung-northwestern/pacmof.git
    cd pacmof/
    python setup.py install

.. _github: https://github.com/arung-northwestern/pacmof

What can PACMOF do...?
***********************

PACMOF uses a Dask_ backend to do calculations in parallel which is useful in processing large CIF or for interactive 
high-throughput screening. All the functions return an ASE_ style atoms object (or a list of objects) with the features included under atoms.info['features']
and the charges (if calculated) included under atoms.info['_atom_site_charges'] repectively. Functions are well documented in their docstrings
and can be availed using help(function_name). The rest of the capabilites of PACMOF can be listed as follows:

Serial Calculations
--------------------
- Compute the features for any CIF, this might be useful for training your own machine learning model.

.. code-block:: python

    data = pacmof.get_features_from_cif(path_to_cif)

- Compute the charges on a CIF file in parallel 
This is sufficient for most CIF files where the number of atoms are less than 2000. 

.. code-block:: python

    data = pacmof.get_charges_single_serial(path_to_cif)

Parallel Calculations
----------------------

Since PACMOF uses Dask_, you can run calculations in parallel on a single CPU using mult-threading without starting 
Dask cluster. If you plan of doing high-throughput screening with many CIF files on an HPC, you could start a Dask cluster. before 
calling any of the get_charges_multiple_serial/parallel. For example to start a cluster with 10 processes with 8 CPU's each use:

.. code-block:: python

    from dask_jobqueue import SLURMCluster
    from distributed import Client
    cluster=SLURMCluster(cores=4, interface='ib0', project='p20XXX', queue='short', walltime='04:00:00', memory='100GB')
    cluster.scale(10)
    client= Client(cluster)

Use the documentation on dask.org for more information on the different types of schedulers and more.

- Calculations on a large CIF

For CIFs with more than say 2000 atoms calculations in serial can be too slow, in those cases


    - Compute the features for a large CIF 

    .. code-block:: python

        data = pacmof.get_features_from_cif_parallel(path_to_cif)


    - Compute the charges for a large CIF 

    .. code-block:: python

        data = pacmof.get_charges_single_parallel(path_to_cif, create_cif=False)
    
Please refer to the docstring from help() to see the options on the output CIF file, to use a different machine learning model other than the 
pre-trained one.

- Calculations on a list of CIFs in parallel
PACMOF can be used to run calculations on a list of CIFs in parallel, where each calculation is run in serial or parallel depending on the need.

    - Compute the charges for a list of CIFs in parallel, on a single CPU or using a dask cluster. 

    .. code-block:: python

        data = pacmof.get_charges_multiple_parallel(lsit_of_cifs, create_cif=False)


    - Compute the charge for a list of large CIFs, one by one, where each calculation is run in parallel. Use this only when the CIFs
    have more than 2000 atoms each, if not the memory overhead for parallelizing will make the calculation slower than the serial case.

    .. code-block:: python

        data = pacmof.get_charges_multiple_large(lsit_of_cifs, create_cif=False)


Note: As usual, you could use the serial functions and submit multiple jobs for different CIFs, however the functions above will save
time by not reloading the model for individual CIF files. 


Citing PACMOF  : Coming Soon!
************** 




.. _Scikit-learn:
.. _paper: https://arxiv.org/abs/1905.12098
.. _ASE:
.. _pymatgen:

### Copyright

Copyright (c) 2020, Snurr Research Group, Northwestern University

### Developers

    Srinivasu Kancharlapalli, Fulbright-Nehru Postdoctoral Research Scholar, Snurr Group (2018-2020), Bhabha Atomic Research Centre.

    Arun Gopalan, Ph.D. Scholar, Snurr Group (2015-2020)

#### Acknowledgements
        
    This work is supported by the U.S. Department of Energy, Office of Basic 
    Energy Sciences, Division of Chemical Sciences, Geosciences and 
    Biosciences through the Nanoporous Materials Genome Center under award 
    DE-FG02-17ER16362.


Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.2.
