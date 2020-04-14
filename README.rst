

Partial Atomic Charges in Metal-Organic Frameworks (PACMOF) from Machine Learning 
**********************************************************************************

PACMOF is a small and easy to use python library that uses machine Learning to quickly estimate the partial atomic charges in 
metal-organic framework. The pre-trained Random Forest model (Scikit-learn) in PACMOF generates charges of the a same accuracy as the
Density Derived Electrostatic and Chemical (DDEC) but without requiring hours of periodic-DFT calculations. PACMOF is made with high-throughput screening
in mind, where you can get high-quality charges with a large number of CIF files as input and generate output CIF files with the '_atom_site_charge' property added 
The inspiration for this work came from this recent paper_ published in the Physics Archives where a vector of the elemental and the local environmental properties
are used to describe every atom in a MOF. However, we chose a much simpler (yet sufficient) set of features for each atom namely,

- Electronegativity (elemental)
- First ionization energy (elemental)
- Number of atoms in the first coordination shell (local environment).
- The average electronegativity of the first coordination shell (local environment). 
- The average coordination distance of the coordinated atoms (local environment) .
- The average ionization energy of the first coordination shell (local environment).

Among these features, we found that the elemental electronegativity and the average electronegativity of the 
first coordination shell are by far the most important features when it comes to predicting partial charges.

.. figure:: ./docs/images/Feature_importance_final.jpg
    :width: 600

About the pre-trained Random Forest model in PACMOF
****************************************************



.. figure:: ./docs/images/DDEC_vs_RF_final.jpg
    :width: 600




Installing PACMOF
***********************

PyIsoP will deployed on PyPI_ soon, then we can install it easily using pip_ 

.. code-block:: bash

    pip install pacmof
    
.. _pip: https://pypi.org/project/pip/
.. _PyPI: https://pypi.org/

..    conda install -c conda-forge pyisop 

.. Tip: Use "--override-channel" option for faster environment resolution.

Currently, please clone from github_

.. code-block:: bash

    git clone git@github.com:arung-northwestern/pacmof.git
    cd pacmof/
    python setup.py install

.. _github: https://github.com/arung-northwestern/pacmof

What can PACMOF do...?
***********************

Each of these functions return an ASE atoms object where the features for machine learning are updated under data.info['features'] 
and the charges are updated under data.info['_atom_site_charges']. One easy way get the info on all the pacmof function arguments 
is to use python's built-in help(function_name) utility.

.. code-block::python

    import pacmof, glob

    files = glob.glob('*.cif') # Get a list of CIF files
    
    # 1. To compute the partial charges on one material. 
    data = pacmof.get_charges_single(files[0], create_cif=True, path_to_output_dir='.', add_string='_charged', use_default_model=True)

    # 2. To compute the partial charges on a list of CIFs but on a single CPU on by one 
    # (not recommended  for high-throughput applications).
    data = pacmof.get_charges_multiple_serial(files, create_cif=True, path_to_output_dir='.', add_string='_charged', use_default_model=True)

    # 3. To compute the partial charges on a the list of CIFs (Dask automatically chooses between threading (1 CPU) or multi-processing (on HPC)). 
    # Recommended for high-throughput screening applications.
    data = pacmof.get_charges_multiple_parallel(files, create_cif=True, path_to_output_dir='.', add_string='_charged', use_default_model=True)

    # Addendum : To use PACMOF on an HPC start a Dask cluster before you call the get_charges_multiple_parallel function from 3.
    # To start a cluster use (more info for different schedulers other that SLURM can be found on dask.org website). 
    from dask_jobqueue import SLURMCluster
    from distributed import Client
    cluster=SLURMCluster(cores=4, interface='ib0', project='p20XXX', queue='short', walltime='04:00:00', memory='100GB')
    cluster.scale(10)
    client= Client(cluster)


    # 4. To get the just features without loading the pre-trained machine larning model or predicting charges
    # This could be useful for training your own machine larning model.
    data = pacmof.get_features_from_cif(files[0])

    # Note: To use a different machine learning model, persist it in a pickle file (.pkl) and use the path_to_pickle_obj argument with 'use_default_model' argument set to False.



Citing PACMOF  : Coming Soon!
************** 




.. _Scikit-learn:
.. _paper: https://arxiv.org/abs/1905.12098
.. _ASE:
.. _pymatgen:

### Copyright

Copyright (c) 2020, Snurr Research Group, Northwestern University

### Developers

    Srinivasu Kancharlapalli, Visiting Scholar Snurr Group (2018-2020), Bhaba Atomic Research Center.

    Arun Gopalan, Ph.D. Scholar, Snurr Group (2015-2020)

#### Acknowledgements
        
    This work is supported by the U.S. Department of Energy, Office of Basic 
    Energy Sciences, Division of Chemical Sciences, Geosciences and 
    Biosciences through the Nanoporous Materials Genome Center under award 
    DE-FG02-17ER16362.


Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.2.
