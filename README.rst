

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
**************************************



.. figure:: ./docs/images/DDEC_vs_RF_final.jpg
    :width: 600




Installing PACMOF
***********************

PyIsoP is deployed on PyPI_ , we can install it easily using pip_ 

.. code-block:: bash

    pip install pacmof
    
.. _pip: https://pypi.org/project/pip/
.. _PyPI: https://pypi.org/

..    conda install -c conda-forge pyisop

.. Tip: Use "--override-channel" option for faster environment resolution.

or clone from github_

.. code-block:: bash

    git clone git@github.com:arung-northwestern/pacmof.git
    cd pacmof/
    python setup.py install

.. _github: https://github.com/arung-northwestern/pacmof

What can PACMOF do...?
***********************

Case 1: Using PACMOF to predict partial charges using the (or any) pre-trained model.
-------------------------------------------------------------------------------------
One easy way get all the needed info about pacmof is use python's built-in help() function.

.. code-block::python

    import pacmof as pac 
    help(pacmof)
    
PACMOF includes built-in functions to compute partial charges in a single or for a list of CIFs files using the 
pre-trained Random forest model using just a few lines of code.

.. code-block:: python

    import pacmof  as pac 
    pac.get_charges_single(path_to_cif)



Case 2: Using PACMOF to generate a dataset of features to train your own model 
------------------------------------------------------------------------------




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
