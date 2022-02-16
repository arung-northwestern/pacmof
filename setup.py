"""
PACMOF computes partial atomic charges in Metal-Organic Framework using a Random Forest machine learning model trained on a set physically meaningful set of properties that describes the individual atom and its local environment.
"""
import sys, os
from setuptools import setup #, find_packages
from setuptools.command.install import install
from setuptools.command.develop import  develop
import versioneer
import subprocess
# from icecream import ic
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


short_description = __doc__.split("\n")

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])

path_to_dir = os.path.dirname(os.path.abspath(__file__))
path_to_model_generator = os.path.join(path_to_dir, "pacmof", "data", "model_generator.py")
# ic.configureOutput(includeContext=True)
# ic(path_to_model_generator)
class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        # subprocess.call(path_to_model_generator)
        # path_to_folder = os.path.dirname(os.path.abspath(__file__))
        path_to_ddec_data = os.path.join(path_to_dir,'pacmof','data', 'data_DDEC.csv')
        path_to_cm5_data = os.path.join(path_to_dir,'pacmof','data', 'data_CM5.csv')
#         # ic.configureOutput(includeContext=True)
#         # ic(path_to_cm5_data)
#         # ic(path_to_ddec_data)
#         # ic(os.path.exists(path_to_ddec_data))
        print("Please wait, compiling the RandomForestRegressor model...\n")
        print("This might take a few minutes...")
        df_DDEC = pd.read_csv(path_to_ddec_data)
        df_CM5 = pd.read_csv(path_to_cm5_data)

        X_DDEC = df_DDEC[['EN', 'IP', 'coord_no', 'coord_dis', 'EN_coord_1', 'IP_coord_1', 'EN_coord_2']]
        X_CM5 = df_CM5[['EN', 'IP', 'coord_no', 'coord_dis', 'EN_coord_1', 'IP_coord_1', 'EN_coord_2']]
        y_DDEC = df_DDEC['DDEC']
        y_CM5 = df_CM5['CM5']
        X_DDEC_train, X_DDEC_test, y_DDEC_train, y_DDEC_test = train_test_split(X_DDEC, y_DDEC, test_size=0.20,
                                                                                random_state=0)
        X_CM5_train, X_CM5_test, y_CM5_train, y_CM5_test = train_test_split(X_CM5, y_CM5, test_size=0.20,
                                                                            random_state=0)
        regressor = RandomForestRegressor(bootstrap=False, max_depth=20, max_features=3, min_samples_leaf=1,
                                          min_samples_split=2, n_estimators=500, verbose=2, n_jobs=-1)
        print("Training the model based on DDEC charges...")
        regressor.fit(X_DDEC_train, y_DDEC_train)
        path_to_ddec_pkl = os.path.join(path_to_dir,'pacmof','data', 'Model_RF_DDEC.pkl')
        joblib.dump(regressor, path_to_ddec_pkl, compress=3)
        print("Training the model based on CM5 charges...")
        path_to_cm5_pkl = os.path.join(path_to_dir,'pacmof','data', 'Model_RF_CM5.pkl')
        regressor.fit(X_CM5_train, y_CM5_train)
        joblib.dump(regressor, path_to_cm5_pkl, compress=3)

        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        # subprocess.call(path_to_model_generator)
        # PUT YOUR POST-INSTALL SCRIPT HERE or CAL
        path_to_ddec_data = os.path.join(path_to_dir, 'pacmof', 'data', 'data_DDEC.csv')
        path_to_cm5_data = os.path.join(path_to_dir, 'pacmof', 'data', 'data_CM5.csv')
#         # ic.configureOutput(includeContext=True)
#         # ic(path_to_cm5_data)
#         # ic(path_to_ddec_data)
#         # ic(os.path.exists(path_to_ddec_data))
        print("Please wait, compiling the RandomForestRegressor model...\n")
        print("This might take a few minutes...")
        df_DDEC = pd.read_csv(path_to_ddec_data)
        df_CM5 = pd.read_csv(path_to_cm5_data)

        X_DDEC = df_DDEC[['EN', 'IP', 'coord_no', 'coord_dis', 'EN_coord_1', 'IP_coord_1', 'EN_coord_2']]
        X_CM5 = df_CM5[['EN', 'IP', 'coord_no', 'coord_dis', 'EN_coord_1', 'IP_coord_1', 'EN_coord_2']]
        y_DDEC = df_DDEC['DDEC']
        y_CM5 = df_CM5['CM5']
        X_DDEC_train, X_DDEC_test, y_DDEC_train, y_DDEC_test = train_test_split(X_DDEC, y_DDEC, test_size=0.20,
                                                                                random_state=0)
        X_CM5_train, X_CM5_test, y_CM5_train, y_CM5_test = train_test_split(X_CM5, y_CM5, test_size=0.20,
                                                                            random_state=0)
        regressor = RandomForestRegressor(bootstrap=False, max_depth=20, max_features=3, min_samples_leaf=1,
                                          min_samples_split=2, n_estimators=500, verbose=2, n_jobs=-1)
        print("Training the model based on DDEC charges...")
        regressor.fit(X_DDEC_train, y_DDEC_train)
        path_to_ddec_pkl = os.path.join(path_to_dir, 'pacmof', 'data', 'Model_RF_DDEC.pkl')
        joblib.dump(regressor, path_to_ddec_pkl, compress=3)
        print("Training the model based on CM5 charges...")
        path_to_cm5_pkl = os.path.join(path_to_dir, 'pacmof', 'data', 'Model_RF_CM5.pkl')
        regressor.fit(X_CM5_train, y_CM5_train)
        joblib.dump(regressor, path_to_cm5_pkl, compress=3)


# ic(versioneer.get_cmdclass())
setup(
    # Self-descriptive entries which should always be present
    name='pacmof',
    author='Snurr Research Group, Northwestern University',
    author_email='arungopalan2020@u.northwestern.edu',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(cmdclass={'develop': PostDevelopCommand,'install': PostInstallCommand}), # This line adds the post install and post develop model sklearn compilation
    license='BSD-3-Clause',

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=['pacmof', 'pacmof.tests'],

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,

    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,
    package_data={'pacmof': ["data/*.dat","data/*.csv","data/*.py","data/*.cif","data/*.pdb","data/*.md", "tests/*", "data/*.pkl"]},
    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='http://www.my_package.com',  # Website
    install_requires=["numpy>=1.13.3", "pymatgen>=2018.6.11", "joblib>= 0.13.2", "ase>=3.19", "tqdm>=4.15","pandas>=0.20.3","scikit-learn>=0.19.0", "pytest>=5.0.1","dask>=2.2", "dask_jobqueue>=0.6.2", "fsspec>=0.7.4"],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    platforms=['Linux','Mac OS-X','Unix','Windows'],            # Valid platforms your code works on, adjust to your flavor
    python_requires=">=3.7",          # Python version restrictions

    # Manual control if final 
    # package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

)
