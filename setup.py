"""
PACMOF computes partial atomic charges in Metal-Organic Framework using a Random Forest machine learning model trained on a set physically meaningful set of properties that describes the individual atom and its local environment.
"""
import sys, os
from setuptools import setup #, find_packages
from setuptools.command.install import install
from setuptools.command.develop import  develop
import versioneer
import subprocess



short_description = __doc__.split("\n")

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        subprocess.call("./pacmof/data/model_generator.py", shell=True)
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        subprocess.call("./pacmof/data/model_generator.py", shell=True)
        # PUT YOUR POST-INSTALL SCRIPT HERE or CAL

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
    package_data={'pacmof': ["data/*.dat","data/*.csv","data/*.py","data/*.cif","data/*.pdb","data/*.md", "tests/*"]},
    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='http://www.my_package.com',  # Website
    install_requires=["numpy>=1.13.3", "pymatgen>=2018.6.11", "joblib>= 0.13.2", "ase>=3.19", "tqdm>=4.15","pandas>=0.20.3","scikit-learn>=0.19.1", "pytest>=5.0.1","dask>=2.2", "dask_jobqueue>=0.6.2", "fsspec>=0.7.4"],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    platforms=['Linux','Mac OS-X','Unix','Windows'],            # Valid platforms your code works on, adjust to your flavor
    python_requires=">=3.7",          # Python version restrictions

    # Manual control if final 
    # package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

)
