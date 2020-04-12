"""
yes
PACMOF computes partial atomic charges in Metal-Organic Framework using a Random Forest machine learning model trained on a set physically meaningful set of properties that describes the individual atom and its local environment.
"""

# Add imports here
from .pacmof import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
