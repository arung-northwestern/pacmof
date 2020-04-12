"""
Unit and regression test for the pacmof package.
"""

# Import package, test suite, and other packages as needed
import pacmof
import pytest
import sys

def test_pacmof_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "pacmof" in sys.modules
