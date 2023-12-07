import os
from setuptools import setup

if os.path.exists("DPGN"):
    setup(
        name='DPGN',
        packages=['DPGN']
    )