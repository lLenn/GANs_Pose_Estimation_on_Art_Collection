import os
from setuptools import setup

if os.path.exists("CycleGAN"):
    setup(
        name='CycleGAN',
        packages=['CycleGAN', 'CycleGAN.data', 'CycleGAN.models', 'CycleGAN.options', 'CycleGAN.util']
    )

if os.path.exists("UGATITLib"):
    setup(
        name='UGATITLib',
        packages=['UGATITLib']
    )