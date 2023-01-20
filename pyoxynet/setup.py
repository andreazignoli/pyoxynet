import setuptools
from setuptools import setup, find_packages
from setuptools.command.install import install as InstallCommand
import os

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory/"README.md").read_text()

setuptools.setup(
    name="pyoxynet",
    version="0.0.12.4",
    author="Andrea Zignoli",
    author_email="andrea.zignoli@unitn.it",
    description="Python package of the Oxynet project",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=['importlib-resources', 'pandas', 
    'uniplot', 'scipy', 'tensorflow', 'shap', 'chardet'],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={'': ['regressor/*', 'generator/*', 'data_test/*']},
    #exclude_package_data={
    #    '': 'debugging.py.c'},
    python_requires='>=3.8',
)