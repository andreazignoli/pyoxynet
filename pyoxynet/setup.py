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
    version="0.1.0",
    author="Andrea Zignoli",
    author_email="andrea.zignoli@unitn.it",
    description="AI-powered CPET analysis for exercise physiology research - lightweight and full installations available",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=['importlib-resources', 'pandas', 
    'uniplot', 'scipy', 'shap', 'chardet', 
    'xlrd', 'openpyxl', 'matplotlib'],
    extras_require={
        "full": ["tensorflow"],  # Full TensorFlow for advanced features
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={'': ['regressor/*', 
                       'generator/*', 
                       'data_test/*', 
                       'murias_lab/*', 
                       'TCN/*', 
                       'LSTMGRUModel/*', 
                       'AIS/*']},
    #exclude_package_data={
    #    '': 'debugging.py.c'},
    python_requires='>=3.8,<3.12',
)