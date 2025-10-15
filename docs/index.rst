Pyoxynet Documentation
======================

**Pyoxynet** is a Python package for automatic interpretation of cardiopulmonary exercise test (CPET) data using deep learning models. It is part of the Oxynet project, which aims to provide universal access to quality healthcare through AI-powered diagnostic tools.

.. image:: https://img.shields.io/pypi/v/pyoxynet.svg
   :target: https://pypi.org/project/pyoxynet/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/pyoxynet.svg
   :target: https://pypi.org/project/pyoxynet/
   :alt: Python versions

Key Features
------------

* **Inference Model**: Automatically estimates exercise intensity domains from CPET data
* **Generator Model**: Creates synthetic CPET data using conditional GANs
* **Flexible Deployment**: Choose between lightweight TFLite or full TensorFlow
* **Model Explainability**: SHAP integration for understanding predictions
* **Easy to Use**: Simple API for both beginners and advanced users

Quick Example
-------------

.. code-block:: python

   import pyoxynet

   # Load the TFLite model
   tfl_model = pyoxynet.load_tf_model(n_inputs=5, past_points=40, model='CNN')

   # Make inference on sample data
   pyoxynet.test_pyoxynet(tfl_model)

Installation
------------

Install the base package:

.. code-block:: bash

   pip install pyoxynet

For TFLite inference support:

.. code-block:: bash

   pip install "pyoxynet[tflite]" --extra-index-url https://google-coral.github.io/py-repo/

For full TensorFlow with training capabilities:

.. code-block:: bash

   pip install "pyoxynet[full]"

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   GitHub Repository <https://github.com/andreazignoli/pyoxynet>
   Web Application <https://pyoxynet-lite-app-b415901c79ab.herokuapp.com/>
   Oxynet Website <http://oxynet.net>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
