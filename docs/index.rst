pyoxynet
========

The open-source models from the Oxynet research papers, for cardiopulmonary exercise test (CPET)
data. Give pyoxynet a test and it returns, second by second, the model's output for the moderate,
heavy and severe exercise-intensity domains, and the two ventilatory thresholds (VT1 and VT2)
where those outputs cross. It can also generate synthetic tests.

Everything runs on your machine. The models are frozen LiteRT files, so TensorFlow is not needed.

.. image:: https://img.shields.io/pypi/v/pyoxynet.svg
   :target: https://pypi.org/project/pyoxynet/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/pyoxynet.svg
   :target: https://pypi.org/project/pyoxynet/
   :alt: Python versions

Install
-------

.. code-block:: bash

   pip install pyoxynet

Python 3.10 to 3.14, on Windows, macOS and Linux.

Quick start
-----------

.. code-block:: python

   import pyoxynet

   # The bundled example test, through the default model
   out_df, thresholds = pyoxynet.test_pyoxynet()

   print(thresholds['VT1']['time'], round(thresholds['VT1']['VO2']))  # 372 2979
   print(thresholds['VT2']['time'], round(thresholds['VT2']['VO2']))  # 550 3656

The hosted Oxynet API
---------------------

pyoxynet is the open research package. Oxynet itself is a hosted service at
`app.oxynet.net <https://app.oxynet.net>`_, with its own, separately trained models. It reads
metabolic cart exports as they come off the cart (COSMED and Cortex among them), checks the
signals, and returns ventilatory thresholds, substrate use and signal-quality checks as versioned
JSON. API keys are free: `app.oxynet.net/signup <https://app.oxynet.net/signup>`_. See the
`developer docs <https://www.oxynet.net/developers>`_. AI agents start at
`app.oxynet.net/llms.txt <https://app.oxynet.net/llms.txt>`_.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User guide

   usage
   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Links

   GitHub repository <https://github.com/andreazignoli/pyoxynet>
   GitHub Discussions <https://github.com/andreazignoli/pyoxynet/discussions>
   Oxynet <https://www.oxynet.net>

pyoxynet is research software, not a medical device: it does not diagnose, and its results are no
substitute for a clinician's judgement.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
