Usage
=====

Installation
------------


To use ``pyoxynet``, first install it using pip:

.. code-block:: console

   (.venv) $ pip install pyoxynet


.. note::

   Starting from version 0.1.5, ``pyoxynet`` offers flexible installation options. You can install the base package, the lightweight TFLite version for inference, or the full version with TensorFlow for training and development.


For TFLite support, the executed command is:

.. code-block:: console

   (.venv) $ pip install "pyoxynet[tflite]" --extra-index-url https://google-coral.github.io/py-repo/

Or for the full TensorFlow version:

.. code-block:: console

   (.venv) $ pip install "pyoxynet[full]" 

Quick Start
-----------

To test the presence of the package you can use the ``PrintHello()`` function:

.. code-block:: python

   import pyoxynet
   pyoxynet.PrintHello()


Basic Workflow
--------------

The typical workflow for using pyoxynet involves three main steps:

1. **Load a model** - Choose between CNN, LSTM, or other available models
2. **Prepare your data** - Load CPET data from CSV or generate synthetic data
3. **Run inference** - Get predictions on exercise intensity domains

Example workflow:

.. code-block:: python

   import pyoxynet

   # Step 1: Load the model
   model = pyoxynet.load_tf_model(n_inputs=5, past_points=40, model='CNN')

   # Step 2: Load your data
   df = pyoxynet.load_csv_data('your_data.csv')

   # Step 3: Run inference
   results = pyoxynet.test_pyoxynet(model, input_df=df, plot=True)


Data Requirements
-----------------

For inference, your CPET data must include the following variables:

**5-input model** (default):

* VO2 - Oxygen uptake
* VCO2 - CO2 output
* VE - Minute ventilation
* PetO2 - End-tidal O2
* PetCO2 - End-tidal CO2

**7-input model** (alternative):

* All of the above plus:
* VEVO2 - Ventilatory equivalent for O2
* VEVCO2 - Ventilatory equivalent for CO2

Data should be sampled at 1-second intervals. For breath-by-breath data, use linear interpolation. For averaged data (5-by-5 or 10-by-10 seconds), cubic interpolation is recommended.


Generating Synthetic Data
--------------------------

Pyoxynet includes a conditional GAN that can generate realistic synthetic CPET data:

.. code-block:: python

   from pyoxynet import *

   # Load the generator
   generator = load_tf_generator()

   # Generate synthetic CPET data
   df = generate_CPET(generator, plot=True)

   # Optionally specify fitness characteristics
   probabilities = create_probabilities(duration=600, VT1=320, VT2=460)
   df = generate_CPET(generator, plot=True, fitness_group=probabilities)


API Reference
-------------

For detailed documentation of all functions and their parameters, see the :doc:`api` page.