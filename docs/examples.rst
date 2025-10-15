Examples
========

This section provides practical examples of using pyoxynet for various tasks.

Basic Inference Example
-----------------------

Here's a simple example of how to use pyoxynet for inference on sample data:

.. code-block:: python

   import pyoxynet

   # Load the TFLite model (default: 5 inputs, 40 past points, CNN model)
   tfl_model = pyoxynet.load_tf_model(n_inputs=5, past_points=40, model='CNN')

   # Make inference on random input data
   pyoxynet.test_pyoxynet(tfl_model)

This will generate random CPET data and perform inference to estimate exercise intensity domains.


Generating Synthetic CPET Data
-------------------------------

To generate synthetic CPET data using the conditional GAN:

.. code-block:: python

   from pyoxynet import *

   # Load the generator model
   generator = load_tf_generator()

   # Generate a Pandas DataFrame with synthetic CPET data
   df = generate_CPET(generator, plot=True)

   # Optionally, run inference on the generated data
   tfl_model = load_tf_model(n_inputs=5, past_points=40, model='CNN')
   test_pyoxynet(tfl_model, input_df=df, plot=True)


Working with Your Own Data
---------------------------

To use pyoxynet with your own CPET data:

.. code-block:: python

   import pyoxynet
   import pandas as pd

   # Load your CPET data (must include required columns)
   # Required columns: VO2, VCO2, VE, PetO2, PetCO2
   df = pd.read_csv('your_cpet_data.csv')

   # Load the model
   tfl_model = pyoxynet.load_tf_model(n_inputs=5, past_points=40, model='CNN')

   # Perform inference
   results = pyoxynet.test_pyoxynet(tfl_model, input_df=df, plot=True)


Data Processing with Optimal Filter
------------------------------------

Apply optimal filtering to smooth your CPET data:

.. code-block:: python

   import pyoxynet
   import numpy as np

   # Your time series data
   t = np.arange(0, 600)  # Time in seconds
   y = your_noisy_data     # Your measurement data

   # Apply optimal filter with lambda parameter
   filtered_y = pyoxynet.optimal_filter(t, y, my_lambda=500)


Custom Probability Functions
-----------------------------

Create custom probability functions for generating synthetic data with specific characteristics:

.. code-block:: python

   import pyoxynet

   # Generate probabilities for VT1 and VT2 transitions
   probabilities = pyoxynet.create_probabilities(
       duration=600,  # Test duration in seconds
       VT1=320,      # First ventilatory threshold at 320s
       VT2=460       # Second ventilatory threshold at 460s
   )

   # Use these probabilities with the generator
   generator = pyoxynet.load_tf_generator()
   df = pyoxynet.generate_CPET(generator, plot=True, fitness_group=probabilities)


Model Explainability with SHAP
-------------------------------

Understand model predictions using SHAP values (requires full installation):

.. code-block:: python

   import pyoxynet

   # Load model and explainer
   tfl_model = pyoxynet.load_tf_model(n_inputs=5, past_points=40, model='CNN')
   explainer = pyoxynet.load_explainer(tfl_model)

   # Load or generate data
   df = pyoxynet.load_csv_data('data_test.csv')

   # Compute SHAP values
   shap_values = pyoxynet.compute_shap(
       explainer,
       df,
       n_inputs=5,
       past_points=40,
       shap_stride=20
   )


Advanced Usage: Custom Model Parameters
----------------------------------------

Load models with different input configurations:

.. code-block:: python

   import pyoxynet

   # For 7-input model (including respiratory frequency)
   tfl_model_7 = pyoxynet.load_tf_model(n_inputs=7, past_points=40, model='CNN')

   # For different temporal window (e.g., 60 past time points)
   tfl_model_60 = pyoxynet.load_tf_model(n_inputs=5, past_points=60, model='CNN')

   # Test with your specific configuration
   results = pyoxynet.test_pyoxynet(
       tfl_model_7,
       input_df=your_df,
       n_inputs=7,
       past_points=40,
       plot=True,
       inference_stride=1
   )
