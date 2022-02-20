Usage
=====

Installation
------------

To use pyoxynet, first install it using pip:

.. code-block:: console

   (.venv) $ pip install pyoxynet


Functions
---------

To test the presence of the package you can use the ``utilities.PrintHello()`` function:

.. autofunction:: utilities.PrintHello()

The optimal filter adopted for the project ``utilities.optimal_filter(t, y, my_lambda)`` function:

.. autofunction:: utilities.optimal_filter(t, y, my_lambda)

To test if the TFLite model has been correctly initiated you can use ``utilities.test_tfl_model(interpreter)`` function:

.. autofunction:: utilities.test_tfl_model()

.. autofunction:: utilities.normalize(df)

.. autofunction:: utilities.load_tf_model(n_inputs=7, past_points=40)

.. autofunction:: utilities.load_tf_generator()

.. autofunction:: utilities.pip_install_tflite()

.. autofunction:: utilities.load_csv_data(csv_file='data_test.csv')

.. autofunction:: utilities.test_pyoxynet(input_df=[], n_inputs=7, past_points=40)

.. autofunction:: utilities.create_probabilities(duration=600, VT1=320, VT2=460)

.. autofunction:: utilities.random_walk(length=1, scale_factor=1, variation=1)

.. autofunction:: utilities.generate_CPET(generator, plot=False, fitness_group=None)