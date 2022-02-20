Usage
=====

Installation
------------

To use ``pyoxynet``, first install it using pip:

.. code-block:: console

   (.venv) $ pip install pyoxynet

Packages that require addition extra url cannot be installed via *setuptools*, which letely allows and suggests to use ``pip`` when possible. To workaround this problem, TFLite is automatically installed with the following command the first time ``pyoxynet`` is imported:

.. code-block:: console

   pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime


Currently, The TFLite installation process is completed with a line command inside Python, which i know is not the best solution. 

Functions
---------

To test the presence of the package you can use the ``utilities.PrintHello()`` function:

.. autofunction:: utilities.PrintHello()


Every-day functions
-------------------

The optimal filter adopted for the project ``utilities.optimal_filter(t, y, my_lambda)`` function:

.. autofunction:: utilities.optimal_filter(t, y, my_lambda)

.. autofunction:: utilities.normalize(df)

.. autofunction:: utilities.random_walk(length=1, scale_factor=1, variation=1)


CPET data functions
-------------------

.. autofunction:: utilities.load_csv_data(csv_file='data_test.csv')


TFLite model
------------

To test if the TFLite model has been correctly initiated you can use ``utilities.test_tfl_model(interpreter)`` function:

.. autofunction:: utilities.test_tfl_model()

.. autofunction:: utilities.load_tf_model(n_inputs=7, past_points=40)

.. autofunction:: utilities.load_tf_generator()

.. autofunction:: utilities.pip_install_tflite()


Production functions
--------------------

.. autofunction:: utilities.test_pyoxynet(input_df=[], n_inputs=7, past_points=40)

.. autofunction:: utilities.create_probabilities(duration=600, VT1=320, VT2=460)

.. autofunction:: utilities.generate_CPET(generator, plot=False, fitness_group=None)