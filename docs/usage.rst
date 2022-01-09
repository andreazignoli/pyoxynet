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