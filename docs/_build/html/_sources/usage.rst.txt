Usage
=====

Installation
------------

To use pyoxynet, first install it using pip:

.. code-block:: console

   (.venv) $ pip install pyoxynet


Functions
---------

To test the presence of the package you can use the ``pyoxynet.PrintHello()`` function:

.. autofunction:: pyoxynet.PrintHello()

The optimal filter adopted for the project ``pyoxynet.optimal_filter(t, y, my_lambda)`` function:

.. autofunction:: pyoxynet.optimal_filter(t, y, my_lambda)

To test if the TFLite model has been correctly initiated you can use ``pyoxynet.test_tfl_model(interpreter)`` function:

.. autofunction:: pyoxynet.test_tfl_model()