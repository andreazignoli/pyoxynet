API reference
=============

utilities
---------

Loading the models, inference, generation and the example tests.

.. automodule:: pyoxynet.utilities
   :members:
   :undoc-members:

testing
-------

The ``Test`` class, which reads some metabolic cart exports into the columns
``test_pyoxynet`` takes.

.. automodule:: pyoxynet.testing
   :members:
   :undoc-members:

model
-----

The networks, defined in Keras for anyone who wants to retrain them. This module needs
TensorFlow (``pip install "pyoxynet[train]"``); inference and generation do not use it.
``Model`` is the architecture of the ``'CNN'`` model, ``LSTMGRUModel`` of the ``'LSTM'`` model,
and ``generator`` of the generator. ``TCN`` was never released as a frozen model.

.. automodule:: pyoxynet.model
   :members:
   :undoc-members:
