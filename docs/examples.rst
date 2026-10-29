Examples
========

The bundled example test
------------------------

.. code-block:: python

   import pyoxynet

   out_df, thresholds = pyoxynet.test_pyoxynet(plot=True)

Without ``input_df``, ``test_pyoxynet`` runs the default model on the example test bundled with
the package. ``plot=True`` draws the three domain outputs in the terminal.

Comparing the two models
------------------------

.. code-block:: python

   import pyoxynet

   df = pyoxynet.load_csv_data()            # the bundled example; or load_csv_data('my_test.csv')
   for name in ('CNN', 'LSTM'):
       _, thresholds = pyoxynet.test_pyoxynet(input_df=df, model=name)
       print(name, thresholds['VT1']['time'], thresholds['VT2']['time'])

Synthetic tests
---------------

pyoxynet includes a conditional generative adversarial network (GAN) that generates synthetic
ramp tests.

.. code-block:: python

   import pyoxynet

   generator = pyoxynet.load_generator()
   df, info = pyoxynet.generate_CPET(generator, fitness_group=2)

   print(info['VT1'], info['VT2'])          # the thresholds it was generated with, in seconds
   out_df, thresholds = pyoxynet.test_pyoxynet(input_df=df)

* ``fitness_group`` goes from ``0`` (very low) to ``3`` (high); leave it out for a random level.
* A generated test has VO2, VCO2, VE, HR, respiratory frequency, PetO2 and PetCO2.
  ``info['data']`` holds a breath-by-breath version built from it.
* Only ramp tests that start under load are generated: ``resting=True`` raises
  ``NotImplementedError``.

The example tests
-----------------

The package includes 138 anonymised tests (88 ramp tests and 50 with a resting phase), each
labelled with its thresholds.

.. code-block:: python

   import pyoxynet

   df, info = pyoxynet.draw_real_test()     # a random one
   print(info['VT1'], info['VT2'])          # the labelled thresholds, in seconds
   out_df, thresholds = pyoxynet.test_pyoxynet(input_df=df)

   df, info = pyoxynet.draw_real_test(resting=False, index=12)   # ramp test 12

``draw_real_test`` adds a little random noise to the signals of the test it draws.

Smoothing a signal
------------------

``optimal_filter`` is the smoothing filter pyoxynet applies to the model outputs and to VO2.

.. code-block:: python

   import numpy as np
   import pyoxynet

   t = np.arange(0, 600, dtype=float)       # time in seconds
   y = np.sin(t / 60) + np.random.normal(0, 0.2, t.size)
   smooth = pyoxynet.optimal_filter(t, y, 500)

A larger last argument smooths more.
