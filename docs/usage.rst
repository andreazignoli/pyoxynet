Usage
=====

Install
-------

.. code-block:: console

   $ pip install pyoxynet

Python 3.10 to 3.14, on Windows, Linux and macOS (Apple silicon). The models run with LiteRT
(``ai-edge-litert``), which pip installs with the package. TensorFlow is not needed.

``pip install "pyoxynet[train]"`` adds TensorFlow, which only :mod:`pyoxynet.model` uses: it
defines the networks in Keras, for anyone who wants to retrain them.

Your own test
-------------

.. code-block:: python

   import pandas as pd
   import pyoxynet

   df = pd.read_csv('my_test.csv')           # your test, with the columns below
   out_df, thresholds = pyoxynet.test_pyoxynet(input_df=df)

   thresholds['VT1']   # {'time': ..., 'HR': ..., 'VE': ..., 'VO2': ...}
   thresholds['VT2']

The columns ``test_pyoxynet`` reads:

=============  ============================================================
Column         Signal
=============  ============================================================
``time``       seconds from the start of the test
``VO2_I``      oxygen uptake
``VCO2_I``     carbon dioxide output
``VE_I``       minute ventilation
``PetO2_I``    end-tidal O2
``PetCO2_I``   end-tidal CO2
``HR_I``       heart rate, optional: it is only reported at the thresholds
=============  ============================================================

* Any sampling works, breath by breath included. ``test_pyoxynet`` averages the data into
  one-second bins and interpolates linearly between them.
* Each signal is min-max scaled over the test before it reaches the model, so units do not
  change the domains. The VO2, VE and HR reported at each threshold are in your file's units.
* ``out_df`` holds the three domain outputs (``p_md``, ``p_hv``, ``p_sv``) over time, next to
  the signals. ``plot=True`` draws them in the terminal.
* Other columns, text included, are ignored.
* ``test_pyoxynet`` raises a ``ValueError`` that says why when it cannot place consistent
  thresholds: the outputs never cross, VT1 would fall after VT2 or in the first 40 s, a signal is
  missing or constant, or the test is shorter than 80 s.

From a cart export
------------------

``test_pyoxynet`` reads a table, not a vendor file. To get from an export to that table, read it
with pandas, rename its columns, and turn its time into seconds:

.. code-block:: python

   import pandas as pd
   import pyoxynet

   # Your export, read with pd.read_excel or pd.read_csv. Here one is made from the bundled
   # example, with a cart's own column names and time as mm:ss.
   example = pyoxynet.load_csv_data()
   raw = pd.DataFrame({
       't': [f'{int(s) // 60}:{int(s) % 60:02d}' for s in example['time']],
       "V'O2": example['VO2_I'], "V'CO2": example['VCO2_I'], "V'E": example['VE_I'],
       'PETO2': example['PetO2_I'], 'PETCO2': example['PetCO2_I'], 'HR': example['HR_I'],
   })

   df = pd.DataFrame({
       'time': raw['t'].map(pyoxynet.get_sec),      # 'mm:ss' or 'h:mm:ss' to seconds
       'VO2_I': raw["V'O2"], 'VCO2_I': raw["V'CO2"], 'VE_I': raw["V'E"],
       'PetO2_I': raw['PETO2'], 'PetCO2_I': raw['PETCO2'], 'HR_I': raw['HR'],
   })
   out_df, thresholds = pyoxynet.test_pyoxynet(input_df=df)

* A units row under the header, or cells such as ``---``, can stay: rows without a time are
  dropped and gaps in a signal are interpolated.
* Units do not matter for the domains, since each signal is scaled over the test, but the VO2,
  VE and HR reported at the thresholds are in your file's units.
* The hosted Oxynet API (`app.oxynet.net <https://app.oxynet.net>`_) reads vendor exports
  directly, with no renaming, COSMED and Cortex among them.

The models
----------

======================  ==============================================  ==========================================
``model=``              What it is                                      Paper
======================  ==============================================  ==========================================
``'CNN'``, the default  Oxynet's convolutional network, fine-tuned on   Evaluated in Keir et al. (2026)
                        ramp tests from the Murias lab
``'LSTM'``              A recurrent network: one LSTM and one GRU       Follows the recurrent approach of Zignoli
                        layer                                           et al. (2019); not evaluated on its own
======================  ==============================================  ==========================================

.. code-block:: python

   import pyoxynet

   df = pyoxynet.load_csv_data()             # or your own test
   out_df, thresholds = pyoxynet.test_pyoxynet(input_df=df, model='LSTM')

   # or load once and reuse, from several threads if you like
   model = pyoxynet.load_model('LSTM')
   out_df, thresholds = pyoxynet.test_pyoxynet(tf_model=model, input_df=df)

   # A loaded model also runs on its own, on windows scaled as test_pyoxynet scales them:
   # one of shape (40, 5), or a batch of shape (n, 40, 5)
   import numpy as np
   windows = np.zeros((3, 40, 5))
   outputs = model(windows)                  # shape (3, 3)

* Both are research models, frozen as they are: they are not retrained.
* Both take 40 one-second samples of VO2, VCO2, VE, PetO2 and PetCO2 and give three outputs.
* The two can disagree by tens of seconds on the same test. On the bundled example the CNN
  places VT1 and VT2 at 372 s and 550 s, the LSTM at 448 s and 628 s. The CNN is the one a paper
  evaluated.
* The evaluation in Keir et al. covered incremental tests in healthy individuals.
* The domain outputs are what the network gives, not calibrated probabilities, and pyoxynet
  reports no confidence score.

Where each file came from is recorded in
`models/README.md <https://github.com/andreazignoli/pyoxynet/blob/master/pyoxynet/pyoxynet/models/README.md>`_.

Upgrading from 0.1.x
--------------------

* ``load_tf_model`` and ``load_tf_generator`` still work. They return the same objects as
  ``load_model`` and ``load_generator``, whose outputs also answer ``.numpy()``.
* **The default model is a different network.** In 0.1.12, ``'CNN'`` loaded ``regressor``; in
  1.0 it is the CNN evaluated in Keir et al. (2026). On the same file, 1.0 gives different
  thresholds than 0.1.12.
* ``murias_lab``, ``AIS``, ``NSWIS``, ``TCN``, ``LSTMGRUModel`` and ``transformer`` were
  removed. Asking for one raises a ``ValueError`` that names the available models. The
  earlier releases remain on PyPI.
* ``load_explainer`` and ``compute_shap`` were removed. They expected a 6-input Keras model,
  which no model shipped since the move to 5 inputs has been.
* The ``[tflite]`` and ``[full]`` extras are gone: the base install runs everything.
* Since 1.0.2, ``generate_CPET`` and ``draw_real_test`` return their labels as numbers, not
  strings, and print nothing unless ``verbose=True``.

Citing
------

If you use the CNN, cite:

   Keir DA, Zignoli A, Iannetta D, Mattioni Maturana F, Murias JM. AI-Driven Analysis of
   Cardiopulmonary Exercise Tests to Identify Gas Exchange and Ventilatory Thresholds. *Sports
   Medicine*. 2026;56(7):1833-1843.
   `doi:10.1007/s40279-026-02403-w <https://doi.org/10.1007/s40279-026-02403-w>`_

Background:

* Zignoli A, et al. Expert-level classification of ventilatory thresholds from cardiopulmonary
  exercising test data with recurrent neural networks. *European Journal of Sport Science*.
  2019;19(9):1221-1229. `doi:10.1080/17461391.2019.1587523 <https://doi.org/10.1080/17461391.2019.1587523>`_
* Zignoli A, et al. Oxynet: A collective intelligence that detects ventilatory thresholds in
  cardiopulmonary exercise tests. *European Journal of Sport Science*. 2022;22(3):425-435.
  `doi:10.1080/17461391.2020.1866081 <https://doi.org/10.1080/17461391.2020.1866081>`_
* Zignoli A. Machine Learning Models for the Automatic Detection of Exercise Thresholds in
  Cardiopulmonary Exercising Tests: From Regression to Generation to Explanation. *Sensors*.
  2023;23(2):826. `doi:10.3390/s23020826 <https://doi.org/10.3390/s23020826>`_
