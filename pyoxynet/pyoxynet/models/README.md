# The pyoxynet models

Three frozen LiteRT (`.tflite`) files. They are run with `ai-edge-litert`, so neither inference
nor generation needs TensorFlow. They are not retrained: a different model would be a new file
and a new major version of the package. The hosted Oxynet API at app.oxynet.net runs its own,
separately trained models, not these.

| File | Loaded by | Input | Output |
|---|---|---|---|
| `cnn.tflite` | `load_model('CNN')`, the default | (1, 40, 5) | (1, 3) |
| `lstm.tflite` | `load_model('LSTM')` | (1, 40, 5) | (1, 3) |
| `generator.tflite` | `load_generator()` | (1, 23) | (1, 40, 7) |

The two threshold models take the same input: 40 one-second samples of VO2, VCO2, VE, PetO2 and
PetCO2, each min-max normalised over the test (`test_pyoxynet` does this). The three outputs are
the moderate, heavy and severe domains.

## Where each one came from

**`cnn.tflite`**: the convolutional network evaluated in Keir, Zignoli, Iannetta, Mattioni
Maturana and Murias, "AI-Driven Analysis of Cardiopulmonary Exercise Tests to Identify Gas
Exchange and Ventilatory Thresholds", *Sports Medicine* (2026),
[doi:10.1007/s40279-026-02403-w](https://doi.org/10.1007/s40279-026-02403-w). Oxynet's CNN,
fine-tuned on ramp tests from the Murias lab. Committed to this repository as
`app/tf_lite_models/tfl_model.tflite` on 2024-02-22 (`7543769`) and last updated on 2024-03-04
(`3bf4d0a`), the version copied here byte for byte.

**`lstm.tflite`**: a recurrent network, one LSTM and one GRU layer of 32 units each, global
average pooling and a softmax over the three domains. Trained in November 2023 and committed as
`pyoxynet/LSTMGRUModel` in `6e4b08d` ("LSTMGRU regressor one hot encoded", 2023-11-29). It
continues the recurrent approach of Zignoli et al., "Expert-level classification of ventilatory
thresholds from cardiopulmonary exercising test data with recurrent neural networks", *European
Journal of Sport Science* (2019),
[doi:10.1080/17461391.2019.1587523](https://doi.org/10.1080/17461391.2019.1587523), but it is
not that paper's network, whose weights were never part of this package. No paper evaluates
this model.

**`generator.tflite`**: the conditional GAN generator: a dense layer, a transposed 1D
convolution and a 1D convolution with tanh output. Its input is 20 noise values followed by
three domain probabilities; its output is a 40-second window of VO2, VCO2, VE, HR, RF, PetO2 and
PetCO2, scaled to [-1, 1]. The version shipped in 0.1.12 (`pyoxynet/generator`, `b07a116`,
2025-10-15).

## How the files were made

The CNN was already a `.tflite` file and was not converted. The other two were Keras 2
SavedModels, converted with TensorFlow 2.15.1, the version that reads them:

- each was rebuilt from the classes in `model.py` and given the saved weights, as
  `load_tf_model` and `load_tf_generator` did up to 0.1.12;
- the LSTM was rebuilt with its two recurrent layers unrolled over the 40 steps, because the
  converter's fused LSTM kernel gave outputs up to 0.58 away from Keras;
- over 200 random inputs, the converted LSTM is within 2.4e-7 of the Keras model and the
  generator within 5.4e-7.

All three files use built-in LiteRT operations only. `tests/test_models.py` checks that the
thresholds on the bundled example tests match those 0.1.12's own code gives with the same
weights.

| File | SHA-256 |
|---|---|
| `cnn.tflite` | `6e867a4031a8759b42553e195ae84a7497af3874c07f76c6dd53cb4865d58bc8` |
| `lstm.tflite` | `396954dd9e01c005f747a537b8dfbb6c196c30f35a1511d167d75c4a04ad9b1a` |
| `generator.tflite` | `b69afc79c4a955d5e3d6dfa2533c74515c651bef6ac812862add9304da0a78f0` |

## Removed in 1.0.0

`regressor` (the CNN `load_tf_model` returned by default up to 0.1.12), `murias_lab`, `AIS`,
`NSWIS`, `TCN`, `LSTMGRUModel` and `transformer`. They remain in this repository's history and
in the earlier releases on PyPI. Asking for one by name raises an error that lists the models
available now.
