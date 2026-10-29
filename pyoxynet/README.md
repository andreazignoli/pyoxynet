# pyoxynet

[![PyPI version](https://img.shields.io/pypi/v/pyoxynet.svg)](https://pypi.org/project/pyoxynet/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyoxynet.svg)](https://pypi.org/project/pyoxynet/)
[![Documentation](https://readthedocs.org/projects/pyoxynet/badge/?version=latest)](https://pyoxynet.readthedocs.io/en/latest/)

The open-source models from the Oxynet research papers, for cardiopulmonary exercise test (CPET)
data. Give pyoxynet a test and it returns, second by second, the model's output for the moderate,
heavy and severe exercise-intensity domains, and the two ventilatory thresholds (VT1 and VT2)
where those outputs cross. It can also generate synthetic tests.

Everything runs on your machine. The models are frozen LiteRT files, so TensorFlow is not needed.

## Install

```sh
pip install pyoxynet
```

Python 3.10 to 3.14, on Windows, Linux and macOS (Apple silicon).

## Quick start

```python
import pyoxynet

# The bundled example test, through the default model
out_df, thresholds = pyoxynet.test_pyoxynet()

print(thresholds['VT1']['time'], round(thresholds['VT1']['VO2']))  # 372 2979
print(thresholds['VT2']['time'], round(thresholds['VT2']['VO2']))  # 550 3656
```

Each threshold gives its `time` (s) and the `VO2`, `VE` and `HR` there. `out_df` holds the three
domain outputs (`p_md`, `p_hv`, `p_sv`) over time, next to the signals.

## Your own test

```python
import pandas as pd
import pyoxynet

df = pd.read_csv('my_test.csv')
out_df, thresholds = pyoxynet.test_pyoxynet(input_df=df)
```

The columns `test_pyoxynet` reads:

| Column | Signal |
|---|---|
| `time` | seconds from the start of the test |
| `VO2_I` | oxygen uptake |
| `VCO2_I` | carbon dioxide output |
| `VE_I` | minute ventilation |
| `PetO2_I` | end-tidal O2 |
| `PetCO2_I` | end-tidal CO2 |
| `HR_I` | heart rate, optional: it is only reported at the thresholds |

- Any sampling works, breath by breath included. `test_pyoxynet` averages the data into
  one-second bins and interpolates linearly between them.
- Each signal is min-max scaled over the test before it reaches the model, so units do not change
  the domains. The VO2, VE and HR reported at each threshold are in your file's units.
- When the outputs never cross, no threshold can be placed and `test_pyoxynet` raises a
  `ValueError`.

## The models

| `model=` | What it is | Paper |
|---|---|---|
| `'CNN'`, the default | Oxynet's convolutional network, fine-tuned on ramp tests from the Murias lab | Evaluated in Keir et al. (2026) |
| `'LSTM'` | A recurrent network: one LSTM and one GRU layer | Follows the recurrent approach of Zignoli et al. (2019); not evaluated on its own |

```python
out_df, thresholds = pyoxynet.test_pyoxynet(input_df=df, model='LSTM')
```

- Both are research models, frozen as they are: they are not retrained.
- The evaluation in Keir et al. covered incremental tests in healthy individuals.
- The domain outputs are what the network gives, not calibrated probabilities, and pyoxynet
  reports no confidence score.

Where each file came from is recorded in
[models/README.md](https://github.com/andreazignoli/pyoxynet/blob/master/pyoxynet/pyoxynet/models/README.md).
The models offered before 1.0.0 (`murias_lab`, `AIS`, `NSWIS`, `TCN`, `LSTMGRUModel`) were removed.
Asking for one raises an error that names the available ones, and the earlier releases remain on
PyPI.

## Synthetic and example tests

```python
generator = pyoxynet.load_generator()
df, info = pyoxynet.generate_CPET(generator)        # a synthetic ramp test
out_df, thresholds = pyoxynet.test_pyoxynet(input_df=df)

df, info = pyoxynet.draw_real_test()                # one of 138 anonymised example tests
```

- `info['VT1']` and `info['VT2']` are the thresholds the test was generated or labelled with, in
  seconds.
- `generate_CPET(generator, fitness_group=0)` to `3` sets the fitness level, from very low to
  high.

## The Oxynet API

pyoxynet is the open research package. Oxynet itself is a hosted service at
[app.oxynet.net](https://app.oxynet.net), with its own, separately trained models. It reads
metabolic cart exports as they come off the cart (COSMED and Cortex among them), checks the
signals, and returns ventilatory thresholds, substrate use and signal-quality checks as versioned
JSON. API keys are free: [app.oxynet.net/signup](https://app.oxynet.net/signup).

- REST API: [developer docs](https://www.oxynet.net/developers) and the
  [OpenAPI schema](https://app.oxynet.net/v1/openapi.json)
- AI assistants: the remote MCP server at `https://app.oxynet.net/oxynet-mcp`
- AI agents: start at [app.oxynet.net/llms.txt](https://app.oxynet.net/llms.txt), and upload the
  vendor file unchanged

## Questions and citing

Questions and ideas go to [GitHub Discussions](https://github.com/andreazignoli/pyoxynet/discussions),
bugs to [issues](https://github.com/andreazignoli/pyoxynet/issues).

If you use the CNN, cite:

> Keir DA, Zignoli A, Iannetta D, Mattioni Maturana F, Murias JM. AI-Driven Analysis of
> Cardiopulmonary Exercise Tests to Identify Gas Exchange and Ventilatory Thresholds. *Sports
> Medicine*. 2026;56(7):1833-1843.
> [doi:10.1007/s40279-026-02403-w](https://doi.org/10.1007/s40279-026-02403-w)

Background:

- Zignoli A, et al. Expert-level classification of ventilatory thresholds from cardiopulmonary
  exercising test data with recurrent neural networks. *European Journal of Sport Science*.
  2019;19(9):1221-1229. [doi:10.1080/17461391.2019.1587523](https://doi.org/10.1080/17461391.2019.1587523)
- Zignoli A, et al. Oxynet: A collective intelligence that detects ventilatory thresholds in
  cardiopulmonary exercise tests. *European Journal of Sport Science*. 2022;22(3):425-435.
  [doi:10.1080/17461391.2020.1866081](https://doi.org/10.1080/17461391.2020.1866081)
- Zignoli A. Machine Learning Models for the Automatic Detection of Exercise Thresholds in
  Cardiopulmonary Exercising Tests: From Regression to Generation to Explanation. *Sensors*.
  2023;23(2):826. [doi:10.3390/s23020826](https://doi.org/10.3390/s23020826)

## License and disclaimer

MIT. Research software, not a medical device: it does not diagnose, and its results are no
substitute for a clinician's judgement.
