# Pyoxynet package

This README has been intentionally created for Pypi. Please find a more extended and detailed description of the project on the [GitHub repository](https://github.com/andreazignoli/pyoxynet). 

## Documentation

Please refer to the extended [documentation](https://pyoxynet.readthedocs.io/en/latest/) to read the docs. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install *pyoxynet*.

```bash
pip install --upgrade pip
pip install pyoxynet
```

## Test settings

```python
import pyoxynet

# Load the TFL model
tfl_model = pyoxynet.load_tf_model()

# Make inference on a random input
test_tfl_model(tfl_model)

# Plot the inference on a test dataset
pyoxynet.test_pyoxynet()
```

Data required for the inference include oxygen uptake (VO2), exhaled CO2 (VCO2), minute ventilation (VE), end tidal O2 (PetO2) and CO2(PetCO2), and ventilatory equivalents (VEVO2 and VEVCO2):

| VO2 | VCO2 | VE | PetO2 | PetCO2 | VEVO2 | VEVCO2 |
|-----|------|----|-------|--------|-------|--------|
|     |      |    |       |        |       |        |
|     |      |    |       |        |       |        |
|     |      |    |       |        |       |        |

This structure might evolve with different package version, so please refer to the main [GitHub repository](https://github.com/andreazignoli/pyoxynet) README for the latest structure details. 

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

Please refer to the LICENSE file at the [GitHub repository](https://github.com/andreazignoli/pyoxynet). 

## Disclaimer

All content found on this website, including: text, images, tables, or other formats are created for informational purposes only. The information provided by this software is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something has been provided by this software.