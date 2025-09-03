# Pyoxynet package

This README has been intentionally created for Pypi. Please find a more extended and detailed description of the project on the [GitHub repository](https://github.com/andreazignoli/pyoxynet). 

## Documentation

Please refer to the extended [documentation](https://pyoxynet.readthedocs.io/en/latest/) to read the docs. 

## Installation

**Important**: Starting from version 0.1.5, pyoxynet requires **Python 3.10** and **NumPy < 2.0** for optimal performance with TensorFlow Lite support.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install *pyoxynet*.

### Lite version (default) - Recommended
```bash
pip install --upgrade pip
pip install pyoxynet
```

### TFLite version (with lightweight TensorFlow Lite support)
For applications requiring local model inference with minimal dependencies:
```bash
pip install "pyoxynet[tflite]" --extra-index-url https://google-coral.github.io/py-repo/
```

### Full version (with complete TensorFlow support)
For advanced users who need full TensorFlow model functionality:
```bash
pip install "pyoxynet[full]"
```

## Architecture Changes in v0.1.5

This version introduces a **lighter, faster architecture** designed for improved deployment and inference performance:

- **Lite version**: Core functionality with minimal dependencies - perfect for data processing and analysis
- **TFLite version**: Adds TensorFlow Lite runtime for efficient model inference with ~90% smaller footprint than full TensorFlow
- **Full version**: Complete TensorFlow support for model training, advanced features, and SHAP explanations

**Why these changes?**
- **Faster deployment**: Reduced package size and installation time
- **Better performance**: TFLite optimized inference for production environments  
- **Flexibility**: Choose only the dependencies you need
- **API compatibility**: Internal API uses TFLite for lightweight, fast predictions

**Compatibility Note**: 
If you encounter NumPy compatibility issues with TFLite (e.g., "_ARRAY_API not found" errors), install NumPy 1.x:
```bash
pip install "numpy<2"
```
This is because `tflite-runtime` was compiled against NumPy 1.x and is not yet compatible with NumPy 2.x.

The lite version provides the core functionality with reduced dependencies, while TFLite and full versions add model inference capabilities.

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