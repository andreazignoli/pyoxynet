# Pyoxynet

**Automatic interpretation of cardiopulmonary exercise test (CPET) data using deep learning.**

[![PyPI version](https://img.shields.io/pypi/v/pyoxynet.svg)](https://pypi.org/project/pyoxynet/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyoxynet.svg)](https://pypi.org/project/pyoxynet/)
[![Documentation](https://readthedocs.org/projects/pyoxynet/badge/?version=latest)](https://pyoxynet.readthedocs.io/en/latest/)

Pyoxynet is a Python package for automated CPET analysis using AI models. Part of the [Oxynet project](http://oxynet.net) for universal access to quality healthcare.

**Key Features:**
- 🔬 AI-powered inference - Automatically estimate exercise intensity domains
- 🎲 Synthetic data generation - Create realistic CPET data with conditional GANs
- ⚡ Lightweight deployment - TFLite support with ~90% smaller footprint
- 📊 Model explainability - SHAP integration for understanding predictions

📚 **[Documentation](https://pyoxynet.readthedocs.io/)** | 🌐 **[Web App](https://pyoxynet-lite-app-b415901c79ab.herokuapp.com/)** | 💻 **[GitHub](https://github.com/andreazignoli/pyoxynet)** 

## Installation

**Requirements:** Python 3.10+ and NumPy < 2.0

```bash
# Lite version (recommended) - Core functionality
pip install pyoxynet

# TFLite version - Adds lightweight model inference
pip install "pyoxynet[tflite]" --extra-index-url https://google-coral.github.io/py-repo/

# Full version - Complete TensorFlow support
pip install "pyoxynet[full]"
```

**NumPy compatibility:** If you encounter NumPy 2.x issues with TFLite, install: `pip install "numpy<2"`

## Quick Start

```python
import pyoxynet

# Load model and run inference on sample data
model = pyoxynet.load_tf_model(n_inputs=5, past_points=40, model='CNN')
pyoxynet.test_pyoxynet(model)

# Generate synthetic CPET data
generator = pyoxynet.load_tf_generator()
df = pyoxynet.generate_CPET(generator, plot=True)
```

**Required data:** VO2, VCO2, VE, PetO2, PetCO2 (sampled at 1-second intervals) 

## Resources

- 📖 [Documentation](https://pyoxynet.readthedocs.io/) - Complete API reference and examples
- 💻 [GitHub](https://github.com/andreazignoli/pyoxynet) - Source code and detailed README
- 🌐 [Web App](https://pyoxynet-lite-app-b415901c79ab.herokuapp.com/) - Try it online
- 🔬 [Research Papers](https://github.com/andreazignoli/pyoxynet#additional-reading) - Scientific background

## License

MIT License - See [LICENSE](https://github.com/andreazignoli/pyoxynet/blob/master/LICENSE.txt) for details.

## Medical Disclaimer

This software is for informational purposes only and is not intended as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers.