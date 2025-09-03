# Pyoxynet package

🧠 **AI-powered CPET analysis for exercise physiology research**

Pyoxynet is a Python package for cardiopulmonary exercise testing (CPET) analysis using machine learning models. It provides automated detection of ventilatory thresholds (VT1, VT2) and exercise intensity domain classification.

This README has been intentionally created for PyPI. Please find a more extended and detailed description of the project on the [GitHub repository](https://github.com/andreazignoli/pyoxynet). 

## 📚 Documentation

Please refer to the extended [documentation](https://pyoxynet.readthedocs.io/en/latest/) to read the docs. 

## 🚀 Installation

Pyoxynet offers **two installation modes** to suit different needs:

### Lightweight Installation (Recommended)
For most users, especially in production environments:

```bash
pip install --upgrade pip
pip install pyoxynet
```

**What you get:**
- 📦 Minimal dependencies (~50MB)
- 🔧 Core utilities and data processing
- ⚡ Basic CPET analysis functions
- 🧮 Mathematical and statistical tools

### Full Installation (Research & Development)
For researchers and developers who need the complete ML functionality:

```bash
pip install --upgrade pip
pip install pyoxynet[full]
```

**What you get:**
- 📦 All lightweight features PLUS
- 🧠 TensorFlow-based ML models (~650MB)
- 🔮 Advanced AI inference capabilities
- 🏋️ Full model training and evaluation tools

### Python Version Support
- ✅ Python 3.8, 3.9, 3.10, 3.11
- 🚫 Python < 3.8 not supported

## 🧪 Quick Start

### Basic Usage (Lightweight Installation)

```python
import pyoxynet
from pyoxynet import utilities
import pandas as pd

# Check installation mode
print(f"TensorFlow available: {utilities.TENSORFLOW_AVAILABLE}")
print(f"Lightweight mode: {utilities.TENSORFLOW_LITE_ONLY}")

# Basic utilities work in both modes
time_seconds = utilities.get_sec("2:30")  # Convert MM:SS to seconds
print(f"2:30 = {time_seconds} seconds")

# Load and process CPET data
data = pd.DataFrame({
    'VO2_I': [1200, 1400, 1600],
    'VCO2_I': [1000, 1200, 1500],
    'VE_I': [30, 35, 45],
    'PetO2_I': [100, 95, 90],
    'PetCO2_I': [35, 38, 42]
})

# Normalize data for analysis
normalized_data = utilities.normalize(data)
```

### Advanced Usage (Full Installation)

```python
import pyoxynet

# Load TensorFlow models (requires full installation)
try:
    tfl_model = pyoxynet.load_tf_model()
    print("✅ TensorFlow model loaded successfully")
    
    # Run complete CPET analysis
    pyoxynet.test_pyoxynet()
    
except ImportError as e:
    print("❌ TensorFlow models require full installation:")
    print("pip install pyoxynet[full]")
```

## 📊 Data Requirements

### Core Variables (Required)
For CPET analysis, your data should include these standardized variables:

| Variable | Description | Units | Example |
|----------|-------------|--------|---------|
| `VO2_I` | Oxygen uptake (inspired) | ml/min | 1200 |
| `VCO2_I` | CO2 output (inspired) | ml/min | 1000 |
| `VE_I` | Minute ventilation (inspired) | L/min | 30 |
| `PetO2_I` | End-tidal O2 (inspired) | mmHg | 100 |
| `PetCO2_I` | End-tidal CO2 (inspired) | mmHg | 35 |

### Optional Variables
| Variable | Description | Units | Notes |
|----------|-------------|--------|-------|
| `time` | Time elapsed | seconds | For time-series analysis |
| `VEVO2` | Ventilatory equivalent for O2 | - | Calculated automatically |
| `VEVCO2` | Ventilatory equivalent for CO2 | - | Calculated automatically |

### Data Format
```python
import pandas as pd

# Example data structure
cpet_data = pd.DataFrame({
    'time': [60, 120, 180, 240, 300],
    'VO2_I': [800, 1200, 1600, 2000, 2200],
    'VCO2_I': [600, 1000, 1400, 1900, 2100],
    'VE_I': [20, 30, 40, 55, 65],
    'PetO2_I': [105, 100, 95, 90, 85],
    'PetCO2_I': [30, 35, 40, 45, 48]
})
```

> ⚠️ **Note**: Data structure may evolve with package versions. Please refer to the [GitHub repository](https://github.com/andreazignoli/pyoxynet) for the latest specifications. 

## 🔧 Troubleshooting

### Import Errors
If you encounter import errors:

```python
# Check your installation mode
from pyoxynet import utilities
print("TensorFlow available:", utilities.TENSORFLOW_AVAILABLE)

# For TensorFlow-dependent features, upgrade to full version:
# pip install pyoxynet[full]
```

### Common Issues
- **"TensorFlow is required"**: Install with `pip install pyoxynet[full]`
- **Memory issues**: Use lightweight installation for production: `pip install pyoxynet`
- **Python version**: Ensure you're using Python 3.8+ (`python --version`)

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

Development setup:
```bash
git clone https://github.com/andreazignoli/pyoxynet.git
cd pyoxynet
pip install -e .[full]  # Editable install with full dependencies
```

## 📄 License

Please refer to the LICENSE file at the [GitHub repository](https://github.com/andreazignoli/pyoxynet). 

## 🏥 Medical Disclaimer

**⚠️ IMPORTANT**: All content provided by this software is created for **informational and research purposes only**. 

This software is **NOT intended** to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something provided by this software.

Use of this software in clinical settings must be validated and approved by appropriate medical authorities in your jurisdiction.