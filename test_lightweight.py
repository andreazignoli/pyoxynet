#!/usr/bin/env python3
"""
Test script to verify pyoxynet works with tflite-runtime only
"""

import sys
import subprocess
import tempfile

def test_lightweight_install():
    """Test that pyoxynet works with tflite-runtime only"""
    
    # Create a temporary Python script that tests the import
    test_script = '''
import sys
# Mock TensorFlow import failure to simulate tflite-runtime only environment
sys.modules['tensorflow'] = None

try:
    import pyoxynet
    from pyoxynet.utilities import TENSORFLOW_AVAILABLE, TENSORFLOW_LITE_ONLY
    
    print(f"TENSORFLOW_AVAILABLE: {TENSORFLOW_AVAILABLE}")
    print(f"TENSORFLOW_LITE_ONLY: {TENSORFLOW_LITE_ONLY}")
    
    # Test that functions requiring TensorFlow raise appropriate errors
    try:
        from pyoxynet.utilities import load_tf_model
        load_tf_model()
        print("ERROR: load_tf_model should have failed")
        sys.exit(1)
    except ImportError as e:
        if "pip install pyoxynet[full]" in str(e):
            print("SUCCESS: load_tf_model properly failed with helpful message")
        else:
            print(f"ERROR: Unexpected error message: {e}")
            sys.exit(1)
    
    # Test that basic utilities work
    from pyoxynet.utilities import get_sec, normalize
    import pandas as pd
    import numpy as np
    
    # Test time conversion
    assert get_sec("2:30") == 150
    print("SUCCESS: get_sec works")
    
    # Test normalization
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    normalized = normalize(df)
    assert normalized.min().min() == -1
    assert normalized.max().max() == 1
    print("SUCCESS: normalize works")
    
    print("\\nAll tests passed! Pyoxynet works in lightweight mode.")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        f.flush()
        
        # Run the test script
        result = subprocess.run([sys.executable, f.name], 
                               capture_output=True, text=True)
        
        print("Test output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
            
        return result.returncode == 0

if __name__ == "__main__":
    if test_lightweight_install():
        print("✅ Lightweight installation test PASSED")
    else:
        print("❌ Lightweight installation test FAILED")
        sys.exit(1)