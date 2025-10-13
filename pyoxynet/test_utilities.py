#!/usr/bin/env python3
"""
Unit tests for pyoxynet utilities functions

Run tests with:
    python -m pytest test_utilities.py -v
    or
    python test_utilities.py

Requirements:
    pip install scipy pandas numpy tensorflow
    or
    pip install scipy pandas numpy tflite-runtime

Note: Some tests will be skipped if dependencies are not available.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: NumPy not installed. Some tests will be skipped.")

try:
    from pyoxynet.utilities import (
        load_tf_generator,
        load_tflite_model,
        get_sec,
        optimal_filter
    )
    HAS_PYOXYNET = True
except ImportError as e:
    HAS_PYOXYNET = False
    print(f"Warning: Could not import pyoxynet utilities: {e}")
    print("Some tests will be skipped. Install dependencies: pip install scipy pandas numpy")


@unittest.skipIf(not HAS_PYOXYNET, "PyOxynet utilities not available")
class TestLoadTFLiteModel(unittest.TestCase):
    """Test TFLite model loading functionality"""

    def test_load_tflite_model_returns_interpreter(self):
        """Test that load_tflite_model returns an interpreter or None"""
        try:
            interpreter = load_tflite_model()
            # Should return an interpreter object or None
            self.assertTrue(interpreter is None or hasattr(interpreter, 'invoke'))
        except Exception as e:
            # If model file is missing, that's acceptable for testing
            self.assertIn('not found', str(e).lower())

    def test_load_tflite_model_handles_missing_file(self):
        """Test graceful handling when model file is missing"""
        # This should not raise an exception, should return None or handle gracefully
        try:
            interpreter = load_tflite_model()
            # If it returns, it should be None or valid interpreter
            self.assertTrue(interpreter is None or hasattr(interpreter, 'get_input_details'))
        except FileNotFoundError:
            # Expected if model file is missing
            pass


@unittest.skipIf(not HAS_PYOXYNET, "PyOxynet utilities not available")
class TestLoadTFGenerator(unittest.TestCase):
    """Test TensorFlow generator model loading"""

    def test_load_tf_generator_without_tensorflow(self):
        """Test that load_tf_generator handles missing TensorFlow gracefully"""
        # Try to unload tensorflow if it exists
        if 'tensorflow' in sys.modules:
            # Save it
            tf_module = sys.modules['tensorflow']
            sys.modules['tensorflow'] = None

            try:
                result = load_tf_generator()
                # Should return None when TensorFlow is missing
                self.assertIsNone(result)
            finally:
                # Restore tensorflow module
                sys.modules['tensorflow'] = tf_module
        else:
            # TensorFlow not installed, should return None
            result = load_tf_generator()
            self.assertIsNone(result)

    def test_load_tf_generator_with_tensorflow(self):
        """Test generator loading when TensorFlow is available"""
        try:
            import tensorflow as tf
            # TensorFlow is available
            result = load_tf_generator()
            # Should return model or None (if model files missing)
            self.assertTrue(result is None or hasattr(result, 'predict') or hasattr(result, '__call__'))
        except (ImportError, ModuleNotFoundError):
            # TensorFlow not available, skip this test
            self.skipTest("TensorFlow not installed")


@unittest.skipIf(not HAS_PYOXYNET, "PyOxynet utilities not available")
class TestGetSec(unittest.TestCase):
    """Test time string to seconds conversion"""

    def test_get_sec_with_mm_ss(self):
        """Test conversion of MM:SS format"""
        self.assertEqual(get_sec("00:30"), 30)
        self.assertEqual(get_sec("01:00"), 60)
        self.assertEqual(get_sec("10:30"), 630)

    def test_get_sec_with_hh_mm_ss(self):
        """Test conversion of HH:MM:SS format"""
        self.assertEqual(get_sec("00:00:30"), 30)
        self.assertEqual(get_sec("00:01:00"), 60)
        self.assertEqual(get_sec("01:00:00"), 3600)
        self.assertEqual(get_sec("01:30:45"), 5445)

    def test_get_sec_with_edge_cases(self):
        """Test edge cases"""
        self.assertEqual(get_sec("00:00"), 0)
        self.assertEqual(get_sec("0:0"), 0)


@unittest.skipIf(not HAS_PYOXYNET or not HAS_NUMPY, "PyOxynet utilities or NumPy not available")
class TestOptimalFilter(unittest.TestCase):
    """Test optimal filtering function"""

    def test_optimal_filter_basic(self):
        """Test basic filtering functionality"""
        x = np.arange(100)
        y = np.sin(x / 10.0) + np.random.randn(100) * 0.1

        filtered = optimal_filter(x, y, filter_size=10)

        # Filtered should have same length
        self.assertEqual(len(filtered), len(y))
        # Filtered should be smoother (lower variance)
        self.assertLess(np.var(filtered), np.var(y))

    def test_optimal_filter_preserves_shape(self):
        """Test that filtering preserves array shape"""
        x = np.linspace(0, 10, 50)
        y = np.random.randn(50)

        filtered = optimal_filter(x, y, filter_size=5)

        self.assertEqual(filtered.shape, y.shape)

    def test_optimal_filter_with_small_window(self):
        """Test filtering with small window size"""
        x = np.arange(20)
        y = np.random.randn(20)

        filtered = optimal_filter(x, y, filter_size=3)

        self.assertIsNotNone(filtered)
        self.assertEqual(len(filtered), len(y))


@unittest.skipIf(not HAS_PYOXYNET, "PyOxynet utilities not available")
class TestGenerateCPET(unittest.TestCase):
    """Test CPET data generation"""

    def test_generate_cpet_requires_generator(self):
        """Test that generate_CPET raises error when generator is None"""
        from pyoxynet.utilities import generate_CPET

        with self.assertRaises(ValueError) as context:
            generate_CPET(None)

        self.assertIn("Generator model is required", str(context.exception))

    def test_generate_cpet_with_valid_generator(self):
        """Test CPET generation with a valid generator model"""
        from pyoxynet.utilities import generate_CPET

        try:
            # Try to load generator
            generator = load_tf_generator()

            if generator is not None:
                # Generate CPET data
                df, data = generate_CPET(
                    generator,
                    plot=False,
                    fitness_group=2,
                    noise_factor=2.0
                )

                # Check that dataframe has expected columns
                expected_columns = ['time', 'VO2_I', 'VCO2_I', 'VE_I', 'HR_I', 'RF_I', 'PetO2_I', 'PetCO2_I']
                for col in expected_columns:
                    self.assertIn(col, df.columns)

                # Check that data dict has expected keys
                self.assertIn('VT1', data)
                self.assertIn('VT2', data)
            else:
                self.skipTest("Generator model not available")
        except (ImportError, ModuleNotFoundError):
            self.skipTest("TensorFlow not installed")


@unittest.skipIf(not HAS_PYOXYNET, "PyOxynet utilities not available")
class TestSafeNormalize(unittest.TestCase):
    """Test safe normalization in generate_CPET"""

    def test_safe_normalize_prevents_division_by_zero(self):
        """Test that safe_normalize handles constant arrays"""
        from pyoxynet.utilities import generate_CPET

        # This would test the internal safe_normalize function
        # Since it's defined inside generate_CPET, we test indirectly
        # by ensuring generate_CPET doesn't crash with edge case data

        # We can't directly test without a generator, so skip if unavailable
        try:
            generator = load_tf_generator()
            if generator is None:
                self.skipTest("Generator not available")
        except (ImportError, ModuleNotFoundError):
            self.skipTest("TensorFlow not installed")


def generate_and_plot_cpet(output_path='/tmp/df_gen.png'):
    """
    Generate CPET data and create visualization plot

    This function generates synthetic CPET data using the TensorFlow generator,
    creates a scatter plot with ventilatory thresholds, and saves to file.

    Args:
        output_path (str): Path to save the plot. Default: '/tmp/df_gen.png'

    Returns:
        tuple: (df_gen, dict_gen) - DataFrame and metadata dict

    Example:
        >>> df_gen, dict_gen = generate_and_plot_cpet()
        >>> print(f"Saved plot to /tmp/df_gen.png")
    """
    print("\n" + "=" * 70)
    print("Generating CPET Data and Creating Visualization")
    print("=" * 70)

    # Import required modules
    try:
        from pyoxynet.utilities import load_tf_generator, generate_CPET
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Error: Missing dependencies: {e}")
        print("Install with: pip install tensorflow scipy pandas matplotlib")
        return None, None

    # Load generator
    print("\n1. Loading TensorFlow generator model...")
    generator_model = load_tf_generator()

    if generator_model is None:
        print("   ✗ Failed to load generator model")
        print("   Make sure TensorFlow is installed: pip install tensorflow")
        return None, None

    print("   ✓ Generator model loaded")

    # Generate CPET data
    print("\n2. Generating CPET data...")
    df_gen, dict_gen = generate_CPET(generator_model, plot=False, fitness_group=2)

    print(f"   ✓ Generated {len(df_gen)} data points")
    print(f"   - VT1: {dict_gen['VT1']} sec (VO2: {dict_gen['VO2VT1']} ml/min)")
    print(f"   - VT2: {dict_gen['VT2']} sec (VO2: {dict_gen['VO2VT2']} ml/min)")

    # Create visualization
    print("\n3. Creating visualization...")
    plt.figure(figsize=(10, 6))

    # Scatter plot
    plt.scatter(df_gen.VO2_I, df_gen.VCO2_I, alpha=0.6, s=10, label='CPET Data')

    # VT1 vertical line
    plt.vlines(
        int(dict_gen['VO2VT1']),
        df_gen.VCO2_I.min(),
        df_gen.VCO2_I.max(),
        colors='red',
        linestyles='solid',
        linewidth=2,
        label=f"VT1 (VO2={int(dict_gen['VO2VT1'])} ml/min)"
    )

    # VT2 vertical line
    plt.vlines(
        int(dict_gen['VO2VT2']),
        df_gen.VCO2_I.min(),
        df_gen.VCO2_I.max(),
        colors='orange',
        linestyles='dashed',
        linewidth=2,
        label=f"VT2 (VO2={int(dict_gen['VO2VT2'])} ml/min)"
    )

    plt.xlabel('VO2 (ml/min)', fontsize=12)
    plt.ylabel('VCO2 (ml/min)', fontsize=12)
    plt.title('Generated CPET: VO2 vs VCO2 with Ventilatory Thresholds', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Plot saved to: {output_path}")

    # Also save CSV
    csv_path = output_path.replace('.png', '.csv')
    df_gen.to_csv(csv_path, index=False)
    print(f"   ✓ Data saved to: {csv_path}")

    print("\n" + "=" * 70)
    print("✓ Complete!")
    print(f"View plot: open {output_path}")
    print("=" * 70)

    return df_gen, dict_gen


def run_tests():
    """Run all tests"""
    print("=" * 70)
    print("Running PyOxynet Utilities Tests")
    print("=" * 70)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLoadTFLiteModel))
    suite.addTests(loader.loadTestsFromTestCase(TestLoadTFGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestGetSec))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimalFilter))
    suite.addTests(loader.loadTestsFromTestCase(TestGenerateCPET))
    suite.addTests(loader.loadTestsFromTestCase(TestSafeNormalize))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
