#!/usr/bin/env python3
"""
Test script to generate CPET data and create visualization

Usage:
    cd /Users/andreazignoli/pyoxynet
    PYTHONPATH=/Users/andreazignoli/pyoxynet/pyoxynet python3 test_generate_cpet.py
"""

import sys
import os

# Add pyoxynet to path
sys.path.insert(0, '/Users/andreazignoli/pyoxynet/pyoxynet')

print("=" * 70)
print("PyOxynet CPET Data Generation and Visualization Test")
print("=" * 70)

# Import required modules
print("\n1. Loading modules...")
try:
    from pyoxynet.utilities import load_tf_generator, generate_CPET
    import matplotlib.pyplot as plt
    print("   ✓ Modules loaded successfully")
except ImportError as e:
    print(f"   ✗ Error loading modules: {e}")
    print("\nPlease install dependencies:")
    print("   pip install tensorflow scipy pandas matplotlib")
    sys.exit(1)

# Load generator model
print("\n2. Loading TensorFlow generator model...")
generator_model = load_tf_generator()

if generator_model is None:
    print("   ✗ Failed to load generator model")
    print("   Make sure TensorFlow is installed: pip install tensorflow")
    sys.exit(1)

print("   ✓ Generator model loaded successfully")

# Generate CPET data
print("\n3. Generating CPET data...")
try:
    df_gen, dict_gen = generate_CPET(generator_model, plot=False, fitness_group=2)
    print("   ✓ CPET data generated successfully")
    print(f"   - Data points: {len(df_gen)}")
    print(f"   - VT1: {dict_gen['VT1']} sec (VO2: {dict_gen['VO2VT1']} ml/min)")
    print(f"   - VT2: {dict_gen['VT2']} sec (VO2: {dict_gen['VO2VT2']} ml/min)")
except Exception as e:
    print(f"   ✗ Error generating CPET data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create visualization
print("\n4. Creating visualization...")
try:
    plt.figure(figsize=(10, 6))

    # Scatter plot
    plt.scatter(df_gen.VO2_I, df_gen.VCO2_I, alpha=0.6, s=10, label='CPET Data')

    # VT1 line
    plt.vlines(
        int(dict_gen['VO2VT1']),
        df_gen.VCO2_I.min(),
        df_gen.VCO2_I.max(),
        colors='red',
        linestyles='solid',
        linewidth=2,
        label=f"VT1 (VO2={int(dict_gen['VO2VT1'])} ml/min)"
    )

    # VT2 line
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
    plt.title('Generated CPET Data: VO2 vs VCO2 with Ventilatory Thresholds', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # Save to /tmp
    output_path = '/tmp/df_gen.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Visualization saved to: {output_path}")

except Exception as e:
    print(f"   ✗ Error creating visualization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Also save the dataframe for inspection
print("\n5. Saving data...")
try:
    csv_path = '/tmp/df_gen.csv'
    df_gen.to_csv(csv_path, index=False)
    print(f"   ✓ Data saved to: {csv_path}")

    # Print first few rows
    print("\n   First 5 rows of generated data:")
    print(df_gen[['time', 'VO2_I', 'VCO2_I', 'VE_I', 'HR_I']].head())

except Exception as e:
    print(f"   ✗ Error saving data: {e}")

print("\n" + "=" * 70)
print("✓ CPET Generation Complete!")
print("=" * 70)
print(f"\nView the plot: open {output_path}")
print(f"View the data: open {csv_path}")
print()
