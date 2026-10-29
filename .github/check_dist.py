"""Fail unless a built wheel carries the three models and the 138 example tests, and no junk."""
import sys
import zipfile

names = zipfile.ZipFile(sys.argv[1]).namelist()
models = sorted(n for n in names if n.startswith('pyoxynet/models/') and n.endswith('.tflite'))
examples = [n for n in names if '/data_test/' in n and '_real_test_' in n]
junk = [n for n in names if '__pycache__' in n or n.endswith('.DS_Store')]
print(f'{len(names)} files, models {models}, {len(examples)} example tests')

assert models == ['pyoxynet/models/cnn.tflite', 'pyoxynet/models/generator.tflite',
                  'pyoxynet/models/lstm.tflite'], models
assert len(examples) == 138, len(examples)
assert not junk, junk
