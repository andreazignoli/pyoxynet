#!/bin/sh
source ../venv/bin/activate
cd pyoxynet
rm -rf dist
rm -rf pyoxynet.egg-info
python3 -m build
#python3 pyoxynet/setup.py sdist bdist_wheel 
#python3 pyoxynet/setup.py sdist
python3 -m twine upload dist/*
# sphinx-build -b html pyoxynet/docs/source/ pyoxynet/docs/build/html
deactivate