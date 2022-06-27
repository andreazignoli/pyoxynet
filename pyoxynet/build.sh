#!/bin/sh
rm -rf dist
rm -rf pyoxynet.egg-info
python setup.py sdist
 twine upload dist/*
cd ..
cd docs
make html
cd ..
cd pyoxynet