#!/bin/sh
rm -rf dist
rm -rf pyoxynet.egg-info
python3 -m build
python3 -m twine upload dist/*
cd ..
cd docs
make html
cd ..
cd pyoxynet