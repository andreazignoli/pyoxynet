#!/bin/sh
rm -rf dist
rm -rf pyoxynet.egg-info
source venv/bin/activate
python3 -m build
python3 -m twine upload dist/*
sphinx-build -b html docs/source/ docs/build/html
deactivate