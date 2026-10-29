# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
sys.path.insert(0, os.path.abspath('../pyoxynet'))

# -- Project information -----------------------------------------------------

project = 'pyoxynet'
copyright = '2025, Andrea Zignoli'
author = 'Andrea Zignoli'

# The version, from the installed package (Read the Docs installs it) or from pyproject.toml
try:
    from importlib.metadata import version as _installed_version
    release = _installed_version('pyoxynet')
except Exception:
    import re
    with open(os.path.join(os.path.dirname(__file__), '..', 'pyoxynet', 'pyproject.toml')) as _f:
        release = re.search(r'^version = "([^"]+)"', _f.read(), re.M).group(1)
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Autosummary settings
autosummary_generate = True

# Autodoc settings
autodoc_mock_imports = ['tensorflow', 'ai_edge_litert']
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# No custom static files: docs/_static is not in the repository.
html_static_path = []