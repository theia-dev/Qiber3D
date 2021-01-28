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
from pathlib import Path
import urllib.request

x3dom_url = 'http://www.x3dom.org/release/x3dom.'
for suffix in ['js', 'css']:
    save_path = Path('static') / suffix / f'x3dom.{suffix}'
    if save_path.is_file():
        save_path.unlink()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(x3dom_url+suffix, save_path)

sys.path.insert(0, os.path.abspath('../'))
sys.path.append(os.path.abspath("./ext"))

# -- Project information -----------------------------------------------------

project = 'Qiber3D'
copyright = '2021, Hagen Eckert, Anna Jaeschke'
author = 'Hagen Eckert, Anna Jaeschke'


from Qiber3D import config
# The full version, including alpha/beta/rc tags
release = config.version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'sphinxcontrib.bibtex',
    'html_extras'
]

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

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['static']

html_css_files = [
    'css/x3dom.css',
]

html_js_files = [
    'js/x3dom.js',
]

# autodoc_typehints = 'description'  # show type hints in doc body instead of signature
autoclass_content = 'both'  # get docstring from class level and init simultaneously
master_doc = 'index'
bibtex_bibfiles = ['refs.bib']