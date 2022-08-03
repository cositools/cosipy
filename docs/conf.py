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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import cosipy

# -- Project information -----------------------------------------------------

project = 'cosipy'
copyright = '2022, COSI Team'
author = 'COSI Team'

# The full version, including alpha/beta/rc tags
release = cosipy.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.mathjax',
              'sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx',
              'sphinx.ext.coverage',
              'nbsphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The master toctree document.
master_doc = 'index'

# intersphinx

intersphinx_mapping = {
    'histpy': ('https://histpy.readthedocs.io/en/latest', None),
    'h5py' : ('https://docs.h5py.org/en/stable/', None),
    'astropy' : ('https://docs.astropy.org/en/stable', None),
    'python' : ('https://docs.python.org/3', None),
    'mhealpy' : ('https://mhealpy.readthedocs.io/en/latest/', None),
    'sparse' : ('https://sparse.pydata.org/en/stable/', None),
    'gammapy' : ('https://docs.gammapy.org/dev', None),
    'scipy' : ('https://scipy.github.io/devdocs', None),
  }

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = '_static/cosipy_logo.png'

# -- Extension configuration -------------------------------------------------

# nbpshinx
nbsphinx_execute = 'never'

# Autodoc
autodoc_member_order = 'bysource'

# Extensions to theme docs

# Fix issue with Napoleon RTD that displays "Variables" instead of "Attributes"
# credit - https://michaelgoerz.net/notes/extending-sphinx-napoleon-docstring-sections.html

from sphinx.ext.napoleon.docstring import GoogleDocstring

# first, we define new methods for any new sections and add them to the class
def parse_keys_section(self, section):
    return self._format_fields('Keys', self._consume_fields())
GoogleDocstring._parse_keys_section = parse_keys_section

def parse_attributes_section(self, section):
    return self._format_fields('Attributes', self._consume_fields())
GoogleDocstring._parse_attributes_section = parse_attributes_section

def parse_class_attributes_section(self, section):
    return self._format_fields('Class Attributes', self._consume_fields())
GoogleDocstring._parse_class_attributes_section = parse_class_attributes_section

# we now patch the parse method to guarantee that the the above methods are
# assigned to the _section dict
def patched_parse(self):
    self._sections['keys'] = self._parse_keys_section
    self._sections['class attributes'] = self._parse_class_attributes_section
    self._unpatched_parse()
GoogleDocstring._unpatched_parse = GoogleDocstring._parse
GoogleDocstring._parse = patched_parse
