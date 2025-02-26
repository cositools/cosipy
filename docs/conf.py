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
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'cosipy'
copyright = '2022, COSI Team'
author = 'COSI Team'

# The full version, including alpha/beta/rc tags
with open('../cosipy/_version.py') as f:
    release = f.readline()

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

# mock dependencies so we don't have to install them
autodoc_mock_imports = ["histpy",
                        'threeML',
                        'astromodels',
                        'past',
                        'numpy',
                        'h5py',
                        'astropy',
                        'healpy',
                        'mhealpy',
                        'sparse',
                        'matplotlib',
                        'yaml',
                        'scoords',
                        'pandas',
                        'tqdm',
                        'scipy',
                        'psutil',
                        'awscli',
                        'yayc',
                        'iminuit'
                        ]

# There seems to be a conflict between unittest.mock (used by sphinx) and metaclasses
# The cosipy.threeml.custom_functions.Band_Eflux includes a metaclass from
# astromodels.functions.function, so we mock that one manually with the mock package
import mock

MOCK_MODULES = ['astromodels.functions.function', 'iminuit']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# intersphinx for mocked dependencies

intersphinx_mapping = {
    'histpy': ('https://histpy.readthedocs.io/en/latest', None),
    'threeML': ('https://threeml.readthedocs.io/en/latest/', None),
    'astromodels': ('https://astromodels.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'h5py' : ('https://docs.h5py.org/en/stable/', None),
    'astropy' : ('https://docs.astropy.org/en/stable', None),
    'python' : ('https://docs.python.org/3', None),
    'mhealpy' : ('https://mhealpy.readthedocs.io/en/latest/', None),
    'sparse' : ('https://sparse.pydata.org/en/stable/', None),
    'matplotlib' : ('https://matplotlib.org/stable/', None),
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

