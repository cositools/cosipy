Installation
============

For developers
--------------

Do the following (preferably inside a conda environment)::

    git clone git@github.com:cositools/cosipy.git
    cd cosipy
    pip install -e .

The flag ``-e`` (``--editable``) allows you to make changes and try them without
having to run ``pip`` again.

Testing
.......

When you make a change, check that it didn't break something by running::

    pytest

You can install ``pytest`` with::

    conda install -c conda-forge pytest

Compiling the docs
------------------

You need sphinx, nbsphinx and sphinx_rtd_theme. Using conda::

    conda install -c conda-forge nbsphinx sphinx_rtd_theme

Onece you have this requirements, run::

    cd docs
    make html

To read the documentation, open ``docs/_build/html/index.html`` in a browser.

