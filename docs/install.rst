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

    pytest --cov=cosipy --cov-report term --cov-report html:tests/coverage_report

Open ``tests/coverage_report/index.html`` in a browser and check the coverage. This
is the percentage of lines that were executed during the tests. The goal is to have
a 100% coverage!
    
You can install ``pytest`` and ``pytest-cov`` with::

    conda install -c conda-forge pytest pytest-cov

Compiling the docs
------------------

You need sphinx, nbsphinx and sphinx_rtd_theme. Using conda::

    conda install -c conda-forge nbsphinx sphinx_rtd_theme

Onece you have this requirements, run::

    cd docs
    make html

To read the documentation, open ``docs/_build/html/index.html`` in a browser.

