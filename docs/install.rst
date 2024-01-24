Installation
============

Using pip
---------

Optional but recommended step: install a conda environment::

  conda create -n <cosipy_env_name> python=3.10 pip
  conda activate <cosipy_env_name>

Note: currently cosipy is not compatible with Python 3.11 and 3.12, mainly due to
installation issues with a dependency (astromodels, see issues `#201 <https://github.com/threeML/astromodels/issues/201>`_ and `#204 <https://github.com/threeML/astromodels/issues/204>`_)

Install with pip::
  
  pip install cosipy==0.0.2a1

Note: you need to specify the alpha release 0.0.2a1, otherwise pip will fall back to
the latest regular release (which is currently unusable). This will be updated when
we have our next regular release.
  

From source (for developers)
----------------------------

Optional but recommended step: install a conda environment::

  conda create -n <cosipy_env_name> python=3.10 pip
  conda activate <cosipy_env_name>

Also optional but recommented: before installing cosipy, install the main
dependencies from source (similar
procedure as for cosipy below). These are histpy, mhealpy, scoords, threeml and
astromodels. The reason is that these libraries might be changing rapidly to
accomodate new features in cosipy. 
  
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

You need pandoc, sphinx, nbsphinx, sphinx_rtd_theme and mock. Using conda::

    conda install -c conda-forge pandoc nbsphinx sphinx_rtd_theme mock

Onece you have this requirements, run::

    cd docs
    make html

To read the documentation, open ``docs/_build/html/index.html`` in a browser.

