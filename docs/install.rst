Installation
============

Using pip
---------

Optional but recommended step: install a conda environment::

  conda create -n <cosipy_env_name> python=3.10 pip
  conda activate <cosipy_env_name>

Note: currently cosipy is not compatible with Python 3.12 due to
installation issues with dependencies (`threeML <https://github.com/threeML/threeML/pull/631>`_ and `astromodels <https://github.com/threeML/astromodels/issues/204>`_)

Install with pip::
  
  pip install --use-pep517 cosipy

Note: ``--use-pep517`` is a temporary workaround to install `astromodels with new setuptools versions<https://github.com/threeML/astromodels/issues/209>`_. 

From source (for developers)
----------------------------

Optional but recommended step: install a conda environment::

  conda create -n <cosipy_env_name> python=3.10 pip
  conda activate <cosipy_env_name>

Also optional but recommended: before installing cosipy, install the main
dependencies from the source (similar
procedure as for cosipy below). These are histpy, mhealpy, scoords, threeml and
astromodels. The reason is that these libraries might be changing rapidly to
accommodate new features in cosipy. 
  
Do the following (preferably inside a conda environment)::

    git clone git@github.com:cositools/cosipy.git
    cd cosipy
    pip install --use-pep517 -e .

The flag ``-e`` (``--editable``) allows you to make changes and try them without
having to run ``pip`` again.

Troubleshooting
---------------

ERROR:: Could not find a local HDF5 installation.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This error is caused by missing h5py wheels for M1 chips. 

See https://github.com/h5py/h5py/issues/1810 and https://github.com/h5py/h5py/issues/1800

Currently, the best workaround for M1 users is to install h5py using conda before the cosipy installation::

    conda install h5py

Example error log::

    × Getting requirements to build wheel did not run successfully.
    │ exit code: 1
    ╰─> [13 lines of output]
        /var/folders/5p/wnc17p7s0gz1vd3krp7gly60v5n_5p/T/H5close39c45pt5.c:1:10: fatal error: 'H5public.h' file not found
        #include "H5public.h"
                 ^~~~~~~~~~~~
        1 error generated.
        cpuinfo failed, assuming no CPU features: 'flags'
        * Using Python 3.10.12 | packaged by conda-forge | (main, Jun 23 2023, 22:41:52) [Clang 15.0.7 ]
        * Found cython 3.0.10
        * USE_PKGCONFIG: True
        * Found conda env: ``/Users/mjmoss/miniforge3``
        .. ERROR:: Could not find a local HDF5 installation.
           You may need to explicitly state where your local HDF5 headers and
           library can be found by setting the ``HDF5_DIR`` environment
           variable or by using the ``--hdf5`` command-line option.


Testing
-------

.. warning::
    Under construction. Unit tests are not ready.
    
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

    conda install -c conda-forge pandoc=3.1.3 nbsphinx=0.9.3 sphinx_rtd_theme=2.0.0 mock=5.1.0

Other versions might work was well.

Once you have these requirements, run::

    cd docs
    make html

To read the documentation, open ``docs/_build/html/index.html`` in a browser.


