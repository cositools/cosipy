Installation
============

Using pip
---------

Optional but recommended step: install a conda environment::

  conda create -n <cosipy_env_name> python=3.12 pip
  conda activate <cosipy_env_name>

Install with pip::
  
  pip install cosipy


Note: The tutorials, examples, and other documentation are not shipped with the PyPi (pip) release —only the embedded docstrings. You can see this information in the `main repository <https://github.com/cositools/cosipy>`_  ("docs" folder).

From source (for developers)
----------------------------

Optional but recommended step: install a conda environment::

  conda create -n <cosipy_env_name> python=3.12 pip
  conda activate <cosipy_env_name>

Also optional but recommended: before installing cosipy, install the main
dependencies from the source (similar
procedure as for cosipy below). These are histpy, mhealpy, scoords, threeml and
astromodels. The reason is that these libraries might be changing rapidly to
accommodate new features in cosipy. 
  
Do the following (preferably inside a conda environment)::

    git clone git@github.com:cositools/cosipy.git
    cd cosipy
    pip install -e .

The flag ``-e`` (``--editable``) allows you to make changes and try them without
having to run ``pip`` again.

Enable machine learning tools ([ml])
------------------------------------

Some cosipy features require ``pytorch`` and other related libraries
which are not installed by default. In order to access these you need to
specify the ``[ml]`` extra packages during the installation. e.g.::

    pip install cosipy[ml]

or, if you are installing from from source::

    pip install '.[ml]'

If you do not install these optional dependencies, then some imports in the
``.ml`` submodules will fail. For example::

    from cosipy.background_estimation.ml import ContinuumEstimationNN

would result in::

    ImportError: Install cosipy with [ml] optional packages to use these features.

NOTE: MacOS users with an M-series chip are recommended to install healpy and pytorch
through conda instead of pip. See `OMP: Error #15`_ below.

Troubleshooting
---------------

OMP: Error #15
^^^^^^^^^^^^^^

This is caused by multiple and incompatible ``OpenMP`` libraries shipped
with ``pip``-installed packages. See `PyTorch Issue 44282 <https://github.com/pytorch/pytorch/issues/44282>`_.

Some observed error messages are:

::

    OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
    OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://openmp.llvm.org/

::

    Current thread 0x00000001fd026240 (most recent call first):
    File "<frozen importlib._bootstrap>", line 488 in _call_with_frames_removed
    File "<frozen importlib._bootstrap_external>", line 1293 in create_module
    File "<frozen importlib._bootstrap>", line 813 in module_from_spec
    File "<frozen importlib._bootstrap>", line 921 in _load_unlocked
    File "<frozen importlib._bootstrap>", line 1331 in _find_and_load_unlocked
    File "<frozen importlib._bootstrap>", line 1360 in _find_and_load
    ...
    Abort trap: 6

or

::

    Segmentation fault: 11

While the root cause of this error is unrelated to ``cosipy``, it can
be caused by the dependencies installed by default through ``pip``.
In particular, we have seen this error under the following conditions::

1. Running on a system with an Apple M-series chip.
2. Importing a class from a machine learning submodule (".ml") --since it imports ``torch``.
3. Running another command which uses OpenMP, `healpy.smoothing` (currently used by
``cosipy.imaging_deconvolution.AllSkyImageModel`` and ``cosipy.background_estimation.LineBackgroundEstimation``).

The current workaround to solve this is to install both ``healpy`` and ``pytorch``
from conda before installing cosipy (so they don't get installed by ``pip``)::

    conda create -n <cosipy_env_name> python=3.12 pip healpy pytorch

The conda installation makes sure that the OpenMP libraries are
compatible an work with an M chip.

Note also that ``pytest`` imports all tests during the initial collection step.
This means that, unless healpy and pytorch are installed through conda, running
``pytest`` with all test can fail despite they working fine individually.

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


OSError: Could not find library XSFunctions. Impossible to compile Xspec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This error can occur while installing astromodels::

    Xspec is detected. Will compile the Xspec extension.
    [...]
    Could not find library XSFunctions. Impossible to compile Xspec


While astromodels support Xspec functions, these are generally not currently relevant for the use and development of cosipy. The most straightforward workaround is to temporarily hide your Xspec installation so that astromodels does not try to link to it. Before running `pip`, run::

    unset HEADAS ASTRO_XSPEC_VERSION


Testing
-------

When you make a change, check that it didn't break something by running::

    pytest --cov=cosipy --cov-report term --cov-report html:tests/coverage_report

Open ``tests/coverage_report/index.html`` in a browser and check the coverage. This
is the percentage of lines that were executed during the tests. The goal is to have
a 100% coverage!
    
You can install ``pytest`` and ``pytest-cov`` with::

    conda install -c conda-forge pytest pytest-cov

For MacOS with an M-series chip, see section `OMP: Error #15`_ above for the necessary
extra steps for all tests to succeed.

Compiling the docs
------------------

You need pandoc, sphinx, nbsphinx, sphinx_rtd_theme and mock. Using conda::

    conda install -c conda-forge pandoc=3.1.3 nbsphinx=0.9.3 sphinx_rtd_theme=2.0.0 mock=5.1.0

Other versions might work was well.

Once you have these requirements, run::

    cd docs
    make html

To read the documentation, open ``docs/_build/html/index.html`` in a browser.


