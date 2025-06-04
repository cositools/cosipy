Unit tests
----------

An unit tests verified that a modules, a class or a method is working as expected. As opposed to integration and functional tests, they should be as specific as possible, attempting to isolate small blocks of code --i.e. one function at a time. The main goals for unit tests are:

  - Confirm that cosipy was succesfully installed.
  - Catch pull request that break previous code and prevent maintainer from merging them.

We are working towards having 100% unit test coverage. That means that all lines of code are run at least by one unit test. New pull request that reduce the coverge percentage should not be merge until this is remedied by adding new tests.
    
Running unit tests
^^^^^^^^^^^^^^^^^^

Install ``pytest`` and ``pytest-cov`` with::

    pip install pytest pytest-cov

Go to the root folder of your local working repository, the one containing the folder ``tests``. You need to clone the repository, as opposed to installing cosipy with ``pip``.
    
Run::

    pytest --cov=cosipy --cov-report term --cov-report html:tests/cov

The last line of the report will tell you how many test failed (if any), how many passed, and how many warnings were generated.
  
Open ``tests/cov/index.html`` in a browser and check the coverage. This
is the percentage of lines that were executed during the tests. The goal is to have
a 100% coverage!

Adding a test
^^^^^^^^^^^^^

The pytest library automatically looks inside the ``tests`` folder for function starting with the word ``test`` inside files starting with the word ``test``. Subfolder inside the ``tests`` folder need to contain a file called ``__init__.py`` (can be empty) in order to be picked up.

Each ``test*`` function constitutes a test. If they are run without any errors or exception, the pytest considers that the test was suscessful, independently of the return value. Use the builtin ``asssert`` keyword to compare a result to an expected value::

  from cosipy import test_data
  from cosipy.response import FullDetectorResponse

  def test_get_effective_area():
    
      response_path = test_data.path/"test_full_detector_response.h5"

      with FullDetectorResponse.open(response_path) as response:

          assert response[0].ndim == 6

The ``assert`` method will raise an exception if the result from the operation on the right return ``False``.

For some tests you might need a dataset. When that's the case, add your files to the ``cosipy/test_data`` folder, which the tests can find by calling ``cosipy.test_data.path``, as shown in the example above. Use the smallest file size possible required by your test. Remember that for an unit test you do not need a full-fledge data file. Instead, mock data and small subsets are enough to test the algorithms, even if they don't result in realistic outputs or with good scientific quality. Full-fledge data might be needed for integration and functional test, but those will be handled separetly using data outside the cosipy respository.








    




