from typing_extensions import override

from cosipy.util import fetch_wasabi_file, fetch_wasabi_file_header
import tempfile
import os
from pathlib import Path
import pytest

def test_fetch_wasabi_file():

    with tempfile.TemporaryDirectory() as tmpdir:

        filename = 'test_files/test_file.txt'

        # Using output
        output = Path(tmpdir)/"test_file.txt"
        fetch_wasabi_file(filename, output = output)
        
        f = open(output)
        
        assert f.read() == 'Small file used for testing purposes.\n'

        # Current directory default and overite
        os.chdir(tmpdir)
        
        fetch_wasabi_file(filename, overwrite= True)
        
        f = open(output)
        
        assert f.read() == 'Small file used for testing purposes.\n'

        # Test error when file exists, is different, and no overwrite
        file = open(output, "a")
        file.write("Append test line.\n")
        file.close()

        with pytest.raises(RuntimeError):
            fetch_wasabi_file(filename)

        # Succeeds with overwrite
        fetch_wasabi_file(filename, overwrite = True)

        # Zipped file
        fetch_wasabi_file(filename + ".zip", overwrite=True, unzip = True)
        f = open(output)
        assert f.read() == 'Small file used for testing purposes.\n'

        fetch_wasabi_file(filename + ".gz", overwrite=True, unzip = True, unzip_output = output)
        f = open(output)
        assert f.read() == 'Small file used for testing purposes.\n'

        # Checksum zipped file
        fetch_wasabi_file(filename + ".gz", unzip=True, checksum = 'c29015230d84e5e44e773c51c99b5911')

        # Fails if checksum not provided, file exists and overwrite = False
        with pytest.raises(RuntimeError):
            fetch_wasabi_file(filename + ".gz", unzip=True)

        # Test unzipped file is different than checksum
        file = open(output, "a")
        file.write("Append test line.\n")
        file.close()

        # Fails if overwrite is False
        with pytest.raises(RuntimeError):
            fetch_wasabi_file(filename + ".gz", unzip=True, checksum='c29015230d84e5e44e773c51c99b5911')

        # Succeeds if file is different, but overwrite = True
        fetch_wasabi_file(filename + ".gz", unzip=True, checksum = 'c29015230d84e5e44e773c51c99b5911', overwrite = True)

        # For multipart uploaded files
        fetch_wasabi_file('test_files/test_multipart_file.txt')
        fetch_wasabi_file('test_files/test_multipart_file.txt') # Already exists, but it's the same file, so it should succeed

        # Fetch only header
        hdr = fetch_wasabi_file_header(filename)
        assert hdr["ETag"] == '"c29015230d84e5e44e773c51c99b5911"'
