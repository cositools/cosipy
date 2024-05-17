from cosipy.util import fetch_wasabi_file
import tempfile
import os
from pathlib import Path
import pytest

def test_fetch_wasabi_file():

    with tempfile.TemporaryDirectory() as tmpdir:

        filename = 'test_file.txt'

        # Using output
        output = Path(tmpdir)/filename
        fetch_wasabi_file(filename, output = output)
        
        f = open(output)
        
        assert f.read() == 'Small file used for testing purposes.\n'

        # Current directory default and override
        os.chdir(tmpdir)
        
        fetch_wasabi_file(filename, override = True)
        
        f = open(filename)
        
        assert f.read() == 'Small file used for testing purposes.\n'

        # Test error when file exists and no override
        with pytest.raises(RuntimeError):
            fetch_wasabi_file(filename)
        
