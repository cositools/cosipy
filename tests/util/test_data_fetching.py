from cosipy.util import fetch_wasabi_file

def test_fetch_wasabi_file():

    fetch_wasabi_file('test_file.txt', override = True)

    f = open('test_file.txt')

    assert f.read() == 'Small file used for testing purposes.\n'

