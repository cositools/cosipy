#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

# Get common version number (https://stackoverflow.com/a/7071358)
import re
VERSIONFILE="cosipy/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='cosipy',
      version = verstr,
      author='COSI Team',
      author_email='imc@umd.edu',
      url='https://github.com/cositools/cosipy',
      packages = find_packages(include=["cosipy", "cosipy.*"]),
      install_requires = ["histpy",
                          "mhealpy",
                          "scoords",
                          'astromodels',
                          'threeml',
                          'numba<=0.58.0',
                          'awscli'],
      description = "High-level analysis for the COSI telescope data",
      entry_points={"console_scripts":[
          "cosi-response = cosipy.response.FullDetectorResponse:cosi_response",
                              ]},
      
      long_description = long_description,
      long_description_content_type="text/markdown",
      )

