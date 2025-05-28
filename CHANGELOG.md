# Changelog

## Version 0.4.0

Under development. 

Developers: please keep track of notable changes here.

### New FullDetectorResponse 

HDF5 response file format has changed substantially.  The new format,
which is *not* compatible with the .h5 response format from prior
releases, features
  - internal compression that decompresses chunks ~2x faster than gzip
    while offering comparable compression on disk
  - much faster conversion from .rsp.gz to .h5, with greatly reduced
    memory usage
  - ability to convert from .rsp.gz to .h5 and back to 
    .rsp.gz without losing any of the original header information
	(which facilitates creating lower-resolution versions of large
	responses)
  - separation of raw counts from effective area, which allows
    FullDetectorResponse to export the latter as needed and
    enables future performance improvements for code that
	needs to compute and average PSRs.  The dtype of the effective 
    area, and hence the values returned when computing PSRs etc.,
	can now be set at load time; hence, the same response can be
	used in float32 or float64 with no additional overhead.
  - automatic "good chunks" chunking of .h5 responses at creation time
  - HDF5 tweaks to avoid writing timestamps, so that (with support
    from a forthcoming release of histpy) the MD5 signature of an
	.h5 response does not change every time it is written.
	
.rsp.gz conversion has been broken out from FullDetectorResponse into
its own class, RspConverter, which simplifies the FullDetectorResponse
code and in particular its open() interface.

The new response code deprecates and removes the following
response-related functionality:
  - sparse response format
  - reading spectrum for effective area computation from a file
     (was broken/bit-rotted in DC3 release)
  - miniDC2 format support (unused since DC2)

### New wasabi location for development files

To support file format changes in the develop branch that are
incompatible with DC3 and cosipy versions <= 0.3, we have established
a "COSI-SMEX/develop" tree in the public wasabi bucket.  Currently,
this tree holds the new-format .h5 detector responses and the
corresponding .rsp.gz files.  The DC3 tutorials have been updated to
use this new tree for files whose format has changed since DC3.



## Version 0.2.x

Version 0.2.0 is the first released version of the "new" cosipy. It was reimplemented from scratch based on the "old" cosipy, now called "[cosipy classic](https://github.com/cositools/mirror-cosipy-classic)". It is the version used for [Data Challenge 2](https://github.com/cositools/cosi-data-challenge-2). Versions 0.2.x have bug fixes for version 0.2.0 but keep backward compatibility and do not add new features.
