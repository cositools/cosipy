"""
Reading and writing NF model files in two formats:

* ``.pt``   - the original PyTorch pickle (``torch.save`` of a nested dict)
* ``.fits`` - a lossless FITS representation of the same nested dict
              (also ``.fit``, ``.fts`` and gzip-compressed ``.fits.gz``)

FITS layout (HDUVERS = FORMAT_VERSION)
--------------------------------------
Every HDU: TELESCOP, INSTRUME, CREATOR, DATE, LONGSTRN, HDUCLASS, HDUCLAS1='NF_MODEL',
HDUVERS (+ HDUCLAS2 in extensions), CHECKSUM/DATASUM.
HDU 0  PRIMARY   header only: NFTYPE, MODELVER, optional keywords, COMMENT and
                 HISTORY (conversion provenance: source file and SHA-256)
HDU 1  INDEX     binary table, one row per node of the nested dict (pre-order),
                 header keywords NNODES (rows) and NARRAYS (array extensions):
                   KEYPATH  path of the node, e.g. '/density_input/bins'
                            dict keys are separated by '/', list elements are
                            written '[i]', tuple elements '(i)'; inside dict keys
                            every character except A-Z a-z 0-9 _ . - ~ is
                            %-escaped (UTF-8), so KEYPATH is always plain ASCII
                   KIND     dict | odict | list | tuple | tensor | ndarray |
                            int | float | bool | str | jsonstr | none | npscalar
                   DTYPE    numpy dtype name, or byte-order string (e.g. '>f8')
                            for non-native numpy arrays
                   SHAPE    JSON list (tensor, ndarray)
                   HDU_NAME extension holding the array (tensor, ndarray) or the
                            state-dict metadata (odict)
                   VALUE    exact text form of scalars (repr for floats);
                            'requires_grad' for tensors with requires_grad=True
                   UNIT     optional physical unit (informational only)
HDU 2+ T_nnnnn   one IMAGE extension per torch tensor
       N_nnnnn   one IMAGE extension per numpy array
       M_nnnnn   BINTABLE with the ``_metadata`` of an OrderedDict state dict

Supported values: dict / OrderedDict with non-empty str keys, list, tuple,
torch.Tensor (plain, not subclasses), numpy.ndarray (plain), numpy scalars,
int, float, bool, str, None. Supported array dtypes: bool, uint8, int8, int16,
int32, int64, uint16, uint32, float16, float32, float64. Anything else
(bfloat16, complex, uint64, object/str arrays, nn.Parameter, MaskedArray,
namedtuple, torch.Size, ...) raises TypeError.

Tensors with Python attributes, dicts with attributes other than ``_metadata``,
and sparse/quantized tensors also raise TypeError instead of losing information.

Not preserved, by design (no effect on any computation): memory layout
(arrays are stored in C order) and storage sharing between tensors.
All values, including NaN sign and payload, are stored bit-exact
(NaN scalars are written as their raw bits, e.g. 'nan(0x7ff8000000000000)').

Command line
------------
python -m cosipy.response.ml.NFModelIO convert model.pt model.fits
python -m cosipy.response.ml.NFModelIO compare model.pt model.fits
"""

import argparse
import collections
import datetime
import gzip
import hashlib
import json
import os
import math
import re
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from urllib.parse import quote, unquote

import numpy as np
import torch
from astropy.io import fits

PathLike = Union[str, Path]

FORMAT_VERSION = "1.0.0"
HDUCLASS = "COSI"
HDUCLAS1 = "NF_MODEL"

_FITS_SUFFIXES = (".fits", ".fit", ".fts")
_FITS_MAGIC = b"SIMPLE  ="
_GZIP_MAGIC = b"\x1f\x8b"

# FITS images have no bool/int8/float16/uint16/uint32 without scaling tricks:
# store them in an exactly representable wider type and cast back on reading.
_STORAGE_DTYPE = {
    "bool": "uint8",
    "int8": "int16",
    "uint16": "int32",
    "uint32": "int64",
    "float16": "float32",
}
_SUPPORTED_DTYPES = {"bool", "uint8", "int8", "int16", "int32", "int64",
                     "uint16", "uint32", "float16", "float32", "float64"}

_NP_SCALAR_TYPES = {np.dtype(d).type for d in _SUPPORTED_DTYPES}

_INDEX_COLUMNS = ("KEYPATH", "KIND", "DTYPE", "SHAPE", "HDU_NAME", "VALUE", "UNIT")

# Keywords that header_meta must not set (structure, checksums and keywords written by this module).
_RESERVED_KEYWORDS = {"SIMPLE", "BITPIX", "NAXIS", "EXTEND", "XTENSION", "PCOUNT", "GCOUNT",
                      "BSCALE", "BZERO", "BUNIT", "EXTNAME", "TFIELDS", "END",
                      "TELESCOP", "INSTRUME", "CREATOR", "DATE", "LONGSTRN",
                      "HDUCLASS", "HDUCLAS1", "HDUCLAS2", "HDUVERS",
                      "CHECKSUM", "DATASUM", "NFTYPE", "MODELVER", "NNODES", "NARRAYS"}
_RESERVED_PATTERN = re.compile(r"^(NAXIS|TTYPE|TFORM|TUNIT|TDIM|TNULL|TSCAL|TZERO)\d+$")
_KEYWORD_PATTERN = re.compile(r"^[A-Z0-9_-]{1,8}$")          # standard FITS keyword, no HIERARCH


# ----------------------------------------------------------------------------
# Format detection and loading
# ----------------------------------------------------------------------------

def is_fits_file(path: PathLike) -> bool:
    """
    True if the file starts with a FITS header (also inside gzip). The content decides;
    the suffix (.fits/.fit/.fts, optionally .gz) is only used if the file cannot be read.
    """
    if not isinstance(path, (str, os.PathLike)):
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(len(_FITS_MAGIC))
        if head[:2] == _GZIP_MAGIC:
            with gzip.open(path, "rb") as f:
                head = f.read(len(_FITS_MAGIC))
        return head == _FITS_MAGIC
    except (OSError, ValueError, EOFError, zlib.error):
        name = str(path).lower()
        if name.endswith(".gz"):
            name = name[:-3]
        return name.endswith(_FITS_SUFFIXES)


def load_nf_model(path: PathLike) -> Dict[str, Any]:
    """Load an NF model file (.pt or FITS) and return the model dictionary."""
    if not isinstance(path, (str, os.PathLike)):          # file-like object: hand it to torch unchanged
        return torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    if is_fits_file(path):
        return read_nf_fits(path)
    return torch.load(str(path), map_location=torch.device('cpu'), weights_only=False)


# ----------------------------------------------------------------------------
# dict  <->  rows / arrays
# ----------------------------------------------------------------------------

def _escape_key(key: str) -> str:
    return quote(key, safe="")                      # keeps A-Z a-z 0-9 _ . - ~


def _unescape_token(token: str) -> str:
    return unquote(token, encoding="utf-8", errors="strict")


def _join(parent: str, token: str) -> str:
    return parent.rstrip("/") + "/" + token


def _split(path: str) -> List[str]:
    return [] if path == "/" else path[1:].split("/")


def _is_plain_str(s: str) -> bool:
    return s.isascii() and s.isprintable() and s == s.strip()


def _float_text(x: float) -> str:
    """Exact text for a Python float: repr, or the raw bits for NaN (keeps sign and payload)."""
    return repr(x) if not math.isnan(x) else f"nan(0x{struct.pack('>d', x).hex()})"


def _npscalar_text(x: np.generic) -> str:
    """Exact text for a numpy scalar: repr of the value, or the raw bits for NaN."""
    if x.dtype.kind == "f" and np.isnan(x):
        bits = int(x.view(np.dtype(f"u{x.dtype.itemsize}")))
        return f"nan(0x{bits:0{2 * x.dtype.itemsize}x})"
    return repr(x.item())


def _dtype_tag(dtype: np.dtype) -> str:
    """dtype name for native byte order, explicit byte-order string otherwise."""
    return dtype.name if dtype.isnative else dtype.str


def _row(keypath, kind, dtype="", shape="", hdu_name="", value="", unit=""):
    return {"KEYPATH": keypath, "KIND": kind, "DTYPE": dtype, "SHAPE": shape,
            "HDU_NAME": hdu_name, "VALUE": value, "UNIT": unit}


class _Flattener:
    """Walks the nested dict in pre-order and collects INDEX rows and arrays."""

    def __init__(self, units: Optional[Mapping[str, str]] = None):
        self.rows: List[Dict[str, str]] = []
        self.arrays: List[Tuple[str, str, str, np.ndarray, str]] = []   # name, keypath, kind, array, unit
        self.metas: List[Tuple[str, str, Mapping]] = []                 # name, keypath, metadata
        self._units = dict(units or {})
        self._numeric_paths = set()          # nodes that may carry a unit
        self._n = {"T": 0, "N": 0, "M": 0}

    def _hdu_name(self, prefix: str) -> str:
        self._n[prefix] += 1
        return f"{prefix}_{self._n[prefix]:05d}"

    def _add_array(self, path: str, kind: str, arr: np.ndarray, value: str = ""):
        if arr.dtype.name not in _SUPPORTED_DTYPES:
            raise TypeError(f"{path}: dtype {arr.dtype} cannot be stored in FITS.")
        name = self._hdu_name("T" if kind == "tensor" else "N")
        unit = self._units.get(path, "")
        self._numeric_paths.add(path)
        self.arrays.append((name, path, kind, arr, unit))
        self.rows.append(_row(path, kind, dtype=_dtype_tag(arr.dtype), shape=json.dumps(list(arr.shape)),
                              hdu_name=name, value=value, unit=unit))

    def check_units(self):
        unknown = sorted(set(self._units) - self._numeric_paths)
        if unknown:
            raise ValueError(f"Units given for KEYPATHs that do not exist or are not numeric: {unknown}")

    def walk(self, obj: Any, path: str = "/"):
        unit = self._units.get(path, "")
        if type(obj) in (int, float, torch.Tensor, np.ndarray) or type(obj) in _NP_SCALAR_TYPES:
            self._numeric_paths.add(path)

        if type(obj) in (dict, collections.OrderedDict):
            kind = "odict" if type(obj) is collections.OrderedDict else "dict"
            extra = sorted(set(getattr(obj, "__dict__", {})) - {"_metadata"})
            if extra:
                raise TypeError(f"{path}: attributes {extra} on the dict cannot be stored in FITS.")
            name = ""
            meta = getattr(obj, "_metadata", None)
            if meta is not None:
                name = self._hdu_name("M")
                self.metas.append((name, path, meta))
            self.rows.append(_row(path, kind, hdu_name=name))
            for key, value in obj.items():
                if type(key) is not str or key == "":
                    raise TypeError(f"{path}: only non-empty string dict keys are supported, got {key!r}.")
                self.walk(value, _join(path, _escape_key(key)))

        elif type(obj) in (list, tuple):
            kind, fmt = ("list", "[{}]") if type(obj) is list else ("tuple", "({})")
            self.rows.append(_row(path, kind))
            for i, value in enumerate(obj):
                self.walk(value, _join(path, fmt.format(i)))

        elif type(obj) is torch.Tensor:
            if obj.layout is not torch.strided:
                raise TypeError(f"{path}: tensors with layout {obj.layout} cannot be stored in FITS.")
            if obj.__dict__:
                raise TypeError(f"{path}: Python attributes {sorted(obj.__dict__)} on the tensor "
                                f"cannot be stored in FITS.")
            try:
                arr = obj.detach().cpu().contiguous().numpy()
            except (TypeError, RuntimeError) as e:
                raise TypeError(f"{path}: tensor of dtype {obj.dtype} cannot be stored in FITS ({e}).") from e
            self._add_array(path, "tensor", arr, value="requires_grad" if obj.requires_grad else "")

        elif type(obj) is np.ndarray:
            self._add_array(path, "ndarray", np.array(obj, order="C", copy=True))

        elif type(obj) in _NP_SCALAR_TYPES:          # before float: np.float64 is a float subclass
            dtype = obj.dtype.name
            self.rows.append(_row(path, "npscalar", dtype=dtype, value=_npscalar_text(obj), unit=unit))

        elif obj is None:
            self.rows.append(_row(path, "none"))
        elif type(obj) is bool:
            self.rows.append(_row(path, "bool", value=repr(obj)))
        elif type(obj) is int:
            self.rows.append(_row(path, "int", value=repr(obj), unit=unit))
        elif type(obj) is float:
            self.rows.append(_row(path, "float", value=_float_text(obj), unit=unit))
        elif type(obj) is str:
            if _is_plain_str(obj):
                self.rows.append(_row(path, "str", value=obj))
            else:
                self.rows.append(_row(path, "jsonstr", value=json.dumps(obj)))
        else:
            raise TypeError(f"{path}: values of type {type(obj).__module__}.{type(obj).__name__} "
                            f"are not supported.")


def _parse_bool(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"Invalid boolean value {value!r} in INDEX.")
    return value == "True"


def _parse_scalar(kind: str, dtype: str, value: str) -> Any:
    if kind == "none":
        return None
    if kind == "bool":
        return _parse_bool(value)
    if kind == "int":
        return int(value)
    if kind == "float":
        if value.startswith("nan(0x"):
            return struct.unpack(">d", bytes.fromhex(value[6:-1]))[0]
        return float(value)
    if kind == "str":
        return value
    if kind == "jsonstr":
        return json.loads(value)
    if kind == "npscalar":
        np_type = np.dtype(dtype).type
        if dtype == "bool":
            return np_type(_parse_bool(value))
        if np.dtype(dtype).kind in "iu":
            return np_type(int(value))
        if value.startswith("nan(0x"):
            size = np.dtype(dtype).itemsize
            return np.array(int(value[4:-1], 16), dtype=f"u{size}").view(np.dtype(dtype))[()]
        return np_type(float(value))
    raise ValueError(f"Unknown KIND {kind!r}.")


def _unflatten(rows: List[Dict[str, str]],
               get_array: Callable[[str, str, Tuple[int, ...]], np.ndarray],
               get_meta: Callable[[str], Mapping]) -> Any:
    """Rebuild the nested object from the INDEX rows (pre-order)."""
    containers = {"dict", "odict", "list", "tuple"}
    nodes: Dict[str, Any] = {}      # keypath -> [kind, children(list of (token, keypath)), hdu_name]
    values: Dict[str, Any] = {}     # keypath -> leaf value
    root = None

    for r in rows:
        path, kind = r["KEYPATH"], r["KIND"]
        if path in nodes or path in values:
            raise ValueError(f"Duplicate KEYPATH {path!r} in INDEX.")
        if root is None:
            if path != "/":
                raise ValueError("The first INDEX row must be the root '/'.")
            root = path
        else:
            tokens = _split(path)
            parent = "/" + "/".join(tokens[:-1]) if len(tokens) > 1 else "/"
            if parent not in nodes:
                raise ValueError(f"Parent of {path!r} missing in INDEX.")
            nodes[parent][1].append((tokens[-1], path))

        if kind in containers:
            nodes[path] = [kind, [], r["HDU_NAME"]]
        elif kind in ("tensor", "ndarray"):
            shape = tuple(json.loads(r["SHAPE"]))
            arr = get_array(r["HDU_NAME"], r["DTYPE"], shape)
            if kind == "tensor":
                tensor = torch.from_numpy(arr)
                if r["VALUE"] == "requires_grad":
                    tensor.requires_grad_(True)
                elif r["VALUE"] != "":
                    raise ValueError(f"Invalid VALUE {r['VALUE']!r} for tensor {path!r}.")
                values[path] = tensor
            else:
                values[path] = arr
        else:
            values[path] = _parse_scalar(kind, r["DTYPE"], r["VALUE"])

    def build(path: str) -> Any:
        if path in values:
            return values[path]
        kind, children, name = nodes[path]
        if kind in ("dict", "odict"):
            out = collections.OrderedDict() if kind == "odict" else {}
            for token, child in children:
                out[_unescape_token(token)] = build(child)
            if name:
                out._metadata = get_meta(name)
            return out
        items = []
        bracket = "[]" if kind == "list" else "()"
        for i, (token, child) in enumerate(children):
            if token != f"{bracket[0]}{i}{bracket[1]}":
                raise ValueError(f"Unexpected element {token!r} in {kind} {path!r}.")
            items.append(build(child))
        return items if kind == "list" else tuple(items)

    if root is None:
        raise ValueError("Empty INDEX table.")
    return build(root)


# ----------------------------------------------------------------------------
# FITS writing / reading
# ----------------------------------------------------------------------------

def _infer_nftype(model: Mapping) -> str:
    if "rate_input" in model:
        return "NF_BACKGROUND"
    if "area_input" in model:
        return "NF_RESPONSE"
    if "norm_dimensions" in model and "is_polarized" in model:
        return "NF_NORM_DENSITY"
    if "norm_dimensions" in model:
        return "NF_BKG_NORM_DENSITY"
    return "UNKNOWN"


def _creator() -> str:
    try:
        from cosipy import __version__
        version = str(__version__)
    except Exception:
        version = "unknown"
    return f"cosipy {version} NFModelIO {FORMAT_VERSION}"


def _common_keywords(header: fits.Header, telescop: str, instrume: str, creator: str, date: str,
                     hduclas2: Optional[str] = None):
    header["TELESCOP"] = (telescop, "Mission name")
    header["INSTRUME"] = (instrume, "Instrument name")
    header["CREATOR"] = (creator, "Software that created this file")
    header["DATE"] = (date, "File creation date (UTC)")
    header["LONGSTRN"] = ("OGIP 1.0", "The OGIP long string convention may be used")
    header["HDUCLASS"] = (HDUCLASS, "Organisation defining the HDU format")
    header["HDUCLAS1"] = (HDUCLAS1, "Normalizing-flow model file")
    if hduclas2 is not None:
        header["HDUCLAS2"] = (hduclas2, "Content of this extension")
    header["HDUVERS"] = (FORMAT_VERSION, "Version of the NF model FITS layout")


def _str_column(name: str, values: List[str]) -> fits.Column:
    for v in values:
        if not v.isascii():
            raise ValueError(f"Non-ASCII text cannot be written to column {name}: {v!r}")
    width = max([1] + [len(v) for v in values])
    return fits.Column(name=name, format=f"{width}A", array=np.array(values, dtype=f"U{width}"))


def _build_hdulist(model: Mapping[str, Any], telescop: str, instrume: str, nftype: Optional[str],
                   description: Optional[str], header_meta: Optional[Mapping[str, Any]],
                   units: Optional[Mapping[str, str]], history: Optional[List[str]]) -> fits.HDUList:
    flat = _Flattener(units)
    flat.walk(model)
    flat.check_units()
    creator = _creator()
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    common = (telescop, instrume, creator, date)

    primary = fits.PrimaryHDU()
    hdr = primary.header
    _common_keywords(hdr, *common)
    hdr["NFTYPE"] = (nftype or _infer_nftype(model), "Type of NF model")
    if "version" in model:
        hdr["MODELVER"] = (str(model["version"]), "Model version used by the cosipy loaders")
    for key, value in (header_meta or {}).items():
        upper = str(key).upper()
        if upper in ("COMMENT", "HISTORY"):
            for line in ([value] if isinstance(value, str) else list(value)):
                (hdr.add_comment if upper == "COMMENT" else hdr.add_history)(str(line))
            continue
        if not _KEYWORD_PATTERN.match(upper) or upper == "CONTINUE":
            raise ValueError(f"Header keyword {key!r} is not a standard FITS keyword (1-8 characters A-Z 0-9 _ -).")
        if upper in _RESERVED_KEYWORDS or _RESERVED_PATTERN.match(upper):
            raise ValueError(f"Header keyword {key} is reserved.")
        hdr[upper] = value
    if description:
        for line in description.splitlines() or [""]:
            hdr.add_comment(line)
    for line in history or []:
        hdr.add_history(line)

    columns = [_str_column(name, [r[name] for r in flat.rows]) for name in _INDEX_COLUMNS]
    index = fits.BinTableHDU.from_columns(columns, name="INDEX")
    _common_keywords(index.header, *common, "INDEX")
    index.header["NNODES"] = (len(flat.rows), "Number of rows (nodes of the dictionary)")
    index.header["NARRAYS"] = (len(flat.arrays), "Number of tensor/array extensions")
    index.header.add_comment("One row per node of the model dictionary (pre-order).")
    index.header.add_comment("KEYPATH: '/'-separated path; list items '[i]', tuple items '(i)'.")
    index.header.add_comment("Tensors/arrays are stored in the extension named in HDU_NAME.")
    index.header.add_comment("UNIT is informational per row, not a column unit (TUNITn).")

    hdus = [primary, index]
    for name, keypath, kind, arr, unit in flat.arrays:
        stored = arr.astype(_STORAGE_DTYPE.get(arr.dtype.name, arr.dtype.name), copy=True)
        if stored.ndim == 0:
            stored = stored.reshape(1)
        image = fits.ImageHDU(data=stored if stored.size > 0 else None, name=name)
        _common_keywords(image.header, *common, "TENSOR" if kind == "tensor" else "NDARRAY")
        image.header["KEYPATH"] = (keypath, "Path in the model dictionary")
        image.header["DTYPE"] = (_dtype_tag(arr.dtype), "Original dtype")
        image.header["SHAPE"] = (json.dumps(list(arr.shape)), "Original shape")
        if unit:
            image.header["BUNIT"] = (unit, "Physical unit of the array values")
        hdus.append(image)

    for name, keypath, meta in flat.metas:
        prefixes = list(meta.keys())
        entries = [json.dumps(meta[p]) for p in prefixes]
        table = fits.BinTableHDU.from_columns(
            [_str_column("PREFIX", [str(p) for p in prefixes]), _str_column("META", entries)], name=name)
        _common_keywords(table.header, *common, "STATE_DICT_META")
        table.header["KEYPATH"] = (keypath, "Path of the state dict in the model dictionary")
        hdus.append(table)

    return fits.HDUList(hdus)


def save_nf_model(model: Mapping[str, Any], path: PathLike, *,
                  telescop: str = "COSI", instrume: str = "COSI",
                  nftype: Optional[str] = None, description: Optional[str] = None,
                  header_meta: Optional[Mapping[str, Any]] = None,
                  units: Optional[Mapping[str, str]] = None,
                  history: Optional[List[str]] = None,
                  overwrite: bool = False, verify: bool = True) -> Path:
    """
    Write an NF model dictionary (as passed to torch.save) to a FITS file.

    The file is first written under a temporary name in the same folder and only
    moved to `path` once it is complete (and, with verify=True, read back and found
    identical), so a failure never leaves a half-written or deleted file behind.

    Parameters
    ----------
    model       : the model dictionary ('version', 'density_input', ...)
    path        : output file (.fits, or .fits.gz for gzip compression)
    telescop    : TELESCOP keyword (mission)
    instrume    : INSTRUME keyword (instrument)
    nftype      : NFTYPE keyword; inferred from the dictionary keys if None
    description : free text (ASCII), written as COMMENT cards in the primary header
    header_meta : additional primary-header keywords, {KEY: value} or {KEY: (value, comment)},
                  e.g. ORIGIN or CALDB keywords; standard 1-8 character names only, keywords
                  written by this module are reserved; COMMENT/HISTORY take a str or list of str
    units       : optional {KEYPATH: unit} written to the UNIT column (and BUNIT of
                  array extensions), e.g. {"/rate_input/start_time": "s"}; informational only;
                  only numeric entries can have a unit, unknown KEYPATHs raise ValueError
    history     : lines written as HISTORY cards in the primary header (provenance)
    overwrite   : overwrite an existing file
    verify      : read the written file back and require an exact match
    """
    if type(model) not in (dict, collections.OrderedDict):
        raise TypeError("The model must be a dict.")

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists (use overwrite=True).")

    hdulist = _build_hdulist(model, telescop, instrume, nftype, description, header_meta, units, history)

    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")   # keeps the suffix, so .gz still compresses
    try:
        hdulist.writeto(tmp, overwrite=True, checksum=True)
        if verify:
            diffs = compare_nf_models(model, read_nf_fits(tmp))
            if diffs:
                raise ValueError(f"Round trip to {path} is not exact:\n  " + "\n  ".join(diffs[:20]))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def _to_str(value: Any) -> str:
    return value.decode("ascii") if isinstance(value, bytes) else str(value)


def read_nf_fits(path: PathLike) -> Dict[str, Any]:
    """Read an NF model FITS file back into the original model dictionary."""
    with fits.open(path, memmap=False, checksum=True) as hdul:
        primary = hdul[0].header
        if primary.get("HDUCLAS1") != HDUCLAS1:
            raise ValueError(f"{path} is not an NF model FITS file (HDUCLAS1 != {HDUCLAS1}).")
        file_major = int(str(primary.get("HDUVERS", "0")).split(".")[0])
        if file_major > int(FORMAT_VERSION.split(".")[0]):
            raise ValueError(f"{path} uses NF FITS layout {primary.get('HDUVERS')}, "
                             f"this cosipy supports up to {FORMAT_VERSION}.")

        for i, hdu in enumerate(hdul):
            if "CHECKSUM" in hdu.header and hdu.verify_checksum() == 0:
                raise ValueError(f"{path}: CHECKSUM of HDU {i} ({hdu.name}) does not match, file is corrupted.")
            if "DATASUM" in hdu.header and hdu.verify_datasum() == 0:
                raise ValueError(f"{path}: DATASUM of HDU {i} ({hdu.name}) does not match, file is corrupted.")

        positions = {hdu.name: i for i, hdu in enumerate(hdul)}
        index_hdu = hdul[positions["INDEX"]]
        table = index_hdu.data
        names = ("KEYPATH", "KIND", "DTYPE", "SHAPE", "HDU_NAME", "VALUE")
        columns = {n: [_to_str(v) for v in table.field(n)] for n in names}
        rows = [{n: columns[n][i] for n in names} for i in range(len(table))]

        n_arrays = sum(r["KIND"] in ("tensor", "ndarray") for r in rows)
        if index_hdu.header.get("NNODES", len(rows)) != len(rows) or \
           index_hdu.header.get("NARRAYS", n_arrays) != n_arrays:
            raise ValueError(f"{path}: INDEX is inconsistent with NNODES/NARRAYS, file is incomplete.")

        def get_array(name: str, dtype: str, shape: Tuple[int, ...]) -> np.ndarray:
            data = hdul[positions[name]].data
            if data is None:
                return np.empty(shape, dtype=np.dtype(dtype))
            return np.array(data, dtype=np.dtype(dtype)).reshape(shape)   # own memory, requested byte order

        def get_meta(name: str) -> collections.OrderedDict:
            meta = hdul[positions[name]].data
            if meta is None or len(meta) == 0:
                return collections.OrderedDict()
            return collections.OrderedDict(
                (_to_str(p), json.loads(_to_str(m))) for p, m in zip(meta.field("PREFIX"), meta.field("META")))

        model = _unflatten(rows, get_array, get_meta)

    if not isinstance(model, dict):
        raise ValueError(f"{path}: the root of the model must be a dict.")
    return model


# ----------------------------------------------------------------------------
# Comparison and conversion
# ----------------------------------------------------------------------------

def _tname(t: type) -> str:
    return f"{t.__module__}.{t.__qualname__}"


def _as_numpy(x: Any) -> np.ndarray:
    """C-ordered numpy view/copy of a plain tensor or array, for bitwise comparison."""
    if type(x) is torch.Tensor:
        if x.layout is not torch.strided:
            raise TypeError(f"layout {x.layout} not supported")
        return x.detach().cpu().contiguous().numpy()
    if x.dtype.hasobject:
        raise TypeError("object arrays cannot be compared bitwise")
    return np.array(x, order="C")        # not ascontiguousarray: that would turn 0-d into (1,)


def _value_difference(x: np.ndarray, y: np.ndarray) -> str:
    """Message for arrays of equal dtype and shape whose bytes differ."""
    msg = "values differ"
    size = x.dtype.itemsize
    if x.size and size in (1, 2, 4, 8):
        unequal = x.reshape(-1).view(f"u{size}") != y.reshape(-1).view(f"u{size}")
        first = np.unravel_index(int(np.argmax(unequal)), x.shape)
        msg += (f" ({int(np.count_nonzero(unequal))} of {x.size} elements, "
                f"first at index {tuple(int(i) for i in first)})")
    return msg


def compare_nf_models(a: Any, b: Any, path: str = "/") -> List[str]:
    """
    Exact recursive comparison. Returns a list of differences (empty if identical).

    * Types must match exactly at every node (dict vs OrderedDict, list vs tuple,
      int vs float vs bool, np.float32 vs np.float64).
    * Dicts: same keys (and key types) in the same order, same state-dict ``_metadata`` (compared
      recursively with the same rules), same extra attributes.
    * Tensors/arrays: same dtype (incl. byte order), shape, bits and requires_grad.
    * Python floats and numpy scalars: same bits (so -0.0 != 0.0, NaN sign/payload count).
    * Fails closed: values of any other type, and tensors/arrays that cannot be
      compared bitwise (e.g. bfloat16, sparse, object arrays), are reported as differences.

    Not compared, by design: memory layout, storage sharing, the device of a tensor
    (both sides are moved to the CPU) and ``.grad``. Self-referencing containers raise
    RecursionError (they cannot occur in torch-saved models).
    """
    if type(a) is not type(b):
        return [f"{path}: type {_tname(type(a))} != {_tname(type(b))}"]

    diffs: List[str] = []
    kind = type(a)

    if kind in (dict, collections.OrderedDict):
        if list(a.keys()) != list(b.keys()):
            only_a = [k for k in a if k not in b]
            only_b = [k for k in b if k not in a]
            diffs.append(f"{path}: keys differ (only in first: {only_a}, only in second: {only_b}"
                         + (", order differs" if not (only_a or only_b) else "") + ")")
        elif [type(k) for k in a] != [type(k) for k in b]:          # e.g. 1 vs True vs 1.0
            diffs.append(f"{path}: key types differ")
        for k in a:
            if k in b:
                diffs += compare_nf_models(a[k], b[k], _join(path, _escape_key(k) if isinstance(k, str) else repr(k)))
        # '@' is %-escaped inside real keys, so these paths cannot collide with dict entries
        diffs += compare_nf_models(getattr(a, "_metadata", None), getattr(b, "_metadata", None),
                                   _join(path, "@metadata"))
        diffs += _compare_attributes(a, b, path)

    elif kind in (list, tuple):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
        fmt = "[{}]" if kind is list else "({})"
        for i, (x, y) in enumerate(zip(a, b)):
            diffs += compare_nf_models(x, y, _join(path, fmt.format(i)))

    elif kind in (torch.Tensor, np.ndarray):
        if kind is torch.Tensor and a.requires_grad != b.requires_grad:
            diffs.append(f"{path}: requires_grad {a.requires_grad} != {b.requires_grad}")
        try:
            x, y = _as_numpy(a), _as_numpy(b)
        except (TypeError, RuntimeError) as e:
            diffs.append(f"{path}: cannot be compared bitwise ({e})")
        else:
            if x.dtype != y.dtype:
                diffs.append(f"{path}: dtype {x.dtype.str} != {y.dtype.str}")
            elif x.shape != y.shape:
                diffs.append(f"{path}: shape {x.shape} != {y.shape}")
            elif x.tobytes() != y.tobytes():
                diffs.append(f"{path}: {_value_difference(x, y)}")
        if kind is torch.Tensor:
            diffs += _compare_attributes(a, b, path)

    elif kind is float:
        if struct.pack("<d", a) != struct.pack("<d", b):
            diffs.append(f"{path}: {_float_text(a)} != {_float_text(b)}")

    elif kind in _NP_SCALAR_TYPES:
        if a.tobytes() != b.tobytes():
            diffs.append(f"{path}: {_npscalar_text(a)} != {_npscalar_text(b)}")

    elif kind in (type(None), bool, int, str, bytes):
        if a != b:
            diffs.append(f"{path}: {a!r} != {b!r}")

    else:
        diffs.append(f"{path}: values of type {_tname(kind)} cannot be compared")
    return diffs


def _compare_attributes(a: Any, b: Any, path: str) -> List[str]:
    """Python attributes on dicts/tensors (other than a state dict's _metadata)."""
    attrs_a = {k: v for k, v in getattr(a, "__dict__", {}).items() if k != "_metadata"}
    attrs_b = {k: v for k, v in getattr(b, "__dict__", {}).items() if k != "_metadata"}
    if not attrs_a and not attrs_b:
        return []
    return compare_nf_models(attrs_a, attrs_b, _join(path, "@attributes"))


def convert_pt_to_fits(pt_path: PathLike, fits_path: PathLike, verify: bool = True,
                       **save_kwargs) -> Path:
    """
    Convert a .pt model file to FITS. With verify=True (default) the FITS file is read
    back and must be identical to what torch.load returns for the .pt file.
    Uses torch.load(weights_only=False): only convert trusted .pt files.
    """
    model = torch.load(str(pt_path), map_location=torch.device('cpu'), weights_only=False)
    sha256 = hashlib.sha256()
    with open(pt_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    history = list(save_kwargs.pop("history", None) or [])
    history.append(f"Converted from {Path(pt_path).name} (SHA-256 {sha256.hexdigest()}) by {_creator()}")
    return save_nf_model(model, fits_path, verify=verify, history=history, **save_kwargs)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cosipy.response.ml.NFModelIO",
                                     description="Convert and compare NF model files (.pt / FITS). "
                                                 "Reading .pt files uses pickle: only use trusted files.")
    sub = parser.add_subparsers(dest="command", required=True)

    conv = sub.add_parser("convert", help="convert a .pt model file to FITS")
    conv.add_argument("input")
    conv.add_argument("output")
    conv.add_argument("--overwrite", action="store_true")
    conv.add_argument("--no-verify", action="store_true", help="skip the exact round-trip check")
    conv.add_argument("--telescop", default="COSI")
    conv.add_argument("--instrume", default="COSI")
    conv.add_argument("--nftype", default=None)
    conv.add_argument("--description", default=None)
    conv.add_argument("--unit", action="append", default=[], metavar="KEYPATH=UNIT",
                      help="unit for one entry, e.g. /rate_input/start_time=s (repeatable)")
    conv.add_argument("--keyword", action="append", default=[], metavar="KEY=VALUE",
                      help="extra primary-header keyword (string value), e.g. ORIGIN=JGU-Mainz (repeatable)")

    comp = sub.add_parser("compare", help="compare two model files (.pt or FITS) exactly")
    comp.add_argument("first")
    comp.add_argument("second")

    args = parser.parse_args(argv)

    if args.command == "convert":
        def pairs(items, option):
            for item in items:
                if "=" not in item:
                    parser.error(f"{option} expects KEY=VALUE, got {item!r}")
            return dict(item.split("=", 1) for item in items)
        units = pairs(args.unit, "--unit")
        keywords = pairs(args.keyword, "--keyword")
        out = convert_pt_to_fits(args.input, args.output, verify=not args.no_verify,
                                 overwrite=args.overwrite, telescop=args.telescop, instrume=args.instrume,
                                 nftype=args.nftype, description=args.description, units=units,
                                 header_meta=keywords)
        print(f"Wrote {out}" + ("" if args.no_verify else " (round trip verified: identical)"))
        return 0

    diffs = compare_nf_models(load_nf_model(args.first), load_nf_model(args.second))
    if diffs:
        print(f"{len(diffs)} difference(s):")
        for d in diffs[:50]:
            print("  " + d)
        return 1
    print("Identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
