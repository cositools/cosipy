"""Environment stability check for local Jupyter kernels.

`run_environment_check()` executes each potentially fragile component of this pipeline
(imports, numba serial jit, numba parallel jit, torch, torch+numba together, mhealpy
plotting) in its OWN throwaway subprocess and reports PASS / CRASH per component.  A
component that would kill a Jupyter kernel (segfault, OpenMP abort, LLVM error) only
kills its subprocess here, so this check is always safe to run - and its report says
exactly which component is responsible on a machine where the notebook dies.
"""

from __future__ import annotations

import signal
import subprocess
import sys

_TESTS = [
    ("scientific imports (numpy/scipy/astropy/healpy/mhealpy)", r"""
import numpy, scipy, astropy, healpy, mhealpy
print("OK")
"""),
    ("numba serial jit compile + run", r"""
import os
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
import numpy as np
from numba import njit
@njit(cache=False)
def f(x):
    s = 0.0
    for i in range(x.shape[0]):
        s += np.sin(x[i])
    return s
assert np.isfinite(f(np.linspace(0, 1, 10000)))
print("OK")
"""),
    ("numba parallel jit (workqueue layer)", r"""
import os
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
import numpy as np
from numba import njit, prange, get_thread_id
import numba
@njit(parallel=True)
def f(x, scratch):
    out = np.empty(x.shape[0])
    for p in prange(x.shape[0]):
        d = scratch[get_thread_id()]
        acc = 0.0
        for i in range(d.shape[0]):
            d[i] = np.sin(x[p] + 0.001 * i)
            acc += d[i]
        out[p] = acc
    return out
scratch = np.empty((numba.get_num_threads(), 256))
r = f(np.linspace(0, 1, 4096), scratch)
assert np.isfinite(r).all()
print("layer:", numba.threading_layer())
print("OK")
"""),
    ("torch import + tiny forward/backward", r"""
import torch
net = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
x = torch.randn(16, 8)
loss = net(x).sum()
loss.backward()
print("OK")
"""),
    ("torch AND numba parallel in one process", r"""
import os
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
import torch
import numpy as np
from numba import njit, prange
@njit(parallel=True)
def f(x):
    out = np.empty(x.shape[0])
    for p in prange(x.shape[0]):
        out[p] = np.sin(x[p])
    return out
r = f(np.linspace(0, 1, 4096))
net = torch.nn.Linear(4, 1)
_ = net(torch.randn(8, 4)).sum()
assert np.isfinite(r).all()
print("OK")
"""),
    ("matplotlib + mhealpy plot (Agg backend)", r"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, healpy as hp
import matplotlib.pyplot as plt
from mhealpy import HealpixMap
m = HealpixMap(data=np.arange(hp.nside2npix(16), dtype=float), scheme="NESTED",
               coordsys="G")
fig = plt.figure()
ax = fig.add_subplot(projection="mollview")
m.plot(ax, ax_kw={'coord': 'G'}, coord="C", cbar=False)
plt.close(fig)
print("OK")
"""),
]


def run_environment_check(timeout_s: float = 300.0) -> dict:
    """Run every component test in its own subprocess; print and return the report."""
    report = {}
    print("environment stability check (each test runs in a disposable subprocess)")
    print("-" * 74)
    for name, code in _TESTS:
        try:
            res = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                 text=True, timeout=timeout_s)
            if res.returncode == 0 and "OK" in res.stdout:
                status = "PASS"
                detail = res.stdout.replace("OK", "").strip()
            else:
                status = "CRASH"
                rc = res.returncode
                sig = ""
                if rc is not None and rc < 0:
                    try:
                        sig = f" (signal {signal.Signals(-rc).name})"
                    except ValueError:
                        sig = f" (signal {-rc})"
                tail = (res.stderr or "").strip().splitlines()[-2:]
                detail = f"returncode {rc}{sig}; " + " | ".join(tail)
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
            detail = f"no result within {timeout_s:.0f} s"
        report[name] = {"status": status, "detail": detail}
        extra = f"  [{detail}]" if (detail and status != "PASS") else ""
        print(f"{status:8s} {name}{extra}")
    print("-" * 74)
    if all(v["status"] == "PASS" for v in report.values()):
        print("all components pass in isolation on this machine.")
    else:
        print("a failing line above is the component that kills the kernel; the")
        print("pipeline avoids the parallel kernel automatically when its probe fails,")
        print("and INR_ANGULAR_ENGINE='serial' or 'numpy' forces safer engines.")
    return report
