# CLAUDE.md

Notes for Claude Code sessions working on this repo, in particular on the
relative-coordinates histogram IRF (`IRFRelativeHistUnpolarized`).

## Repo and branch workflow

- This is `israelmcmc-ai/cosipy`, a fork of `cositools/cosipy`. The working
  branch is `rel_irf_hist`, which feeds the upstream PR `cositools/cosipy#641`.
- **Don't commit directly to `rel_irf_hist`.** Put work on a new branch and
  open a PR with base `rel_irf_hist`. Unrelated side fixes (e.g. to
  `EnergySelector`, `DistanceSelector`, the chain selector) go on their own
  branch with a PR against `develop`.
- The maintainer often pushes to the same PR branch while you work (e.g.
  comment edits, scratch scripts). Always `git fetch` before pushing, and
  only rebase *your own unpushed* commits on top; never force-push over
  their commits. Resolve conflicts keeping their wording.
- If asked to "put X in a separate PR and leave branch Z as it was": create
  the new branch at Z's tip, `git reset --hard <old tip>` on Z, then
  `git push --force-with-lease origin Z`, and open the PR from the new
  branch.
- Closed-unmerged PRs in this fork (e.g. #7, #8, #10, #11) were closed on
  purpose; don't redo them unless asked.
- The maintainer prefers simple code: refinements that "don't matter much"
  get reverted. Verify a change actually moves the needle before adding
  complexity.

## Running the tests in a sandbox without astromodels

`import cosipy` pulls in `astromodels`/`threeML` (via
`cosipy/__init__.py` → `response` → `threeml`), which may fail to install
(`antlr4-python3-runtime` build error). Try `pip install astromodels threeML`
first; if that fails, install the light deps and load just the needed
submodules with stub packages:

```bash
pip install histpy scoords mhealpy h5py tqdm yayc pytest typing_extensions
```

Save as e.g. `$SCRATCH/load_relative_irf_hist.py` (not in the repo):

```python
import sys, types, importlib.util
ROOT = "/home/user/cosipy"  # adjust

def load(fullname, path):
    spec = importlib.util.spec_from_file_location(fullname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod

def stub_pkg(name, subdir):
    pkg = types.ModuleType(name)
    pkg.__path__ = [f"{ROOT}/{subdir}"]
    sys.modules[name] = pkg
    return pkg

def export(pkg, mod):
    for n in dir(mod):
        if not n.startswith('_'):
            setattr(pkg, n, getattr(mod, n))

stub_pkg('cosipy', 'cosipy')
stub_pkg('cosipy.util', 'cosipy/util')
load('cosipy.util.iterables', f'{ROOT}/cosipy/util/iterables.py')

pol = stub_pkg('cosipy.polarization', 'cosipy/polarization')
for m in ['conventions', 'polarization_angle', 'polarization_axis']:
    export(pol, load(f'cosipy.polarization.{m}', f'{ROOT}/cosipy/polarization/{m}.py'))

iface = stub_pkg('cosipy.interfaces', 'cosipy/interfaces')
for m in ['event', 'data_interface', 'event_selection', 'photon_parameters',
          'instrument_response_interface']:
    export(iface, load(f'cosipy.interfaces.{m}', f'{ROOT}/cosipy/interfaces/{m}.py'))

stub_pkg('cosipy.event_selection', 'cosipy/event_selection')
for m in ['time_selection', 'energy_selection', 'distance_selection']:
    load(f'cosipy.event_selection.{m}', f'{ROOT}/cosipy/event_selection/{m}.py')

stub_pkg('cosipy.response', 'cosipy/response')
load('cosipy.response.relative_coordinates', f'{ROOT}/cosipy/response/relative_coordinates.py')
load('cosipy.response.relative_irf_hist', f'{ROOT}/cosipy/response/relative_irf_hist.py')
```

Then run pytest from the same interpreter so the stubs stay in `sys.modules`:

```python
exec(open('load_relative_irf_hist.py').read())
import pytest, sys
sys.exit(pytest.main(['-v', 'tests/response/test_relative_irf_hist.py']))
```

The ~28 `RuntimeWarning: divide by zero` warnings from histpy come from
existing code and are harmless.

The sandbox's network policy blocks the Wasabi bucket
(`s3.us-west-1.wasabisys.com`), so the real response files
(`relative_hist_irf_from_nf_response.h5`,
`ResponseContinuum.area.relative.nonsparse_smoothing1p0.h5`) can't be
downloaded. Validate with synthetic histograms instead.

## `IRFRelativeHistUnpolarized` (`cosipy/response/relative_irf_hist.py`)

- 6D histogram with axes `[NuLambda, Ei, Epsilon, Phi, Theta, Zeta]`, where
  `Epsilon = (Em - Ei)/Ei`. `Ei` is usually a log-scaled axis.
- **Bin contents are per-bin effective area (cm²), not densities.** Both
  builders write it that way:
  - `scripts/IRFRelativeHist/relative_hist_irf_from_rsp.py`: counts ×
    EFF_AREA (MEGAlib simulation, `hist_simple` mode).
  - `scripts/IRFRelativeHist/relative_hist_irf_from_nf_response.py`:
    `tot_aeff · density · phase_space_cds · Ei_center·ΔEps` (`hist_nn` mode).

  `__init__` divides by the phase space (CDS volume and `Ei_center·ΔEps`)
  to get `_diff_aeff`.
- `_tot_aeff` is either `irf.project('NuLambda', 'Ei')` or a separate `aeff`
  histogram (usually finer). `from_h5` reads an `AEFF` group automatically
  if present. `_tot_aeff` is linearly interpolated in `Ei` at evaluation
  time.
- The tutorial is
  `docs/tutorials/spectral_fits/continuum_fit/grb/example_grb_fit_relative_hist_response.ipynb`
  (`irf_mode` = `hist_simple` / `hist_nn` / `nn`).

### Energy selections (`selections=EnergySelector(...)`)

- A tuple of selectors is OR'd via `EnergySelector.union`. `EnergySelector`
  (`cosipy/event_selection/energy_selection.py`) has
  `energy_ranges_keV`, `min/max_energy[_keV]`, `union`, `intersect`,
  `except_`.
- `_apply_energy_selection` scales `_tot_aeff` per `(NuLambda, Ei)` by the
  fraction of area with `Em` inside the cut. `_diff_aeff` is never modified.
  - **Same grid** (no separate `aeff`): the fraction is computed at irf's own
    `Ei` centers and the grid is not refined.
  - **Separate `aeff`**:
    1. `_refine_ei_edges` adds `Ei` edges at `E_cut / (1 + Epsilon)` for
       every Epsilon edge and center (the fraction's kinks), keeping all the
       original edges.
    2. `_regrid_ei` resamples `aeff` onto the refined grid using the axis's
       own `interp_weights`.
    3. irf's `(NuLambda, Ei, Epsilon)` projection is interpolated onto that
       grid one Epsilon center at a time.
    4. `_selection_fraction` evaluates the cut at each exact target `Ei`.

    So `_tot_aeff` can end up with more `Ei` bins than the `aeff` passed in.
- `_integrate_piecewise_linear` integrates the per-Epsilon density
  (content/ΔEps), linear between centers and flat beyond the first/last
  center. Integrating over the full range is **not** exactly the content
  sum when bins are non-uniform, so `content.sum()` is used as the total.

### Lessons from debugging narrow energy cuts

- The two changes that mattered:
  1. Evaluate the fraction at the target `Ei` instead of interpolating a
     fraction computed on irf's coarse grid.
  2. Refine the `aeff` grid. With `geomspace(50, 10000, 41)` and a
     495–505 keV cut, sampling only at bin centers made the integral over
     `Ei` about 2× too large; refining brings it within 0.2%.
- Integrating over `dEps` vs `dEm` at a fixed `Ei` is just a change of
  variables (numerically identical), so it's not a source of error.
- Interpolating content as a density across `Ei` (divide by native `Ei`,
  multiply by the target) was tried and reverted as not worth the
  complexity.
- A synthetic test whose Epsilon profile has the same shape and scale at
  every `Ei` can't catch `Ei`-interpolation problems. Use shapes that vary
  with `Ei` when testing those.

## histpy gotchas

- `Histogram.interp()` / `Axis.interp_weights()` on a `scale='log'` axis
  interpolate linearly in **log(x)** between bin centers, and clamp to the
  first/last center.
- `Histogram.copy()` doesn't deep-copy `Axis` objects, and
  `IRFRelativeHistUnpolarized.__init__` strips units from axes in place.
  So after constructing a model, the caller's original `irf`/`aeff`
  histograms already have unitless (keV) axes, even with `copy=True`.
- Projections sum contents over the dropped axes.

## Conventions

- Default to no code comments unless the "why" is non-obvious. Docstrings in
  this module are numpy-style.
- PR replies and review comments in this fork end with the Claude Code
  attribution footer.
