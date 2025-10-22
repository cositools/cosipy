#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import yaml
from cosipy.util import fetch_wasabi_file


@dataclass
class _Cfg:
    wasabi_key: str
    local_fits: str
    time_col: str
    period: float
    nbins_profile: int
    nbins_phase: int
    nbins_time: int
    n_segments_stat: int
    fig_size: tuple[float, float]
    save_path: str | None
    dpi: int
    tight_layout: bool


class PulsarAnalyzer:
    """Minimal Crab pulsar analyzer using Z²₂ test and cosipy utilities."""

    def __init__(self, cfg: _Cfg):
        self.cfg = cfg
        self._t = None
        self._ph = None


    @staticmethod
    def from_yaml(path: str) -> "PulsarAnalyzer":
        with open(path, "r") as f:
            d = yaml.safe_load(f) or {}
        fs = d.get("fig_size", [12, 9])
        d["fig_size"] = (float(fs[0]), float(fs[1]))
        if isinstance(d.get("save_path"), str) and not d["save_path"].strip():
            d["save_path"] = None
        cfg = _Cfg(**d)
        return PulsarAnalyzer(cfg)

 
    def ensure_fits(self) -> None:
        """
        Ensure the FITS file exists locally.
        - Use local .fits or .fits.gz if present.
        - Otherwise fetch from Wasabi.
        """
        if not self.cfg.wasabi_key:
            raise ValueError("No wasabi_key specified in config.")

        local = Path(self.cfg.local_fits)
        local.parent.mkdir(parents=True, exist_ok=True)

        # Determine both possible names
        if local.suffix == ".gz":
            local_gz = local
            local_plain = local.with_suffix("")
        else:
            local_plain = local
            local_gz = local.with_suffix(local.suffix + ".gz")

        # Prefer an existing file (plain or gz)
        if local.exists():
            print(f"File already exists, using local copy: {local}")
            self.cfg.local_fits = str(local)
            return
        if local_gz.exists():
            print(f"Compressed local copy found, using: {local_gz}")
            self.cfg.local_fits = str(local_gz)
            return

        # Neither exists — fetch new file
        target = local_gz if str(self.cfg.wasabi_key).endswith(".gz") else local_plain
        try:
            fetch_wasabi_file(self.cfg.wasabi_key, output=str(target))
            print(f"Fetched from Wasabi: {target}")
            self.cfg.local_fits = str(target)
        except RuntimeError as e:
            if "already exists" in str(e):
                print(f"File already exists, using local copy: {target}")
                self.cfg.local_fits = str(target)
            else:
                raise

    def read_times(self) -> np.ndarray:
        self.ensure_fits()
        with fits.open(self.cfg.local_fits, memmap=True) as hdul:
            t = np.asarray(hdul[1].data[self.cfg.time_col], dtype=np.float64)
        t = np.sort(t[np.isfinite(t)])
        if t.size == 0:
            raise RuntimeError("No finite times found in FITS file.")
        self._t = t
        print(f"Loaded {t.size} events from {self.cfg.local_fits}")
        return t


    @staticmethod
    def fold_phases(t: np.ndarray, P: float) -> np.ndarray:
        return (t / P) % 1.0

    @staticmethod
    def z2_2(phases: np.ndarray) -> float:
        """Z²₂ statistic (two harmonics for pulsars with two peaks)."""
        N = phases.size
        if N == 0:
            return np.nan
        twopi = 2 * np.pi
        Z2 = 0.0
        for k in (1, 2):
            ang = twopi * k * phases
            c, s = np.cos(ang).sum(), np.sin(ang).sum()
            Z2 += (2.0 / N) * (c * c + s * s)
        return float(Z2)

    def significance(self) -> float:
        if self._t is None:
            self.read_times()
        if self._ph is None:
            self._ph = self.fold_phases(self._t, self.cfg.period)
        Z2 = self.z2_2(self._ph)
        print(f"Z²₂ = {Z2:.3f}")
        return Z2

    def make_three_panel_figure(self) -> plt.Figure:
        if self._t is None:
            self.read_times()
        if self._ph is None:
            self._ph = self.fold_phases(self._t, self.cfg.period)

        c = self.cfg
        t, ph = self._t, self._ph
        ts = t - t.min()

        # Profile
        hist, edges = np.histogram(ph, bins=c.nbins_profile, range=(0, 1))
        centers = 0.5 * (edges[1:] + edges[:-1])

        # Phaseogram
        phase_edges = np.linspace(0, 1, c.nbins_phase + 1)
        time_edges = np.linspace(ts.min(), ts.max(), c.nbins_time + 1)
        H2D, _, _ = np.histogram2d(ph, ts, bins=[phase_edges, time_edges])
        H2D = H2D.T

        # Cumulative Z²₂
        idx = np.linspace(0, t.size, c.n_segments_stat + 1, dtype=int)[1:]
        Z_series = [self.z2_2(ph[:i]) for i in idx]
        T_series = [ts[i - 1] for i in idx]

        fig = plt.figure(figsize=c.fig_size)
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], wspace=0.3, hspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.step(np.r_[centers, centers + 1], np.r_[hist, hist], where="mid", color="purple")
        ax1.set_xlim(0, 2)
        ax1.set_xlabel("Pulse phase")
        ax1.set_ylabel("Counts")

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(T_series, Z_series, ".-", lw=1.5, ms=4, color="purple")
        ax2.set_xlabel("Time since MET start (s)")
        ax2.set_ylabel("Z²₂")
        ax2.grid(alpha=0.25)

        ax3 = fig.add_subplot(gs[:, 1])
        pc = ax3.pcolormesh(phase_edges, time_edges, H2D, shading="auto")
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("Pulse phase")
        ax3.set_ylabel("Time since MET start (s)")
        fig.colorbar(pc, ax=ax3, pad=0.01, label="Counts/bin")

        if c.tight_layout:
            fig.tight_layout()
        return fig


    def run(self, show=False) -> plt.Figure:
        fig = self.make_three_panel_figure()
        if self.cfg.save_path:
            Path(self.cfg.save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.cfg.save_path, dpi=self.cfg.dpi)
            print(f"Saved to {self.cfg.save_path}")
        if show:
            plt.show()
        return fig
