"""Configuration defaults for Method 1 (event-data-driven INR localization).

Every constant here is either a fixed design choice, an instrument-characterization
placeholder, or a per-burst quantity computed from that burst's own events.  Nothing is
derived from any GRB's true position (see `INR_Two_Method_Localization_Methodology.md`,
Part VII, "truth firewall").
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Reproducibility ---------------------------------------------------------------------
SEED: int = 12345

# Wilks reference for 2 location d.o.f. (reported always, adopted never without checks)
CHI2_2_90: float = 4.605170185988092          # scipy.stats.chi2.ppf(0.90, 2)
CHI2_2_999: float = 13.815510557964274        # scipy.stats.chi2.ppf(0.999, 2) - MOC keep unit


@dataclass
class KernelConfig:
    """ARM kernel h(r|phi) family parameters (angles internally in RADIANS).

    The family is a zero-mean Gaussian core plus a Cauchy (Lorentzian) shoulder,
    truncated to the physical residual support r in [-phi, pi - phi] and renormalized
    per event.  The *scale* is selected truth-free from the on-burst events by K-fold
    pseudo-likelihood cross-validation (methodology Sec. II.2, route 3), because no
    sim-distillation product ships with this repository; the family parameters below are
    declared instrument-characterization placeholders, not fits to any GRB truth.
    """

    sigma_deg: float = 2.0          # Gaussian core scale (overwritten by the CV selection)
    tail_weight: float = 0.15       # Cauchy shoulder mixture weight w_t in [0, 1)
    tail_gamma_over_sigma: float = 3.0   # gamma = ratio * sigma (heavy shoulder)

    def with_sigma(self, sigma_deg: float) -> "KernelConfig":
        return KernelConfig(sigma_deg=float(sigma_deg),
                            tail_weight=self.tail_weight,
                            tail_gamma_over_sigma=self.tail_gamma_over_sigma)


@dataclass
class ObjectiveConfig:
    """Exact Method-1 objective evaluation settings."""

    sin_alpha_floor_deg: float = 0.05
    # Numerical-stability guard for the integrable 1/(2 pi sin alpha) singularity of the
    # solid-angle density at alpha -> 0 or 180 deg (candidate exactly on an event axis /
    # antipode).  The floor caps the annulus Jacobian at a scale far below the pixel
    # spacing and the ARM width, so it cannot influence the localization; it only
    # prevents isolated +inf spikes of the log-density at the (measure-zero) axis points.
    newton_max_iter: int = 60
    newton_tol: float = 1e-10
    chunk_pixels: int = 512         # pixels per numpy-path block; bounds the
                                # (chunk x N) temporaries to ~0.1 GB


@dataclass
class ScanConfig:
    """Regular full-sky HEALPix scan (methodology Sec. IV.1)."""

    nside: int = 128                # 196,608 pixels, ~0.46 deg - the TS-map baseline scale
    nest: bool = True


@dataclass
class MOCConfig:
    """Multi-resolution (MOC) scan (methodology Sec. IV.2). Pure acceleration of ScanConfig."""

    nside0: int = 16                # coarse full-sky rung (3072 pixels, ~3.7 deg)
    nside_max: int = 512
    kappa_coarse: float = 4.0       # keep-margin inflation at the coarsest level ...
    kappa_fine: float = 2.0         # ... shrinking to this at the finest level
    pad_neighbors: bool = True
    argmax_move_frac: float = 0.1   # early-stop: argmax moved < frac * pixel scale ...
    # ... AND the active set straddles the keep band (see moc_scan.moc_scan).


@dataclass
class INRConfig:
    """INR surrogate f_theta(l, b) ~ TS1(l, b) (methodology Part V)."""

    n_frequencies: int = 4          # positional-encoding dyadic frequencies L
    hidden: tuple = (64, 64)        # tanh MLP hidden layers
    adam_iters: int = 800
    adam_lr: float = 1e-3
    lbfgs_iters: int = 200
    val_fraction: float = 0.2       # held-out fraction for the acceptance gate
    gate_band_ts: float = 25.0      # "inside the candidate region" = TS >= TSmax - band
    gate_tol_ts: float = 0.2        # acceptance gate tolerance in TS units (doc Sec. V)
    peak_weight: float = 4.0        # loss up-weighting toward the peak band
    seed: int = SEED


@dataclass
class BootstrapConfig:
    """Conditional parametric bootstrap for c90 (methodology Sec. VI.2)."""

    n_replicates: int = 200
    cap_radius_deg: float = 8.0     # reduced-search cap around s_hat (legitimate: the
                                    # replicates are generated there)
    grid_step_deg: float = 0.4      # coarse replicate-search grid step inside the cap
    polish: bool = True             # Nelder-Mead polish of each replicate argmax
    seed: int = SEED


@dataclass
class RegionConfig:
    """Exact 90% C.L. region extraction (methodology Sec. VI.4)."""

    nside_region: int = 1024        # membership resolution (~0.057 deg pixels)
    bracket_margin_ts: float = 3.0  # exact evaluation band beyond c90 (safety margin)
    inr_prefilter_margin_ts: float = 8.0   # INR bracket: drop children whose surrogate
                                           # deficit exceeds c90 + this margin


@dataclass
class Method1Config:
    kernel: KernelConfig = field(default_factory=KernelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    moc: MOCConfig = field(default_factory=MOCConfig)
    inr: INRConfig = field(default_factory=INRConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    region: RegionConfig = field(default_factory=RegionConfig)
    seed: int = SEED
