"""inr_angular - Method 1 (event-data-driven INR) GRB localization for COSI.

Localization from the on-burst angular triplets (phi_i, l_axis_i, b_axis_i) alone -
no instrument response, no off-burst background, no truth-dependent calibration.

Module map:
    config          - dataclass configuration; fixed design constants
    events          - loading + mandated on-burst selection of the angular triplets
    geometry        - SkyCoord-separation cone geometry: alpha_i(s), r_i(s)
    kernel          - ARM kernel h(r|phi), truncated + normalized; truth-free scale CV
    scoring         - g_i, the eta-profiled mixture, the exact localization score
                      S(s) = 2[ell(s) - ell_0]; parallel evaluation
    healpix_search  - regular full-sky HEALPix scoring (SkyMap; NESTED, Galactic)
    moc_scan        - hierarchical NESTED parent->child ladder (separate notebook only)
    inr_model       - INR surrogate f_theta ~ S (torch): training data, training, gate,
                      continuous argmax -> (l_INR, b_INR)
    refine          - exact Nelder-Mead polish (calibrated pipeline)
    calibrate       - Delta-score, sandwich/Godambe, bootstrap c90 (calibrated pipeline)
    region          - score regions, convex hull, spherical-ellipse fitting
    plotting        - localization figures (mhealpy mollview/cartview)
    results         - JSON / ellipse txt / npz / FITS persistence
"""

from . import (calibrate, config, diagnostics, events, geometry, healpix_search,
               inr_model, kernel, moc_scan, plotting, refine, region, results, scoring)
from .config import (CHI2_2_90, CHI2_2_999, SEED, BootstrapConfig, INRConfig,
                     KernelConfig, Method1Config, MOCConfig, ObjectiveConfig,
                     RegionConfig, ScanConfig)
from .events import AngularEvents, find_background_cache, find_data_file, load_onburst_events
from .geometry import (alpha_deg, lb_to_local_chart, local_chart_to_lb, offset_by,
                       residual_deg, separation_deg, separation_skycoord_deg,
                       validate_against_skycoord)
from .healpix_search import SkyMap, regular_scan
from .inr_model import INRSurrogate, prepare_training_data, train_inr, train_inr_active
from .kernel import ARMKernel, select_scale_cv
from .moc_scan import moc_scan
from .scoring import (U_FLOOR, LocalizationScore, Method1Objective,
                      probe_parallel_safe, verify_parallel_consistency)
from .refine import exact_polish
from .calibrate import (bootstrap_c90, delta_score, delta_ts, sandwich_diagnostic,
                        simulate_replicate)
from .region import (disc_pixels, extract_region, fit_ellipse_from_hull,
                     fit_ellipse_summary, hull_boundary)
from .results import save_ellipse_result_txt, save_healpix_map_fits, save_results
from .diagnostics import run_environment_check
from .plotting import (dense_healpix_map, moc_healpix_map, plot_fullsky_map,
                       plot_local_zoom, plot_region_ellipse,
                       plot_score_map_with_ellipse, window_moc_map)

__version__ = "1.1.0"
