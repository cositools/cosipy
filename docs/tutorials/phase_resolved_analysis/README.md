# Phase Analysis Tools
This subpackage provides utilities for phase-resolved pulsar analysis in **cosipy**.

## `PulsarAnalyzer`

### Overview
`PulsarAnalyzer` is a minimal tool to:
- Read **unbinned FITS** event lists (from Wasabi or local storage)
- Fold photon **arrival times** with a known pulsar period
- Compute the **Z²₂ test statistic** (useful for multi-peaked pulsars like the Crab)
- Generate a **three-panel diagnostic figure**:
  1. Folded pulse profile  
  2. Z²₂ statistic vs. time  
  3. Phaseogram (time vs. phase)

This tool supports direct integration with COSIpy’s **Wasabi data utilities**, enabling automatic data fetching and reproducible configuration through YAML.

---

## ⚙️ Configuration (via `config.yaml`)
Example configuration file:

```yaml
wasabi_key: "COSI-SMEX/DC2/Data/Sources/Crab_DC2_3months_unbinned_data.fits.gz"
local_fits: "dc2/Crab_DC2_3months_unbinned_data.fits.gz"
time_col: "TimeTags"
period: 0.0333924123
nbins_profile: 100
nbins_phase: 64
nbins_time: 128
n_segments_stat: 20
fig_size: [12, 9]
title_prefix: "Crab Pulsar"
save_path: "crab_three_panel.png"
dpi: 150
tight_layout: true
