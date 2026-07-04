#!/usr/bin/env python
"""Visualize geometric wake-steering profiles (angular roses) at different sigma_smooth
(and the harmonic budget) -- the picture behind the v11 sharpness sweep.

v11 found that the sharpest profile arm `s2h32` (sigma_smooth=2, n_harmonics=32) lifts
large-farm power_ratio above greedy, while the over-smoothed default `s10h8` sits at greedy.
The story: at sigma_smooth=10 every turbine's directional rose collapses to a near-identical
smooth blob -- in a big farm the per-turbine profiles become indistinguishable, so there is no
per-turbine signal. Sharpening (lower sigma) restores structure, but capturing it needs more
Fourier harmonics (h=8 is too few; h must rise as sigma falls).

These figures reproduce EXACTLY the geometric profiles v11 used (--profile_source geometric,
n_profile_directions=180, geom_mode="wake"), so "sigma=2" here means what it did in s2h32.
NOTE sigma_smooth is in BINS: at 180 directions that is 2 deg/bin, so sigma=2 -> 4 deg.

Outputs (figs/profiles/):
  profiles_sigma_sweep.png/pdf    - one representative turbine per layout, rose at sigma={2,3,5,10}
  profiles_discriminability.png   - ALL turbines' roses, sigma=10 (blob) vs sigma=2 (distinct)
  profiles_harmonics.png          - sigma=2 rose vs FFT reconstruction at h={8,16,32} (why s2h32)

Usage
-----
    uv run python TransformerSac/plot_profiles.py
    uv run python TransformerSac/plot_profiles.py --sigmas 2,3,5,10 --harmonics 8,16,32
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)  # so `from helpers... import` resolves (mirrors spectral_diagnostic.py)

from helpers.geometric_profiles import compute_layout_profiles_vectorized  # noqa: E402
from helpers.layouts import get_layout_positions  # noqa: E402
from helpers.layout_gen import generate_layout_pool  # noqa: E402

CHANNEL_IDX = {"receptivity": 0, "influence": 1}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_turbine():
    from py_wake.examples.data.dtu10mw import DTU10MW
    return DTU10MW()


def cluster_sample(D: float, seed: int):
    """One ~25-turbine PLayGen cluster layout (screened like training; unscreened fallback)."""
    for screen in (True, False):
        pool = generate_layout_pool(6, 25, 25, D, seed=seed, generator="cluster",
                                    screen_headroom=screen)
        if pool:
            name, x, y = pool[0]
            return f"cluster ({len(x)}t)", np.asarray(x, float), np.asarray(y, float)
    raise RuntimeError("cluster generator returned no layouts")


def build_layouts(names, D, seed):
    """Return [(label, x, y), ...] for the requested layout keys."""
    wt = get_turbine()
    out = []
    for n in names:
        if n == "cluster":
            out.append(cluster_sample(D, seed))
        else:
            x, y = get_layout_positions(n, wt)
            out.append((f"{n} ({len(x)}t)", np.asarray(x, float), np.asarray(y, float)))
    return out


def roses(x, y, D, sigma, n_dir, channel):
    """Receptivity (or influence) angular roses for every turbine: shape (n_turb, n_dir)."""
    recep, influ = compute_layout_profiles_vectorized(
        x, y, rotor_diameter=D, k_wake=0.04, n_directions=n_dir,
        sigma_smooth=sigma, scale_factor=15.0, mode="wake")
    return (recep if channel == "receptivity" else influ)


def harmonic_recon(rose: np.ndarray, h: int) -> np.ndarray:
    """What a FourierProfileEncoder with n_harmonics=h sees: keep modes 0..h, inverse-FFT."""
    fft = np.fft.rfft(rose)
    fft[h + 1:] = 0.0
    return np.fft.irfft(fft, n=rose.shape[-1])


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def sigma_colors(sigmas):
    """Sharp (small sigma) = dark/purple, smooth (large sigma) = yellow (viridis by rank)."""
    cmap = plt.get_cmap("viridis")
    lo, hi = min(sigmas), max(sigmas)
    span = (hi - lo) or 1.0
    return {s: cmap(0.12 + 0.78 * (s - lo) / span) for s in sigmas}


def _close(vals):
    return np.append(vals, vals[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", default="square_3x3,25_turb_test,cluster")
    ap.add_argument("--sigmas", default="2,3,5,10")
    ap.add_argument("--harmonics", default="8,16,32")
    ap.add_argument("--n-dir", type=int, default=180, help="profile directions (v11 used 180)")
    ap.add_argument("--channel", choices=("receptivity", "influence"), default="receptivity")
    ap.add_argument("--seed", type=int, default=0, help="seed for the cluster sample")
    ap.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "..", "figs", "profiles"))
    args = ap.parse_args()

    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)
    sigmas = [float(s) for s in args.sigmas.split(",")]
    harmonics = [int(h) for h in args.harmonics.split(",")]
    ndir = args.n_dir
    chan = args.channel
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})

    wt = get_turbine()
    D = wt.diameter()
    layouts = build_layouts(args.layouts.split(","), D, args.seed)

    # Precompute roses[layout_idx][sigma] = (n_turb, n_dir); representative turbine = most
    # wake-exposed at the SHARPEST sigma (clearest angular structure).
    s_sharp, s_smooth = min(sigmas), max(sigmas)
    cache, rep = [], []
    print(f"channel={chan}  n_dir={ndir}  sigmas={sigmas}  harmonics={harmonics}")
    for label, x, y in layouts:
        per_sigma = {s: roses(x, y, D, s, ndir, chan) for s in sigmas}
        ti = int(np.argmax(per_sigma[s_sharp].sum(axis=1)))
        cache.append(per_sigma)
        rep.append(ti)
        print(f"  {label:18s}  n_turb={x.size:3d}  representative turbine = T{ti}")

    deg = np.arange(ndir) * (360.0 / ndir)
    ang = np.deg2rad(deg)
    ang_c = _close(ang)
    scolors = sigma_colors(sigmas)
    nrows = len(layouts)

    def savefig(fig, name):
        for ext in ("png", "pdf"):
            p = os.path.join(out, f"{name}.{ext}")
            fig.savefig(p, bbox_inches="tight")
        print(f"Wrote {os.path.join(out, name)}.png")
        plt.close(fig)

    # ---- Figure 1: sigma sweep on one representative turbine (cartesian | polar) ----
    fig = plt.figure(figsize=(13, 3.6 * nrows))
    gs = GridSpec(nrows, 2, figure=fig, width_ratios=[1.4, 1.0])
    for r, (label, x, y) in enumerate(layouts):
        ti = rep[r]
        axc = fig.add_subplot(gs[r, 0])
        axp = fig.add_subplot(gs[r, 1], projection="polar")
        for s in sigmas:
            rose = cache[r][s][ti]
            axc.plot(deg, rose, color=scolors[s], lw=2, label=f"σ={s:g}")
            axp.plot(ang_c, _close(rose), color=scolors[s], lw=2)
        axc.set_xlim(0, 360)
        axc.set_xticks(range(0, 361, 90))
        axc.set_xlabel("wind direction [deg]")
        axc.set_ylabel(f"{chan}")
        axc.set_title(f"{label} — turbine T{ti}")
        axc.grid(True, alpha=0.3)
        axc.legend(fontsize=8, frameon=False, ncol=len(sigmas))
        axp.set_theta_zero_location("N")
        axp.set_theta_direction(-1)
        axp.set_title("polar", fontsize=9)
        axp.set_xticklabels([])
        axp.set_yticklabels([])
    fig.suptitle(f"Profile sharpness vs sigma_smooth  ·  geometric, n_dir={ndir}  "
                 f"(small σ = sharper)", y=1.005, fontsize=12)
    fig.tight_layout()
    savefig(fig, "profiles_sigma_sweep")

    # ---- Figure 2: within-farm discriminability, all turbines, sigma_smooth vs sigma_sharp ----
    fig = plt.figure(figsize=(11, 4.2 * nrows))
    gs = GridSpec(nrows, 2, figure=fig)
    for r, (label, x, y) in enumerate(layouts):
        n_turb = x.size
        tcmap = plt.get_cmap("turbo")
        for ci, (s, tag) in enumerate([(s_smooth, "over-smoothed"), (s_sharp, "sharp")]):
            axp = fig.add_subplot(gs[r, ci], projection="polar")
            arr = cache[r][s]
            for t in range(n_turb):
                axp.plot(ang_c, _close(arr[t]), color=tcmap(t / max(1, n_turb - 1)),
                         lw=1.0, alpha=0.6)
            axp.set_theta_zero_location("N")
            axp.set_theta_direction(-1)
            axp.set_xticklabels([])
            axp.set_yticklabels([])
            axp.set_title(f"{label}\nσ={s:g}  ({tag}, all {n_turb} turbines)", fontsize=10)
    fig.suptitle("Within-farm discriminability: over-smoothing collapses per-turbine roses "
                 "into near-identical blobs (worst in big farms)", y=1.003, fontsize=12)
    fig.tight_layout()
    savefig(fig, "profiles_discriminability")

    # ---- Figure 3: harmonic budget on the sharp (sigma=sharp) representative rose ----
    hcmap = plt.get_cmap("plasma")
    hcolors = {h: hcmap(0.15 + 0.7 * i / max(1, len(harmonics) - 1))
               for i, h in enumerate(harmonics)}
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 3.0 * nrows), squeeze=False)
    for r, (label, x, y) in enumerate(layouts):
        ti = rep[r]
        rose = cache[r][s_sharp][ti]
        ax = axes[r, 0]
        ax.plot(deg, rose, color="black", lw=2.4, label="raw (σ=%g)" % s_sharp, zorder=5)
        for h in harmonics:
            ax.plot(deg, harmonic_recon(rose, h), color=hcolors[h], lw=1.8,
                    label=f"h={h}")
        ax.set_xlim(0, 360)
        ax.set_xticks(range(0, 361, 90))
        ax.set_ylabel(chan)
        ax.set_title(f"{label} — turbine T{ti}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=False, ncol=len(harmonics) + 1)
    axes[-1, 0].set_xlabel("wind direction [deg]")
    fig.suptitle(f"Harmonic budget: how many Fourier modes the encoder needs to keep a σ={s_sharp:g} "
                 f"rose (h=8 rounds it off → s2h32 needs h up)", y=1.004, fontsize=12)
    fig.tight_layout()
    savefig(fig, "profiles_harmonics")

    print(f"\nDone. 3 figures in {out}/")


if __name__ == "__main__":
    main()
