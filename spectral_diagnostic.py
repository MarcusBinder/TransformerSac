#!/usr/bin/env python
"""Spectral diagnostic for receptivity / influence profiles.

WHY THIS EXISTS
---------------
The Transformer-SAC wake-steering controller transfers poorly to large/irregular
farms, and DR-trained runs show per-step rewards only slightly above zero on DR
layouts (barely beating greedy -> not extracting layout-specific headroom). Hypothesis:
the Fourier profile encoding (DC + 8 harmonics) is fine for smooth grid-like layouts
but too coarse for random/clustered layouts, whose distinctive wake structure is sharp,
narrow-angle and high-frequency. The DR runs (pywake_transfer_v8/v11) fed BOTH a
relative_mlp attention bias AND Fourier profiles with profile_source=geometric,
n_profile_directions=180, 8 harmonics. So the profile the model saw is the *analytic*
geometric rose (smoother than PyWake) -> three stacked low-pass stages:
geometric idealisation -> Gaussian smoothing -> 8-harmonic truncation.

This script quantifies, in frequency space and PER LAYOUT FAMILY:
  1. how much rose energy lives BEYOND the 8-harmonic cut (tail-energy fraction),
  2. whether there is a train->test spectral shift (test/cluster farms rougher than
     the training grids the harmonic weights were tuned on),
  3. RECEPTIVITY vs INFLUENCE redundancy: in the geometric source influence is an exact
     180-deg rotation of receptivity, so one channel may be droppable during training.

WHAT IT DOES *NOT* ISOLATE
--------------------------
Geometry also reaches the policy via relative_mlp, so a lossy profile channel is not
proof of the *cause* of poor transfer. A positive result motivates the follow-ups:
  - harmonic-budget sweep n_harmonics in {4,8,16,32} (+ sigma_smooth down): does
    large-farm transfer improve? If yes, truncation was the binding constraint.
  - geometric-vs-pywake spectral gap: how much the geometric *approximation* discards
    vs the FFT truncation.
  - channel ablation (profiles-off vs relative-off) to attribute between the two
    geometry channels.

Runs locally on CPU. Geometric source is instant; PyWake source loops a wake sim per
turbine (slower) -- keep --n_dr modest when including it.

  python spectral_diagnostic.py --source geometric            # fast sanity
  python spectral_diagnostic.py --source both --n_dr 12        # full headline figure
  python spectral_diagnostic.py --selftest                     # convention cross-checks
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- make the project's helpers importable whether run from repo root or here ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (SCRIPT_DIR, os.path.join(SCRIPT_DIR, "helpers")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from helpers.layouts import get_layout_positions                      # noqa: E402
from helpers.layout_gen import generate_layout_pool                   # noqa: E402
from helpers.geometric_profiles import compute_layout_profiles_vectorized  # noqa: E402
from helpers.receptivity_profiles import compute_layout_profiles as compute_layout_profiles_pywake  # noqa: E402

K_MODEL = 8          # harmonic budget the FourierProfileEncoder actually keeps
ROSE_TYPES = ("receptivity", "influence")
# Stable colour per layout group.
GROUP_COLORS = {
    "train_grid":   "#1f77b4",
    "train_irreg":  "#17becf",
    "dr_irregular": "#2ca02c",
    "dr_cluster":   "#ff7f0e",
    "test_25":      "#d62728",
}
SOURCE_STYLE = {"geometric": "-", "pywake": "--"}


# --------------------------------------------------------------------------- #
# Layout groups
# --------------------------------------------------------------------------- #
def get_turbine():
    from py_wake.examples.data.dtu10mw import DTU10MW
    return DTU10MW()


def build_groups(args, wt, D):
    """Return {group_name: [(layout_name, x, y), ...]}."""
    groups = {}
    groups["train_grid"] = [(n, *get_layout_positions(n, wt)) for n in ("T1", "T2", "T3", "T4", "T5", "T6")]
    groups["train_irreg"] = [(n, *get_layout_positions(n, wt)) for n in ("T7", "T8", "T9")]
    groups["dr_irregular"] = generate_layout_pool(
        args.n_dr, args.dr_lo, args.dr_hi, D, seed=args.seed,
        min_dist_D=args.min_dist_D, generator="irregular",
    )
    try:
        groups["dr_cluster"] = generate_layout_pool(
            args.n_dr, args.dr_lo, args.dr_hi, D, seed=args.seed,
            min_dist_D=args.min_dist_D, generator="cluster",
        )
    except Exception as exc:  # PLayGen optional dependency may be missing
        print(f"[warn] cluster generator unavailable ({exc}); skipping dr_cluster")
    x25, y25 = get_layout_positions("25_turb_test", wt)
    assert len(x25) == 25, f"25_turb_test has {len(x25)} turbines, expected 25"
    groups["test_25"] = [("25_turb_test", x25, y25)]
    return groups


def roses_for(x, y, wt, D, source, n_dir, args, geom_mode="wake"):
    """Return (receptivity, influence) arrays, each (n_turb, n_dir)."""
    if source == "geometric":
        return compute_layout_profiles_vectorized(
            np.asarray(x, float), np.asarray(y, float),
            rotor_diameter=D, k_wake=args.k_wake, n_directions=n_dir,
            sigma_smooth=args.sigma_smooth, scale_factor=args.scale_factor,
            mode=geom_mode,
        )
    return compute_layout_profiles_pywake(
        np.asarray(x, float), np.asarray(y, float), wt,
        n_directions=n_dir, verbose=False,
    )


def within_farm_discriminability(roses):
    """Mean pairwise (1 - cosine) between turbines' roses inside ONE farm.

    0 => turbines have identical angular patterns (indistinguishable to the encoder);
    higher => more separable. NaN for n_turb<2. Measures how distinct the per-turbine
    angular signatures are — the structure sigma_smooth erodes. Empirically: large/random
    farms (test_25) are the LEAST discriminable, and sharpening (sigma↓) raises disc most for
    them. Raw (uncentered) roses, so it reflects what the encoder sees.
    """
    n = roses.shape[0]
    if n < 2:
        return float("nan")
    unit = roses / np.maximum(np.linalg.norm(roses, axis=1, keepdims=True), 1e-12)
    sims = unit @ unit.T
    iu = np.triu_indices(n, k=1)
    return float(1.0 - sims[iu].mean())


# --------------------------------------------------------------------------- #
# Spectral metrics  (magnitude spectrum is rotation-invariant -> wind-relative
# rotation only shifts phase, so analysing raw, unrotated roses is clean)
# --------------------------------------------------------------------------- #
def rose_metrics(rose, K=K_MODEL):
    """Per-rose spectral metrics. cum/specdens indexed by harmonic 0..n//2."""
    n = rose.shape[-1]
    R = np.fft.rfft(rose)
    power = np.abs(R) ** 2
    ac = power[1:]                      # drop DC (the encoder always keeps DC)
    ac_tot = ac.sum()
    n_harm = power.shape[-1]            # n//2 + 1
    if ac_tot < 1e-12:                  # flat rose (isolated turbine) -> no AC structure
        return dict(tail_frac=0.0, recon_err=0.0, k90=0,
                    dc=float(np.abs(R[0])), h1_mag=0.0,
                    cum=np.ones(n_harm), specdens=np.zeros(n_harm))
    tail_frac = float(ac[K:].sum() / ac_tot)            # energy in harmonics > K
    cum = np.concatenate([[0.0], np.cumsum(ac) / ac_tot])  # cum[k] = frac in first k harmonics
    k90 = int(np.searchsorted(cum, 0.90))
    # low-pass-K reconstruction (DC..K kept, like the encoder), error vs full rose
    Rlp = np.zeros_like(R)
    Rlp[:K + 1] = R[:K + 1]
    recon = np.fft.irfft(Rlp, n=n)
    recon_err = float(np.linalg.norm(rose - recon) / max(np.linalg.norm(rose), 1e-12))
    return dict(tail_frac=tail_frac, recon_err=recon_err, k90=k90,
                dc=float(np.abs(R[0])), h1_mag=float(np.abs(R[1]) * 2 / n),
                cum=cum, specdens=power / ac_tot)


def redundancy(rec_i, inf_i):
    """How well influence_i matches receptivity_i rolled by 180 deg.

    In the geometric source these are equal by construction -> err~0, corr~1.
    In PyWake influence is a yaw-sensitivity, not a mirror -> they diverge.
    """
    n = len(rec_i)
    rr = np.roll(rec_i, n // 2)
    err = float(np.linalg.norm(inf_i - rr) / max(np.linalg.norm(rec_i), 1e-12))
    a, b = inf_i - inf_i.mean(), rr - rr.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    corr = float(a @ b / denom) if denom > 1e-12 else 1.0
    return err, corr


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def collect(args, wt, D, sources):
    groups = build_groups(args, wt, D)
    rows = []                                   # CSV rows
    # spectra[(source, rose, group)] -> list of specdens / cum arrays for averaging
    spectra = {}
    for source in sources:
        for gname, layouts in groups.items():
            print(f"[{source}] group {gname}: {len(layouts)} layout(s)")
            for (lname, x, y) in layouts:
                rec, inf = roses_for(x, y, wt, D, source, args.n_dir, args)
                roses = {"receptivity": rec, "influence": inf}
                for t in range(rec.shape[0]):
                    rerr, rcorr = redundancy(rec[t], inf[t])
                    for rtype in ROSE_TYPES:
                        m = rose_metrics(roses[rtype][t], K=args.K)
                        rows.append(dict(
                            source=source, group=gname, layout=lname, turb=t,
                            rose_type=rtype, n_turb=rec.shape[0],
                            tail_frac=m["tail_frac"], recon_err=m["recon_err"],
                            k90=m["k90"], dc=m["dc"], h1_mag=m["h1_mag"],
                            redun_err=rerr, redun_corr=rcorr,
                        ))
                        spectra.setdefault((source, rtype, gname), {"sd": [], "cum": []})
                        spectra[(source, rtype, gname)]["sd"].append(m["specdens"])
                        spectra[(source, rtype, gname)]["cum"].append(m["cum"])
    return rows, spectra, list(groups.keys())


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _stat(rows, source, group, rtype, key):
    vals = [r[key] for r in rows
            if r["source"] == source and r["group"] == group and r["rose_type"] == rtype]
    return (np.mean(vals), np.median(vals)) if vals else (np.nan, np.nan)


def print_summary(rows, sources, group_names, K):
    print("\n" + "=" * 78)
    print(f"SPECTRAL SUMMARY  (harmonic budget K={K}; tail = energy in harmonics > K)")
    print("=" * 78)
    for source in sources:
        print(f"\n--- source = {source} ---")
        print(f"{'group':<14}{'rose':<13}{'tail_frac':>11}{'recon_err':>11}{'k90(med)':>10}")
        for g in group_names:
            for rtype in ROSE_TYPES:
                tf, _ = _stat(rows, source, g, rtype, "tail_frac")
                re_, _ = _stat(rows, source, g, rtype, "recon_err")
                _, k90 = _stat(rows, source, g, rtype, "k90")
                if np.isnan(tf):
                    continue
                print(f"{g:<14}{rtype:<13}{tf:>11.4f}{re_:>11.4f}{k90:>10.1f}")

    # Train -> test spectral-shift verdict (receptivity is the dominant channel)
    print("\n" + "-" * 78)
    print("TRAIN->TEST SHIFT (receptivity tail_frac vs train_grid):")
    for source in sources:
        base, _ = _stat(rows, source, "train_grid", "receptivity", "tail_frac")
        if np.isnan(base):
            continue
        for g in ("dr_irregular", "dr_cluster", "test_25"):
            tf, _ = _stat(rows, source, g, "receptivity", "tail_frac")
            if np.isnan(tf):
                continue
            ratio = tf / base if base > 1e-9 else float("inf")
            flag = "  <-- rougher than training" if ratio >= 1.3 else ""
            print(f"  [{source:<9}] {g:<14} tail={tf:.4f}  ({ratio:.2f}x train_grid){flag}")

    # Receptivity vs influence redundancy
    print("\n" + "-" * 78)
    print("RECEPTIVITY vs INFLUENCE redundancy (influence vs receptivity rolled 180 deg):")
    print(f"{'source':<11}{'group':<14}{'redun_err(med)':>16}{'redun_corr(med)':>17}")
    for source in sources:
        for g in group_names:
            _, e = _stat(rows, source, g, "receptivity", "redun_err")
            _, c = _stat(rows, source, g, "receptivity", "redun_corr")
            if np.isnan(e):
                continue
            note = "  (redundant -> droppable)" if (e < 0.05 and c > 0.98) else ""
            print(f"{source:<11}{g:<14}{e:>16.4f}{c:>17.4f}{note}")
    print("=" * 78 + "\n")


def write_csv(rows, path):
    cols = ["source", "group", "layout", "turb", "rose_type", "n_turb",
            "tail_frac", "recon_err", "k90", "dc", "h1_mag", "redun_err", "redun_corr"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[out] wrote {path}  ({len(rows)} rows)")


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def _mean_curve(spectra, key, field):
    arrs = spectra.get(key, {}).get(field, [])
    return np.mean(np.stack(arrs), axis=0) if arrs else None


def _single_vs_summed(ax, spectra, sources, group_names, nharm=10):
    """Show that folding upstream+downstream into ONE rose (summing) zeros odd harmonics.

    A 180-deg roll multiplies harmonic k by (-1)^k, so for c(t)=r(t)+r(t+180):
    |C_k|^2 = (1+(-1)^k)^2 |R_k|^2 = 4|R_k|^2 (even k) or 0 (odd k). Derived analytically
    from the stored single-rose power spectra -- no recompute.
    """
    prim = "geometric" if "geometric" in sources else sources[0]
    sds = []
    for g in group_names:
        sds.extend(spectra.get((prim, "receptivity", g), {}).get("sd", []))
    if not sds:
        ax.axis("off")
        return
    single = np.mean(np.stack(sds), axis=0)             # mean norm. power, index 0 = DC
    kk = np.arange(len(single))
    summed = single * np.where(kk % 2 == 0, 4.0, 0.0)   # (1+(-1)^k)^2 factor
    odd_frac = single[kk % 2 == 1].sum() / single[1:].sum()
    ks = np.arange(1, nharm + 1)
    w = 0.4
    ax.bar(ks - w / 2, single[1:nharm + 1], width=w, color="#1f77b4", label="single rose r(θ)")
    ax.bar(ks + w / 2, summed[1:nharm + 1], width=w, color="#d62728", alpha=0.8,
           label="summed r(θ)+r(θ+180°)")
    ax.set_xticks(ks)
    ax.set_xlabel("harmonic index", fontsize=9)
    ax.set_ylabel("mean power / single-rose AC energy", fontsize=8)
    ax.set_title(f"single vs summed rose ({prim}): summing zeros odd harmonics\n"
                 f"(odd harmonics = {odd_frac:.0%} of single-rose AC energy, incl. h1)",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


def _grouped_box(ax, rows, sources, group_names, rtype, key, title, ylabel):
    """Box per group, sources side-by-side."""
    width, positions, labels, ticks = 0.36, [], [], []
    for gi, g in enumerate(group_names):
        any_data = False
        for si, source in enumerate(sources):
            vals = [r[key] for r in rows if r["source"] == source
                    and r["group"] == g and r["rose_type"] == rtype]
            if not vals:
                continue
            any_data = True
            pos = gi + (si - (len(sources) - 1) / 2) * width
            bp = ax.boxplot([vals], positions=[pos], widths=width * 0.9,
                            patch_artist=True, showfliers=False)
            for box in bp["boxes"]:
                box.set_facecolor(GROUP_COLORS.get(g, "gray"))
                box.set_alpha(0.45 if source == "pywake" else 0.85)
            for med in bp["medians"]:
                med.set_color("black")
        if any_data:
            positions.append(gi)
            labels.append(g)
            ticks.append(gi)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)


def make_figure(rows, spectra, sources, group_names, K, path, xmax=40):
    fig, axes = plt.subplots(3, 4, figsize=(22, 13))
    harm = None
    for ri, rtype in enumerate(ROSE_TYPES):
        ax_spec, ax_cum = axes[ri][0], axes[ri][1]
        for g in group_names:
            for source in sources:
                sd = _mean_curve(spectra, (source, rtype, g), "sd")
                cu = _mean_curve(spectra, (source, rtype, g), "cum")
                if sd is None:
                    continue
                if harm is None:
                    harm = np.arange(len(sd))
                ls = SOURCE_STYLE[source]
                ax_spec.semilogy(harm[1:xmax + 1], sd[1:xmax + 1], ls,
                                 color=GROUP_COLORS.get(g, "gray"), alpha=0.85,
                                 label=f"{g}/{source}" if ri == 0 else None)
                ax_cum.plot(np.arange(len(cu))[:xmax + 1], cu[:xmax + 1], ls,
                            color=GROUP_COLORS.get(g, "gray"), alpha=0.85)
        for ax in (ax_spec, ax_cum):
            ax.axvline(K, color="k", ls=":", lw=1)
            ax.set_xlabel("harmonic index", fontsize=9)
            ax.grid(True, alpha=0.3)
        ax_spec.set_title(f"{rtype}: mean normalised power spectrum", fontsize=10)
        ax_spec.set_ylabel("power / AC energy", fontsize=9)
        ax_cum.axhline(0.90, color="gray", ls="--", lw=1)
        ax_cum.set_title(f"{rtype}: cumulative AC energy", fontsize=10)
        ax_cum.set_ylabel("fraction captured", fontsize=9)

        _grouped_box(axes[ri][2], rows, sources, group_names, rtype, "tail_frac",
                     f"{rtype}: tail-energy fraction (harmonics > {K})", "tail_frac")
        _grouped_box(axes[ri][3], rows, sources, group_names, rtype, "recon_err",
                     f"{rtype}: low-pass-{K} reconstruction error", "recon_err")

    # Row 3: redundancy, single-vs-summed, legend+note
    _grouped_box(axes[2][0], rows, sources, group_names, "receptivity", "redun_err",
                 "influence vs receptivity@180deg: rel. L2 error", "redun_err")
    _grouped_box(axes[2][1], rows, sources, group_names, "receptivity", "redun_corr",
                 "influence vs receptivity@180deg: correlation", "redun_corr")
    _single_vs_summed(axes[2][2], spectra, sources, group_names)
    axes[2][3].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[2][3].legend(handles, labels, loc="upper center", fontsize=7,
                          title="group / source  (solid=geometric, dashed=pywake)")
    note = ("tail_frac/recon_err high on test_25 & dr_cluster vs train_grid\n"
            "=> 8-harmonic cut drops test-relevant structure (pywake) / harmless (geom).\n"
            "redun_err~0 & corr~1 => influence redundant with receptivity (droppable).\n"
            "summing roses zeros odd harmonics (h1) => keep ONE signed 360-deg rose.")
    axes[2][3].text(0.0, 0.02, note, transform=axes[2][3].transAxes,
                    fontsize=8, va="bottom", family="monospace")

    fig.suptitle("Receptivity / influence rose spectra by layout family "
                 f"(model keeps DC + {K} harmonics)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=130)
    print(f"[out] wrote {path}")


# --------------------------------------------------------------------------- #
# sigma_smooth sweep: locate the low-pass (smoothing vs FFT truncation)
# --------------------------------------------------------------------------- #
MODE_STYLE = {"wake": "-", "distance": "--"}


def run_sigma_sweep(args, wt, D, sigmas, geom_mode="wake"):
    """For each smoothing level, aggregate geometric-rose spectral content + discriminability
    per group, for one geom_mode.

    Shows the high-frequency structure exists in the raw geometry but is erased by
    sigma_smooth BEFORE the encoder: at the as-trained sigma=10 the rose collapses to
    ~2-3 harmonics (8-harmonic cut harmless) AND turbines become indistinguishable
    (disc≈0); at sigma=2 it needs ~15 harmonics and turbines separate (disc rises).
    """
    groups = build_groups(args, wt, D)
    results = {s: {} for s in sigmas}        # results[sigma][group] = (tail, k90, recon, disc)
    rows = []
    for s in sigmas:
        for gname, layouts in groups.items():
            tfs, k90s, res, discs = [], [], [], []
            for (_lname, x, y) in layouts:
                rec, _inf = compute_layout_profiles_vectorized(
                    np.asarray(x, float), np.asarray(y, float), rotor_diameter=D,
                    k_wake=args.k_wake, n_directions=args.n_dir,
                    sigma_smooth=s, scale_factor=args.scale_factor, mode=geom_mode)
                discs.append(within_farm_discriminability(rec))
                for t in range(rec.shape[0]):
                    m = rose_metrics(rec[t], K=args.K)
                    tfs.append(m["tail_frac"]); k90s.append(m["k90"]); res.append(m["recon_err"])
            disc = float(np.nanmean(discs)) if discs else float("nan")
            results[s][gname] = (float(np.mean(tfs)), float(np.median(k90s)), float(np.mean(res)), disc)
            rows.append(dict(geom_mode=geom_mode, sigma_smooth=s, group=gname,
                             tail_frac=results[s][gname][0], k90=results[s][gname][1],
                             recon_err=results[s][gname][2], disc=disc))
        print(f"[sigma_sweep:{geom_mode}] sigma={s} done")
    return results, list(groups.keys()), rows


def print_sweep(results_by_mode, group_names, K):
    print("\n" + "=" * 78)
    print(f"SIGMA_SMOOTH SWEEP (geometric; receptivity; K={K}). tail = energy in harm > K; "
          f"disc = within-farm turbine discriminability")
    print("=" * 78)
    for mode, results in results_by_mode.items():
        sigmas = sorted(results.keys(), reverse=True)
        print(f"\n--- geom_mode = {mode} ---")
        print(f"{'sigma':>7} | " + "".join(f"{g[:11]:>13}" for g in group_names))
        for label, idx in (("tail_frac", 0), ("k90", 1), ("recon_err", 2), ("disc", 3)):
            print(f"-- {label} --")
            for s in sigmas:
                cells = "".join(f"{results[s][g][idx]:>13.4f}" for g in group_names)
                star = "  (as-trained)" if abs(s - 10.0) < 1e-9 else ""
                print(f"{s:>7} | {cells}{star}")
    print("=" * 78 + "\n")


def make_sweep_figure(results_by_mode, group_names, K, path):
    modes = list(results_by_mode.keys())
    primary = results_by_mode[modes[0]]
    sigmas = sorted(primary.keys(), reverse=True)
    fig, axes = plt.subplots(1, 4, figsize=(23, 5.5))
    titles = (f"tail-energy fraction (harmonics > {K})",
              "harmonics for 90% AC energy (k90)",
              f"low-pass-{K} reconstruction error",
              "within-farm turbine discriminability")
    # Panels 0-2: spectral metrics for the primary mode. Panel 3 (disc): overlay all modes.
    for gname in group_names:
        color = GROUP_COLORS.get(gname, "gray")
        for j in range(3):
            axes[j].plot(sigmas, [primary[s][gname][j] for s in sigmas], "o-",
                         color=color, label=gname if j == 0 else None)
        for mode in modes:
            res = results_by_mode[mode]
            axes[3].plot(sigmas, [res[s][gname][3] for s in sigmas],
                         marker="o", ls=MODE_STYLE.get(mode, "-"), color=color,
                         label=f"{gname}/{mode}" if gname == group_names[0] or mode != modes[0] else None)
    for j, title in enumerate(titles):
        axes[j].set_xscale("log")
        axes[j].invert_xaxis()                      # sharper roses to the right
        axes[j].axvline(10, color="k", ls=":", lw=1)
        axes[j].text(10, axes[j].get_ylim()[1], " as-trained", fontsize=8, va="top")
        axes[j].set_xlabel("sigma_smooth (bins)")
        axes[j].set_title(title, fontsize=10)
        axes[j].grid(True, alpha=0.3)
    axes[1].axhline(K, color="red", ls="--", lw=1)
    axes[1].text(axes[1].get_xlim()[0], K, f" budget K={K}", color="red", fontsize=8, va="bottom")
    axes[0].legend(fontsize=8, title="group")
    if len(modes) > 1:
        axes[3].legend(fontsize=7, title="group / mode (solid=wake, dashed=distance)")
    note = "primary mode shown in panels 1-3: " + modes[0]
    fig.suptitle("Geometric rose spectral content + discriminability vs smoothing — the "
                 f"binding low-pass is sigma_smooth, not the {K}-harmonic FFT cut  ({note})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=130)
    print(f"[out] wrote {path}")


# --------------------------------------------------------------------------- #
# Self-test: convention cross-checks (no heavy compute)
# --------------------------------------------------------------------------- #
def selftest(args, wt, D):
    print("\n[selftest] geometric redundancy + isolated-turbine + encoder-convention checks")
    # 1. tight pair -> sharp notch -> high tail; isolated single turbine -> flat -> ~0 tail
    pair_x, pair_y = np.array([0.0, 4 * D]), np.array([0.0, 0.0])
    rec_pair, inf_pair = roses_for(pair_x, pair_y, wt, D, "geometric", args.n_dir, args)
    solo_rec, _ = roses_for(np.array([0.0]), np.array([0.0]), wt, D, "geometric", args.n_dir, args)
    print(f"  tight-pair receptivity tail_frac = {rose_metrics(rec_pair[0], args.K)['tail_frac']:.4f} (expect > 0)")
    print(f"  isolated   receptivity tail_frac = {rose_metrics(solo_rec[0], args.K)['tail_frac']:.4f} (expect ~ 0)")
    # 2. geometric redundancy: influence == receptivity rolled 180 deg
    e, c = redundancy(rec_pair[0], inf_pair[0])
    print(f"  geometric redundancy: rel_err={e:.5f} corr={c:.5f} (expect err~0, corr~1)")
    # 3. FFT convention vs the model's encoder (DC + h1 magnitude must match)
    try:
        import torch
        from profile_encodings import FourierProfileEncoder
        enc = FourierProfileEncoder(embed_dim=16, n_harmonics=args.K)
        feats = enc.get_interpretable_features(torch.tensor(rec_pair[None], dtype=torch.float32))
        m = rose_metrics(rec_pair[0], args.K)
        print(f"  encoder dc={float(feats['dc'][0,0]):.4f} vs ours={m['dc']/args.n_dir:.4f}; "
              f"encoder h1={float(feats['h1_magnitude'][0,0]):.4f} vs ours={m['h1_mag']:.4f}")
    except Exception as exc:
        print(f"  [skip] encoder cross-check unavailable ({exc})")
    print("[selftest] done\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("geometric", "pywake", "both"), default="both")
    ap.add_argument("--n_dr", type=int, default=12, help="layouts sampled per DR group")
    ap.add_argument("--dr_lo", type=int, default=4, help="min turbines (v8 spanned 4..25)")
    ap.add_argument("--dr_hi", type=int, default=25, help="max turbines")
    ap.add_argument("--min_dist_D", type=float, default=3.0)
    ap.add_argument("--n_dir", type=int, default=180, help="rose resolution (DR used 180)")
    ap.add_argument("--K", type=int, default=K_MODEL, help="harmonic budget the model keeps")
    ap.add_argument("--k_wake", type=float, default=0.04)
    ap.add_argument("--sigma_smooth", type=float, default=10.0)
    ap.add_argument("--scale_factor", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(SCRIPT_DIR, "..", "figs"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sigma_sweep", default=None,
                    help="comma-separated sigma_smooth values, e.g. '10,5,2,0.5' (geometric)")
    ap.add_argument("--geom_mode", choices=("wake", "distance", "both"), default="wake",
                    help="geometric rose construction for the sweep")
    args = ap.parse_args()

    wt = get_turbine()
    D = wt.diameter()
    os.makedirs(args.outdir, exist_ok=True)

    if args.selftest:
        selftest(args, wt, D)
        return

    if args.sigma_sweep:
        sigmas = [float(s) for s in args.sigma_sweep.split(",")]
        modes = ["wake", "distance"] if args.geom_mode == "both" else [args.geom_mode]
        results_by_mode, all_rows, group_names = {}, [], None
        for m in modes:
            results, group_names, rows = run_sigma_sweep(args, wt, D, sigmas, geom_mode=m)
            results_by_mode[m] = results
            all_rows.extend(rows)
        print_sweep(results_by_mode, group_names, args.K)
        with open(os.path.join(args.outdir, "spectral_sigma_sweep.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["geom_mode", "sigma_smooth", "group",
                                              "tail_frac", "k90", "recon_err", "disc"])
            w.writeheader()
            w.writerows(all_rows)
        make_sweep_figure(results_by_mode, group_names, args.K,
                          os.path.join(args.outdir, "spectral_sigma_sweep.png"))
        return

    sources = ["geometric", "pywake"] if args.source == "both" else [args.source]
    rows, spectra, group_names = collect(args, wt, D, sources)
    print_summary(rows, sources, group_names, args.K)
    write_csv(rows, os.path.join(args.outdir, "spectral_diagnostic.csv"))
    make_figure(rows, spectra, sources, group_names, args.K,
                os.path.join(args.outdir, "spectral_diagnostic.png"))


if __name__ == "__main__":
    main()
