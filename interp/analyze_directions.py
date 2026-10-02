"""Stage 2: find a "conservativeness" direction in the captured activations.

Per site (warmup + padded tokens dropped):
1. PCA scatter of token activations colored by per-turbine derate_cmd,
   del_ratio, del_limit (when it varies) and TURBINE INDEX -- the last panel
   decides whether one shared direction is enough or tokens cluster by
   turbine role.
2. Linear probes (ridge, grouped 5-fold CV): per-token -> derate_cmd; pooled
   masked-mean -> mean derate / del_ratio / del_limit. Shuffle-label control
   per probe (R^2 ~ 0 expected). Note final_norm -> derate_cmd is
   near-trivial (fc_mean reads it linearly); the interesting rows are the
   earlier sites and the OUTCOME labels.
3. Difference-of-means direction: conservative-token mean minus
   aggressive-token mean, where conservative = top quantile of the signed
   label (derate_cmd: high = conservative; del_limit / del_ratio inverted).
   Cosine similarity vs. the ridge-coefficient direction is stored.
4. Artifact per site: direction_{site}.npz with the unit vector, act_std
   (projection std -- steer_eval's sigma unit), class means and metadata.

Variance guard: aborts when the chosen label's IQR ~ 0. Expected on
delmax2x2_A_nopen (near-zero derate variance) -- keep A as a CONTROL, develop
on B_pen.

Usage (from TransformerSac/):
    ../.venv/bin/python interp/analyze_directions.py \
        data/interp/delmax2x2_B_pen_s1/capture_limnone_s0.npz
"""

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

_INTERP_DIR = os.path.dirname(os.path.abspath(__file__))
_TSAC_DIR = os.path.dirname(_INTERP_DIR)
_REPO_ROOT = os.path.dirname(_TSAC_DIR)
for _p in (_REPO_ROOT, _TSAC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Direction sign convention: +vector points toward MORE conservative behavior.
# derate_cmd: more derating = conservative. del_limit: a LOW limit forces
# conservatism. del_ratio: high ratio = running hot = aggressive.
LABEL_SIGN = {"derate_cmd": +1.0, "del_limit": -1.0, "del_ratio": -1.0}


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: probe + direction.")
    p.add_argument("captures", nargs="+",
                   help="capture_*.npz files (or dirs to glob them from).")
    p.add_argument("--sites", type=str, default=None,
                   help="Comma site names (default: every act_* in the npz).")
    p.add_argument("--label", type=str, default="derate_cmd",
                   choices=sorted(LABEL_SIGN),
                   help="Label defining the conservative/aggressive split.")
    p.add_argument("--quantile", type=float, default=0.25)
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--pca-max-points", type=int, default=20000)
    p.add_argument("--outdir", type=str, default=None,
                   help="Direction npz dir (default: <first capture dir>/directions).")
    p.add_argument("--figdir", type=str, default=None,
                   help="Default: TransformerSac/figs/interp/<run_name>/")
    return p.parse_args()


def resolve_captures(paths):
    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "capture_*.npz"))))
        else:
            files.append(path)
    if not files:
        raise SystemExit(f"No capture npz found under {paths}")
    return files


def load_tokens(files, sites):
    """Concatenate captures into token-level and step-level tables.

    Returns dict with:
      X[site]      (N_tok, d)   real, post-warmup tokens
      tok_y        per-token labels: derate_cmd, turbine_idx, and the
                   step-level del_ratio/del_limit/episode broadcast per token
      P[site]      (N_step, d)  masked-mean pooled activations
      step_y       step-level labels: mean_derate, del_ratio, del_limit, group
      meta         metadata json of the first file
    """
    X = {s: [] for s in sites}
    P = {s: [] for s in sites}
    tok = {k: [] for k in ("derate_cmd", "del_ratio", "del_limit",
                           "turbine_idx", "group")}
    step = {k: [] for k in ("mean_derate", "del_ratio", "del_limit", "group")}
    meta = None
    group_offset = 0
    for fi, path in enumerate(files):
        z = np.load(path, allow_pickle=False)
        if meta is None:
            meta = json.loads(str(z["metadata_json"]))
        keep = ~z["is_warmup"]
        real = ~z["mask"][keep]                      # (S, T) True = real token
        n_tok = real.sum(axis=1).clip(min=1)
        group = z["episode_idx"][keep] + group_offset
        group_offset = int(group.max()) + 1 if group.size else group_offset
        for s in sites:
            act = z[f"act_{s}"][keep]                # (S, T, d)
            X[s].append(act[real])                   # (N_tok, d)
            pooled = np.where(real[..., None], act, 0.0).sum(axis=1) / n_tok[:, None]
            P[s].append(pooled)
        dc = z["derate_cmd"][keep]
        tok["derate_cmd"].append(dc[real])
        tok["turbine_idx"].append(
            np.broadcast_to(np.arange(dc.shape[1]), dc.shape)[real])
        for key in ("del_ratio", "del_limit"):
            tok[key].append(np.broadcast_to(z[key][keep][:, None], dc.shape)[real])
        tok["group"].append(np.broadcast_to(group[:, None], dc.shape)[real])
        step["mean_derate"].append(
            np.nanmean(np.where(real, dc, np.nan), axis=1))
        step["del_ratio"].append(z["del_ratio"][keep])
        step["del_limit"].append(z["del_limit"][keep])
        step["group"].append(group)
    return (
        {s: np.concatenate(v) for s, v in X.items()},
        {k: np.concatenate(v) for k, v in tok.items()},
        {s: np.concatenate(v) for s, v in P.items()},
        {k: np.concatenate(v) for k, v in step.items()},
        meta,
    )


def grouped_folds(groups, n_folds=5):
    """Yield (train_idx, test_idx) splitting by group (episode); falls back to
    contiguous blocks when there are fewer groups than folds (single-episode
    smoke captures)."""
    uniq = np.unique(groups)
    n = len(groups)
    if len(uniq) < n_folds:
        print(f"    [warn] only {len(uniq)} episode group(s) < {n_folds} "
              "folds: CV falls back to contiguous blocks, which splits "
              "temporally-correlated tokens across train/test -- probe R^2 "
              "is optimistic on such small captures.")
    if len(uniq) >= n_folds:
        chunks = np.array_split(uniq, n_folds)
        for chunk in chunks:
            test = np.isin(groups, chunk)
            yield np.where(~test)[0], np.where(test)[0]
    else:
        order = np.arange(n)
        for chunk in np.array_split(order, n_folds):
            test = np.zeros(n, dtype=bool)
            test[chunk] = True
            yield np.where(~test)[0], np.where(test)[0]


def cv_r2(X, y, groups, alpha, seed=0, shuffle=False):
    """Grouped 5-fold CV R^2 of a ridge probe (optionally on shuffled labels)."""
    ok = np.isfinite(y)
    X, y, groups = X[ok], y[ok], groups[ok]
    if len(y) < 10 or np.subtract(*np.percentile(y, [75, 25])) < 1e-9:
        return np.nan
    if shuffle:
        y = np.random.default_rng(seed).permutation(y)
    scores = []
    for tr, te in grouped_folds(groups):
        model = Ridge(alpha=alpha).fit(X[tr], y[tr])
        scores.append(r2_score(y[te], model.predict(X[te])))
    return float(np.mean(scores))


def diff_of_means_direction(X, y, sign, quantile):
    """Unit conservative-minus-aggressive direction + projection stats."""
    ys = sign * y
    lo, hi = np.quantile(ys, [quantile, 1.0 - quantile])
    aggressive, conservative = ys <= lo, ys >= hi
    mu_c = X[conservative].mean(axis=0)
    mu_a = X[aggressive].mean(axis=0)
    v = mu_c - mu_a
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise RuntimeError("Degenerate direction (class means coincide)")
    v = v / norm
    act_std = float((X @ v).std())
    return v, act_std, mu_c, mu_a


def main():
    cli = parse_args()
    files = resolve_captures(cli.captures)
    probe = np.load(files[0], allow_pickle=False)
    all_sites = [k[len("act_"):] for k in probe.files if k.startswith("act_")]
    sites = cli.sites.split(",") if cli.sites else all_sites
    print(f"[analyze] {len(files)} capture file(s), sites={sites}")

    X, tok, P, step, meta = load_tokens(files, sites)
    run_name = meta["run_name"]
    outdir = cli.outdir or os.path.join(os.path.dirname(files[0]), "directions")
    figdir = cli.figdir or os.path.join(_TSAC_DIR, "figs", "interp", run_name)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    # ---- variance guard on the split label ----
    y_dir = tok[cli.label]
    finite = np.isfinite(y_dir)
    iqr = float(np.subtract(*np.percentile(y_dir[finite], [75, 25])))
    print(f"[analyze] N_tokens={len(y_dir)} label={cli.label} IQR={iqr:.5f}")
    if iqr < 1e-4:
        raise SystemExit(
            f"Label {cli.label!r} is (near-)constant (IQR={iqr:.2e}): no "
            "conservative/aggressive split exists in this capture. On "
            "delmax2x2_A_nopen this is the EXPECTED control result (no DEL "
            "penalty -> no derating variance); use B_pen to extract the "
            "direction, then test it on A."
        )

    limit_varies = (np.subtract(*np.percentile(
        tok["del_limit"][np.isfinite(tok["del_limit"])], [75, 25])) > 1e-6)

    # ---- per-site PCA scatters + probes + direction ----
    tok_probes = [("derate_cmd", tok["derate_cmd"])]
    pool_probes = [("mean_derate", step["mean_derate"]),
                   ("del_ratio", step["del_ratio"])]
    if limit_varies:
        pool_probes.append(("del_limit", step["del_limit"]))

    r2 = {}       # (site, probe_name) -> (real, shuffled)
    for site in sites:
        Xs, Ps = X[site], P[site]

        # PCA scatter
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(Xs))[: cli.pca_max_points]
        Xc = Xs[idx] - Xs.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        pcs = Xc @ Vt[:2].T
        panels = [("derate_cmd", tok["derate_cmd"][idx]),
                  ("del_ratio", tok["del_ratio"][idx]),
                  ("del_limit" if limit_varies else "del_ratio (dup)",
                   (tok["del_limit"] if limit_varies else tok["del_ratio"])[idx]),
                  ("turbine_idx", tok["turbine_idx"][idx])]
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        for ax, (name, c) in zip(axes.ravel(), panels):
            good = np.isfinite(c)  # del_ratio is NaN inside the DEL warm-up
            sc = ax.scatter(pcs[good, 0], pcs[good, 1], c=c[good], s=4,
                            alpha=0.6, cmap="viridis")
            fig.colorbar(sc, ax=ax, label=name)
            ax.set_title(f"{site}: PCA colored by {name}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, f"pca_{site}.png"), dpi=120)
        plt.close(fig)

        # Probes (token-level + pooled) with shuffle controls
        for name, y in tok_probes:
            r2[(site, f"tok:{name}")] = (
                cv_r2(Xs, y, tok["group"], cli.ridge_alpha),
                cv_r2(Xs, y, tok["group"], cli.ridge_alpha, shuffle=True))
        for name, y in pool_probes:
            r2[(site, f"pool:{name}")] = (
                cv_r2(Ps, y, step["group"], cli.ridge_alpha),
                cv_r2(Ps, y, step["group"], cli.ridge_alpha, shuffle=True))
        row = {k[1]: f"{v[0]:.3f} (shuf {v[1]:.3f})"
               for k, v in r2.items() if k[0] == site}
        print(f"  {site}: {row}")

        # Difference-of-means direction (+ ridge-coefficient cosine)
        ok = np.isfinite(y_dir)
        v, act_std, mu_c, mu_a = diff_of_means_direction(
            Xs[ok], y_dir[ok], LABEL_SIGN[cli.label], cli.quantile)
        ridge_w = Ridge(alpha=cli.ridge_alpha).fit(Xs[ok], y_dir[ok]).coef_
        ridge_w = LABEL_SIGN[cli.label] * ridge_w
        cos = float(v @ ridge_w / (np.linalg.norm(ridge_w) + 1e-12))
        np.savez(
            os.path.join(outdir, f"direction_{site}.npz"),
            vector=v.astype(np.float32),
            act_std=np.float32(act_std),
            mu_conservative=mu_c.astype(np.float32),
            mu_aggressive=mu_a.astype(np.float32),
            label=cli.label,
            quantile=np.float32(cli.quantile),
            site=site,
            cosine_vs_ridge=np.float32(cos),
            r2_token_probe=np.float32(r2[(site, "tok:derate_cmd")][0]),
            metadata_json=json.dumps({
                **meta, "stage": "direction",
                "captures": [os.path.abspath(f) for f in files]}),
        )
        print(f"    direction saved: act_std={act_std:.3f} "
              f"cos(diff-means, ridge)={cos:.3f}")

    # ---- probe R^2 bar chart across sites ----
    probe_names = sorted({k[1] for k in r2})
    fig, ax = plt.subplots(figsize=(2.2 + 1.6 * len(sites), 5))
    width = 0.8 / (len(probe_names) * 2)
    xs = np.arange(len(sites))
    for pi, pname in enumerate(probe_names):
        vals = [r2.get((s, pname), (np.nan,) * 2)[0] for s in sites]
        shuf = [r2.get((s, pname), (np.nan,) * 2)[1] for s in sites]
        off = (pi * 2 - len(probe_names)) * width
        ax.bar(xs + off, vals, width, label=pname)
        ax.bar(xs + off + width, shuf, width, color="lightgray",
               label="shuffled" if pi == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels(sites)
    ax.set_ylabel("5-fold CV $R^2$")
    finite_r2 = [v for v, _ in r2.values() if np.isfinite(v)]
    ax.set_ylim(max(-1.0, min([0.0] + finite_r2) - 0.05), 1.05)
    ax.axhline(0, color="k", lw=0.8)
    ax.legend(fontsize=8)
    ax.set_title(f"{run_name}: linear-probe $R^2$ by site (gray = shuffled control)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "probes_r2.png"), dpi=120)
    plt.close(fig)

    print(f"[analyze] directions -> {outdir}")
    print(f"[analyze] figures    -> {figdir}")


if __name__ == "__main__":
    main()
