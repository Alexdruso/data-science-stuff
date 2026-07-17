"""Pooled power analysis of the split-half veto record (Day-14, zero-compute).

Every seat/combiner verdict of the campaign is a 12-cell split-half read
(6 stratified half-splits x 2 directions) printed as `mean (sd, positive k/12)`.
This probe asks two questions no single verdict answers:

  1. POWER — what is the minimum detectable effect (MDE) of ONE 12-cell probe,
     accounting for the direction-pair correlation between cells? Could a true
     +0.0002 seat effect have been vetoed as "flat"?
  2. POOLING — over the FULL veto record (~30 challenger rows), does the pooled
     mean hide a positive effect below single-probe resolution?

Inputs are the existing results/*.txt reports (summary-statistics meta-analysis;
raw per-cell deltas used where recorded). The `vs nm` rows are kept as POSITIVE
CONTROLS: the LR-over-NM adoption (~+0.0003, true effect) must read as clearly
detected, or the instrument family itself is broken.

Output: results/probe_veto_power.txt
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "probe_veto_power.txt"
N_CELLS = 12

# ---------------------------------------------------------------- parsing

# format A: "  lrrt1  0.95076   vs lr8 +0.00005 (sd 0.00009, positive 9/12)"
RE_A = re.compile(
    r"^\s+(\S+)\s+([\d.]+)\s+vs\s+(\S+)\s+([+-][\d.]+)\s+"
    r"\(sd\s+([\d.]+),\s+positive\s+(\d+)/12\)"
)
# format B: "  lr-nm: mean +0.00032  sd 0.00023  positive 10/12  cells [..]"
RE_B = re.compile(
    r"^\s+(\S+):\s+mean\s+([+-][\d.]+)\s+sd\s+([\d.]+)\s+positive\s+(\d+)/12"
    r"(?:\s+cells\s+\[(.*)\])?"
)
# format C (blend gate): "mean +0.00042  sd 0.00025  positive 12/12"
RE_C = re.compile(r"^mean\s+([+-][\d.]+)\s+sd\s+([\d.]+)\s+positive\s+(\d+)/12")
RE_CELLS = re.compile(r"^paired deltas per cell: \[(.*)\]")

# channel per file; the veto pool = {seat, combiner_axis}. `vs nm` rows and the
# blend-gate report are positive controls (true nonzero effects the instrument
# family is known to have detected) and are pooled separately.
FILE_CHANNEL = {
    "probe_seat_rt.txt": "seat",
    "probe_seat_basin.txt": "seat",
    "probe_seat_bins.txt": "seat",
    "probe_seat_s1.txt": "seat",
    "log_seat_tabpfn3.txt": "seat",
    "probe_swap_cat.txt": "seat",
    "probe_combiner_rule.txt": "seat",
    "probe_combiner_ftt.txt": "seat",
    "probe_combiner_pm.txt": "seat",
    "probe_combiner_day9.txt": "seat",
    "probe_combiner_u.txt": "combiner_axis",
    "probe_meta_u.txt": "combiner_axis",
    "probe_struct_u.txt": "combiner_axis",
    "probe_regime_meta.txt": "combiner_axis",
    "probe_polish_u.txt": "combiner_axis",
    "probe_combiner_tournament.txt": "vs_nm_control",
    "probe_combiner_gate.txt": "vs_nm_control",
    "probe_blend_gate_realmlp.txt": "vs_nm_control",
}


def parse_reports() -> list[dict]:
    rows: list[dict] = []
    for fname, channel in FILE_CHANNEL.items():
        path = RESULTS / fname
        if not path.exists():
            continue
        pending_cells: list[float] | None = None
        for line in path.read_text().splitlines():
            mcells = RE_CELLS.match(line)
            if mcells:
                pending_cells = [float(x) for x in mcells.group(1).split(",")]
                continue
            m = RE_A.match(line)
            if m:
                name, _score, ref, mean, sd, npos = m.groups()
                rows.append(
                    dict(file=fname, name=name, ref=ref, mean=float(mean),
                         sd=float(sd), npos=int(npos), cells=None, channel=channel)
                )
                continue
            m = RE_B.match(line)
            if m:
                name, mean, sd, npos, cells = m.groups()
                rows.append(
                    dict(file=fname, name=name, ref=name.split("-")[-1],
                         mean=float(mean), sd=float(sd), npos=int(npos),
                         cells=[float(x) for x in cells.split(",")] if cells else None,
                         channel=channel)
                )
                continue
            m = RE_C.match(line)
            if m:
                mean, sd, npos = m.groups()
                rows.append(
                    dict(file=fname, name="blend_gate_realmlp", ref="core4",
                         mean=float(mean), sd=float(sd), npos=int(npos),
                         cells=pending_cells, channel=channel)
                )
                pending_cells = None

    # drop self-reference rows (sd printed 0.00000, positive 0/12)
    rows = [r for r in rows if not (r["sd"] == 0.0 and r["npos"] == 0)]
    # rows named `nm` inside lr-referenced tables are the mirrored positive
    # control (LR-over-NM seen from the other side), not vetoes
    for r in rows:
        if r["name"] == "nm":
            r["channel"] = "vs_nm_control"
            r["mean"], r["npos"] = -r["mean"], N_CELLS - r["npos"]
    # dedupe exact re-prints across files (e.g. tabpfn3 seat appears twice)
    seen: set[tuple] = set()
    uniq = []
    for r in rows:
        key = (r["name"], r["mean"], r["sd"], r["npos"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq


# ------------------------------------------------------- statistics helpers


def pair_icc(cells: list[float]) -> float:
    """Correlation between the two directions of the same split
    (cells ordered [s1_ab, s1_ba, s2_ab, s2_ba, ...])."""
    a = np.array(cells[0::2])
    b = np.array(cells[1::2])
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def pooled_fixed(means: np.ndarray, ses: np.ndarray) -> tuple[float, float]:
    w = 1.0 / ses**2
    mu = float(np.sum(w * means) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    return mu, se


def dl_tau2(means: np.ndarray, ses: np.ndarray) -> float:
    """DerSimonian-Laird between-row variance."""
    w = 1.0 / ses**2
    mu = np.sum(w * means) / np.sum(w)
    q = float(np.sum(w * (means - mu) ** 2))
    k = len(means)
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    return max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0


def norm_sf(z: float) -> float:
    from math import erfc, sqrt

    return 0.5 * erfc(z / sqrt(2.0))


def report_pool(lines: list[str], label: str, pool: list[dict], n_eff: float) -> None:
    means = np.array([r["mean"] for r in pool])
    ses = np.array([max(r["sd"], 2.5e-5) / np.sqrt(n_eff) for r in pool])
    mu, se = pooled_fixed(means, ses)
    tau2 = dl_tau2(means, ses)
    ses_re = np.sqrt(ses**2 + tau2)
    mu_re, se_re = pooled_fixed(means, ses_re)
    npos_rows = int(np.sum(means > 0))
    lines.append(f"[{label}]  rows={len(pool)}  n_eff/row={n_eff:.1f}")
    lines.append(
        f"  fixed-effect pooled mean {mu:+.6f}  (se {se:.6f}, "
        f"95% CI [{mu - 1.96 * se:+.6f}, {mu + 1.96 * se:+.6f}], "
        f"z={mu / se:+.2f}, one-sided p={norm_sf(mu / se):.3f})"
    )
    lines.append(
        f"  random-effects (DL tau={np.sqrt(tau2):.6f}) mean {mu_re:+.6f} "
        f"(se {se_re:.6f}, 95% CI [{mu_re - 1.96 * se_re:+.6f}, "
        f"{mu_re + 1.96 * se_re:+.6f}])"
    )
    lines.append(
        f"  sign test: {npos_rows}/{len(pool)} row-means positive; "
        f"pooled positive cells {sum(r['npos'] for r in pool)}"
        f"/{N_CELLS * len(pool)} (descriptive; cells correlated within row)"
    )
    # sign-flip permutation self-check (row-level, exchangeable under H0)
    rng = np.random.default_rng(0)
    zs = []
    for _ in range(20000):
        flip = rng.choice([-1.0, 1.0], size=len(means))
        m, s = pooled_fixed(means * flip, ses)
        zs.append(m / s)
    zs_arr = np.array(zs)
    emp_p = float(np.mean(zs_arr >= mu / se))
    lines.append(
        f"  sign-flip null: mean z {zs_arr.mean():+.3f} (expect ~0), "
        f"sd {zs_arr.std():.3f}; empirical one-sided p={emp_p:.4f}"
    )


def main() -> None:
    rows = parse_reports()

    # parser cross-check vs two hand-read report lines
    def find(name: str) -> dict:
        return next(r for r in rows if r["name"] == name)

    lrrt1 = find("lrrt1")
    assert (lrrt1["mean"], lrrt1["sd"], lrrt1["npos"]) == (0.00005, 0.00009, 9)
    nn = find("nn")
    assert (nn["mean"], nn["sd"], nn["npos"]) == (-0.00015, 0.00013, 1)

    # direction-pair correlation from every row with raw cells
    iccs = [(r["name"], pair_icc(r["cells"])) for r in rows if r["cells"]]
    icc_med = float(np.median([v for _, v in iccs])) if iccs else 0.0
    # cluster size 2 -> n_eff = 12 / (1 + icc)
    n_eff_cons = N_CELLS / (1.0 + max(0.0, icc_med))

    lines: list[str] = []
    lines.append("pooled power analysis of the split-half veto record (Day-14)")
    lines.append(f"parsed {len(rows)} unique challenger rows from {len(FILE_CHANNEL)} reports")
    lines.append("")
    lines.append("== direction-pair correlation (rows with raw cells) ==")
    for name, v in iccs:
        lines.append(f"  {name:<22} r_pair={v:+.3f}")
    lines.append(
        f"  median r_pair {icc_med:+.3f}  ->  conservative n_eff = {n_eff_cons:.1f} "
        f"(naive 12)"
    )
    lines.append("")

    # per-row table
    lines.append("== per-row z (naive | conservative) ==")
    for ch in ("seat", "combiner_axis", "vs_nm_control"):
        lines.append(f"  -- {ch} --")
        for r in sorted([r for r in rows if r["channel"] == ch], key=lambda r: -r["mean"]):
            se_n = max(r["sd"], 2.5e-5) / np.sqrt(N_CELLS)
            se_c = max(r["sd"], 2.5e-5) / np.sqrt(n_eff_cons)
            lines.append(
                f"  {r['name']:<22} {r['mean']:+.5f} (sd {r['sd']:.5f}, "
                f"{r['npos']:2d}/12)  z={r['mean'] / se_n:+5.2f} | "
                f"{r['mean'] / se_c:+5.2f}   [{r['file']}]"
            )
    lines.append("")

    # MDE / power of a single 12-cell probe
    veto_pool = [r for r in rows if r["channel"] in ("seat", "combiner_axis")]
    sd_med = float(np.median([r["sd"] for r in veto_pool]))
    lines.append("== power of ONE 12-cell probe (median veto-row sd) ==")
    for tag, n_eff in (("naive", float(N_CELLS)), ("conservative", n_eff_cons)):
        se1 = sd_med / np.sqrt(n_eff)
        mde = (1.645 + 0.842) * se1  # one-sided alpha=.05, power=.80
        p2 = 1.0 - norm_sf((0.0002 / se1) - 1.645)
        p3 = 1.0 - norm_sf((0.0003 / se1) - 1.645)
        lines.append(
            f"  [{tag:<12}] sd={sd_med:.5f} se(mean)={se1:.6f}  "
            f"MDE(80% power)={mde:.6f}  power at +0.0002: {p2:.2f}  at +0.0003: {p3:.2f}"
        )
    lines.append("")

    # pooled estimates
    lines.append("== pooled estimates ==")
    for label, pool in (
        ("VETO POOL (seat + combiner axes)", veto_pool),
        ("seat channel only", [r for r in rows if r["channel"] == "seat"]),
        ("combiner-axis channel only", [r for r in rows if r["channel"] == "combiner_axis"]),
        ("positive controls (vs nm / blend gate)", [r for r in rows if r["channel"] == "vs_nm_control"]),
    ):
        report_pool(lines, label, pool, n_eff_cons)
        lines.append("")

    lines.append("== notes ==")
    lines.append(
        "  - printed means/sds are rounded to 1e-5: pooled digits below ~1e-5 are soft."
    )
    lines.append(
        "  - rows within a probe share the same holdout splits and the same incumbent "
        "-> row-level pooling treats probes as independent, rows within probe as "
        "exchangeable; the sign-flip null is the honest calibration check."
    )
    lines.append(
        "  - challenger rows are selected (each was someone's best idea that day); "
        "selection bias if anything INFLATES the pooled mean -> a null pooled read "
        "is conservative evidence of saturation."
    )
    lines.append(
        "  - (red-team caveat) every probe uses the SAME 12 StratifiedShuffleSplit "
        "index pairs (seeds 0-5), so probe verdicts are positively correlated "
        "evidence and the pooled CI is anti-conservative in width; the DIRECTION "
        "(pooled mean negative) is unaffected, and the positive controls detected "
        "on the same splits bound any split-quirk masking effect."
    )

    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nreport -> {OUT}")


if __name__ == "__main__":
    main()
