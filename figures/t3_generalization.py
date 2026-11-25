"""Draft: one figure for both generalization sweeps (slope and material), sharing the metric rows.

Columns: slope sweep on firm sand (downhill to uphill) | material ladder on level ground.
Rows: success rate, velocity tracking error, and optionally step time.

It replots the data behind figures/t3_slope_full_final3.pdf and figures/t3_material_ladder_final2.pdf; it does
not compute anything new. Style follows scripts/benchmarks/t3/paper_figures.py of the T3 session.

    uv run --with matplotlib --with numpy python figures/t3_generalization.py OUT.pdf [--step_time]
    uv run --with matplotlib --with numpy python figures/t3_generalization.py --table   # rows of the table
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T3 = Path.home() / "isaac/worktrees/mpm-unification/IsaacLab/logs/t3/final_1"
GEOMETRY = T3 / "d50/geometry/geometry.json"  # slope sweep, 0.50 m bed
LADDER = T3 / "ladderT/behaviour/material_behaviour.json"  # material ladder, setting T, 0.50 m bed

WIDTH = 3.45  # one column of the paper [in]
POLICIES = (("mjwarp", "MJWarp", "#D55E00", "o"), ("rft_deep", "RFT", "#009E73", "s"), ("mpm_retrain", "MPM", "#0072B2", "^"))
TRAIN_SLOPE_DEG = math.degrees(math.atan(0.4))
GROUND_NAMES = {
    "hard ground": "hard",
    "dense_sand": "dense",
    "beach_sand": "beach",
    "poppy_seed_packed": "packed",
    "light_sand": "light",
    "poppy_seed_loose": "loose",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.titlepad": 3.0,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    }
)


def finish(axis):
    axis.grid(True, which="major", linestyle="--", linewidth=0.35, alpha=0.5)
    axis.tick_params(length=2, pad=1.5)
    axis.xaxis.labelpad = 1.5
    axis.yaxis.labelpad = 1.5
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def draw(axis, xs, mean, std, color, marker):
    mean, std = np.asarray(mean, float), np.asarray(std, float)
    ok = ~np.isnan(mean)
    axis.plot(xs[ok], mean[ok], color=color, marker=marker, markersize=2.8, linewidth=0.9)
    if np.any(~np.isnan(std)):
        axis.fill_between(xs[ok], (mean - std)[ok], (mean + std)[ok], color=color, alpha=0.15, linewidth=0)


TABLE_ANGLES = (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0)
TABLE_GROUNDS = {
    "hard ground": "Hard ground",
    "dense_sand": "Dense sand",
    "beach_sand": "Beach sand",
    "poppy_seed_packed": "Packed poppy",
    "light_sand": "Light sand",
    "poppy_seed_loose": "Loose poppy",
}


def table():
    """Print the rows of the table in the paper (success rate, velocity error, step time), from the same data."""
    slope = json.loads(GEOMETRY.read_text())["cells"]
    ladder = json.loads(LADDER.read_text())
    cells = {(c["policy"], c["ground"]): c for c in ladder["cells"]}
    labels = [p[0] for p in POLICIES]

    def line(name, goals, trials, error, step):
        rate = [f"{round(100 * g / t):d}" for g, t in zip(goals, trials)]
        fmt = lambda xs: [f"{x:.2f}" if g > 0 and x == x else "--" for x, g in zip(xs, goals)]  # noqa: E731
        return f"        {name} & " + " & ".join(rate + fmt(error) + fmt(step)) + r" \\"

    print("% slope rows")
    for angle in TABLE_ANGLES:
        row = [next(c for c in slope if c["policy"] == p and c["angle_deg"] == angle) for p in labels]
        name = f"${angle:+.0f}^\\circ$" if angle else "$0^\\circ$"
        print(line(name, [c["goals"] for c in row], [c["trials"] for c in row], [c["error"][0] for c in row], [c["step_time"][0] for c in row]))
    print("% material rows")
    for ground, name in TABLE_GROUNDS.items():
        row = [cells[(p, ground)] for p in labels]
        get = lambda c, k: c[k]["mean"] if isinstance(c.get(k), dict) and c[k].get("mean") is not None else float("nan")  # noqa: E731
        print(line(name, [c["goals"] for c in row], [c["trials"] for c in row], [get(c, "error_m_s") for c in row], [get(c, "step_time_s") for c in row]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--step_time", action="store_true", help="Add a third row with the step time.")
    parser.add_argument("--table", action="store_true", help="Print the LaTeX rows of the table and exit.")
    args = parser.parse_args()
    if args.table:
        table()
        return

    slope = json.loads(GEOMETRY.read_text())["cells"]
    ladder = json.loads(LADDER.read_text())
    grounds = ladder["grounds"]
    cells = {(c["policy"], c["ground"]): c for c in ladder["cells"]}
    angles = np.array(sorted({c["angle_deg"] for c in slope}))
    gx = np.arange(len(grounds))

    # row: (label, slope key, ladder key)
    rows = [("Success rate", "goals", "goal_share"), ("Lin. vel.\nerror [m/s]", "error", "error_m_s")]
    if args.step_time:
        rows.append(("Step time [s]", "step_time", "step_time_s"))

    fig, axes = plt.subplots(len(rows), 2, figsize=(WIDTH, 0.72 * len(rows) + 0.72), sharex="col", sharey="row")
    for r, (ylabel, skey, lkey) in enumerate(rows):
        left, right = axes[r]
        for label, _name, color, marker in POLICIES:
            mine = sorted((c for c in slope if c["policy"] == label), key=lambda c: c["angle_deg"])
            if skey == "goals":
                mean, std = [c["goals"] / c["trials"] for c in mine], [np.nan] * len(mine)
            else:  # [mean, std, number of trials]; no value where no trial gave one
                mean = [c[skey][0] if c[skey][2] > 0 else np.nan for c in mine]
                std = [c[skey][1] if c[skey][2] > 0 else np.nan for c in mine]
            draw(left, np.array([c["angle_deg"] for c in mine]), mean, std, color, marker)

            def value(ground, part):
                v = cells.get((label, ground), {}).get(lkey)
                return np.nan if not isinstance(v, dict) or v.get(part) is None else v[part]

            mean = [value(g, "mean") for g in grounds]
            std = [np.nan] * len(grounds) if lkey == "goal_share" else [value(g, "std") for g in grounds]
            draw(right, gx, mean, std, color, marker)

        left.axvspan(-TRAIN_SLOPE_DEG, TRAIN_SLOPE_DEG, color="0.5", alpha=0.12, linewidth=0)
        left.set_ylabel(ylabel)
        left.yaxis.label.set_size(6.5)
        if skey == "goals":
            left.set_ylim(0.0, 1.08)
        else:
            left.set_ylim(0.0, None)
        finish(left)
        finish(right)

    axes[0, 0].set_title("Slope (firm sand)")
    axes[0, 1].set_title("Material (level ground)")
    axes[-1, 0].set_xticks(angles[::2])
    axes[-1, 0].set_xlim(angles[0] - 1.5, angles[-1] + 1.5)
    axes[-1, 0].set_xlabel("slope [deg] (< 0: downhill)")
    axes[-1, 1].set_xticks(gx)
    axes[-1, 1].set_xticklabels(
        [GROUND_NAMES.get(g, g) for g in grounds], fontsize=5.5, rotation=30, ha="right", rotation_mode="anchor"
    )
    axes[-1, 1].set_xlim(-0.35, len(grounds) - 0.65)
    axes[-1, 1].set_xlabel(r"ground, hard $\rightarrow$ soft")

    handles = [
        matplotlib.lines.Line2D([], [], color=color, marker=marker, markersize=2.8, linewidth=0.9, label=name)
        for _label, name, color, marker in POLICIES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=True,
        edgecolor="#bbbbbb",
        fancybox=False,
        columnspacing=1.0,
        handlelength=1.6,
        handletextpad=0.4,
        borderpad=0.3,
    )
    bottom = 0.105 if len(rows) == 2 else 0.085
    fig.tight_layout(rect=(0, bottom, 1, 1), pad=0.15, h_pad=0.35, w_pad=0.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=300)
    print("wrote", args.out, f"({fig.get_size_inches()[0]:.2f} x {fig.get_size_inches()[1]:.2f} in)")


if __name__ == "__main__":
    main()
