"""Draw the report figures from outputs/report/ into report/figures/ as PDF."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from indoor_loc import filters  # noqa: E402
from indoor_loc import floorplan as fp  # noqa: E402
from indoor_loc import layouts as ly  # noqa: E402
from plot_map import draw_floorplan  # noqa: E402
from run_filter import truth_at  # noqa: E402
from report_experiments import CROWDS, EXAMPLE, OUT, SWEEP_CROWDS, SWEEPS, log_path  # noqa: E402

FIGURES = ROOT / "report" / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {  # Fixed categorical order, one hue per filter.
    "plain": "#2a78d6",
    "multi_wall": "#eb6834",
    "fingerprint_3m": "#1baf7a",
    "fingerprint_6m": "#6250d6",
    "fingerprint_12m": "#e87ba4",
}
BASELINE_COLOR = "#8f8e88"
NAMES = {
    "plain": "Plain path loss",
    "multi_wall": "Multi-wall",
    "fingerprint_3m": "Fingerprint 3 m",
    "fingerprint_6m": "Fingerprint 6 m",
    "fingerprint_12m": "Fingerprint 12 m",
    "ble_only": "BLE centroid (no filter)",
    "pdr_only": "Steps only (no radio)",
}
TRUTH = "#0b0b0b"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 1.6, "legend.frameon": False, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def floorplan_figure():
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    draw_floorplan(ax)
    tags = ly.layout_e()
    ax.scatter([t.x for t in tags], [t.y for t in tags], s=10, c=COLORS["plain"], zorder=3, label="BLE tag (100)")
    xs = np.arange(1.5, fp.WIDTH, 6.0)
    ys = np.arange(1.5, fp.DEPTH, 6.0)
    gx, gy = np.meshgrid(xs, ys)
    ax.scatter(gx, gy, s=1.5, c=MUTED, zorder=2, label="Survey point (6 m grid)")
    ax.set_xlabel("x (m)")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    save(fig, "floorplan")


def model_maps_figure(survey):
    """Expected RSSI of one room tag under each measurement model."""
    tag = int(np.argmin(np.hypot(survey["tags"][:, 0] - 52.0, survey["tags"][:, 1] - 10.0)))
    xs, ys = np.arange(0.25, fp.WIDTH, 0.5), np.arange(0.25, fp.DEPTH, 0.5)
    grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
    window = (grid[:, 0] < 110)
    points = grid[window]
    panels = [("plain", "Plain path loss"), ("multi_wall", "Multi-wall"), ("fingerprint_6m", "Fingerprint map, 6 m")]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.6), sharex=True)
    for ax, (method, title) in zip(axes, panels):
        pf = filters.make(method, survey)
        pf.xy = points
        value = pf.expected_rssi(np.array([tag]))[:, 0].reshape(len(ys), -1)
        image = ax.imshow(value, origin="lower", extent=(0, 110, 0, fp.DEPTH), cmap="Blues", vmin=-95, vmax=-55)
        for wall in fp.build_walls():
            ax.plot([wall.x0, wall.x1], [wall.y0, wall.y1], color=INK, lw=0.6)
        ax.plot(*survey["tags"][tag, :2], marker="*", color="#eb6834", ms=9, mec="white")
        ax.set_xlim(0, 110)
        ax.set_title(title, loc="left")
        ax.set_ylabel("y (m)")
        ax.grid(False)
    axes[-1].set_xlabel("x (m)")
    fig.colorbar(image, ax=axes, shrink=0.6, label="Expected RSSI (dBm)")
    save(fig, "model_maps")


def example_figures():
    crowd, seed = EXAMPLE
    file = np.load(log_path(crowd, seed))
    log = {k: file[k] for k in ("truth", "dt", "heading")}
    runs = {m: dict(np.load(OUT / f"example_{m}.npz")) for m in ("plain", "multi_wall", "fingerprint_6m")}

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=True)
    for ax, (method, run) in zip(axes, runs.items()):
        draw_floorplan(ax)
        for text in list(ax.texts):
            text.remove()
        ax.plot(*log["truth"].T, color=TRUTH, lw=1.0, label="True path")
        ax.plot(*run["position"].T, color=COLORS[method], lw=1.0, label="Filter estimate")
        error = np.hypot(*(run["position"] - truth_at(log, run["time"])[0]).T)
        ax.set_title(f"{NAMES[method]}: median error {np.median(error[run['time'] >= 30]):.2f} m", loc="left")
        ax.grid(False)
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.08), ncol=2)
    axes[-1].set_xlabel("x (m)")
    save(fig, "trajectories")

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for method, run in runs.items():
        error = np.hypot(*(run["position"] - truth_at(log, run["time"])[0]).T)
        ax.plot(run["time"], error, color=COLORS[method], lw=1.0, label=NAMES[method])
    run = runs["fingerprint_6m"]
    ax.fill_between(run["time"], 0, run["spread"], color=COLORS["fingerprint_6m"], alpha=0.15, lw=0,
                    label="Fingerprint 6 m particle spread")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (m)")
    ax.set_ylim(0, 15)
    ax.set_xlim(0, run["time"][-1])
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "error_time")

    cloud = runs["fingerprint_6m"]["cloud"]
    times = runs["fingerprint_6m"]["time"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, at in zip(axes, (1.0, 5.0, 30.0)):
        n = int(np.argmin(np.abs(times - at)))
        true_xy = truth_at(log, times[n:n + 1])[0][0]
        draw_floorplan(ax)
        for text in list(ax.texts):
            text.remove()
        ax.scatter(*cloud[n].T, s=2, c=COLORS["fingerprint_6m"], alpha=0.6, lw=0, label="Particles")
        ax.plot(*true_xy, marker="x", color=TRUTH, ms=7, mew=1.6, label="True position")
        ax.plot(*runs["fingerprint_6m"]["position"][n], marker="o", mfc="none", mec="#eb6834", ms=7, mew=1.4,
                label="Estimate")
        ax.set_xlim(true_xy[0] - 35, true_xy[0] + 35)
        ax.set_ylim(-2, 62)
        ax.set_title(f"t = {times[n]:.0f} s", loc="left")
        ax.set_ylabel("")
        ax.grid(False)
    axes[0].set_ylabel("y (m)")
    axes[1].set_xlabel("x (m)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=3)
    save(fig, "particles")


def cdf_figure():
    errors = np.load(OUT / "main_errors.npz")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for method in list(COLORS) + ["ble_only", "pdr_only"]:
        pooled = np.sort(np.concatenate([errors[k] for k in errors.files if k.startswith(f"{method}__")]))
        style = dict(color=COLORS.get(method, BASELINE_COLOR), ls="-" if method in COLORS else ("--" if method == "ble_only" else ":"))
        ax.plot(pooled, np.linspace(0, 1, len(pooled)), label=NAMES[method], **style)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("Fraction of time below")
    ax.legend(loc="lower right")
    save(fig, "cdf")


def crowd_figure(summary):
    methods = list(COLORS)
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    width = 0.16
    x = np.arange(len(CROWDS))
    for i, method in enumerate(methods):
        values = [summary[f"{method}/{c}"]["median_m"] for c in CROWDS]
        ax.bar(x + (i - 2) * width, values, width * 0.9, color=COLORS[method], label=NAMES[method], zorder=2)
    ax.set_xticks(x, [c.capitalize() for c in CROWDS])
    ax.set_xlabel("Crowd level (0, 300, 800, 2000 other people)")
    ax.set_ylabel("Median error (m)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    save(fig, "crowd")


def sweep_figure(sweeps):
    labels = {
        "particles": "Number of particles",
        "ble_sigma_db": r"Measurement noise $\sigma$ (dB)",
        "ble_top_k": "Strongest tags used per scan, $K$",
        "offset_shrink": r"Offset removal share $\beta$",
    }
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4))
    for ax, (setting, values) in zip(axes.flat, SWEEPS.items()):
        for crowd, style in zip(SWEEP_CROWDS, ("-", "--")):
            y = [sweeps[f"{setting}={v}/{crowd}"]["mean_of_run_medians_m"] for v in values]
            worst = [sweeps[f"{setting}={v}/{crowd}"]["worst_run_median_m"] for v in values]
            ax.plot(values, y, style, marker="o", ms=4, color=COLORS["fingerprint_6m"], label=f"{crowd.capitalize()} crowd: mean")
            ax.plot(values, worst, style, marker="^", ms=4, color=BASELINE_COLOR, label=f"{crowd.capitalize()} crowd: worst run")
        if setting == "particles":
            ax.set_xscale("log")
        ax.set_xticks(values, [str(v) for v in values])
        ax.minorticks_off()
        ax.set_xlabel(labels[setting])
        ax.set_ylabel("Median error (m)")
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(loc="upper center", bbox_to_anchor=(1.1, 1.42), ncol=2)
    fig.tight_layout()
    save(fig, "sweeps")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    survey = dict(np.load(ROOT / "data" / "survey.npz"))
    summary = json.loads((OUT / "main_summary.json").read_text())
    sweeps = json.loads((OUT / "sweep_summary.json").read_text())
    floorplan_figure()
    model_maps_figure(survey)
    example_figures()
    cdf_figure()
    crowd_figure(summary)
    sweep_figure(sweeps)


if __name__ == "__main__":
    main()
