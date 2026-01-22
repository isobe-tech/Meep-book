import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import meep as mp
import numpy as np


def main() -> None:
    # Helper script that visualizes “a wave launches and is absorbed at boundaries.”
    # A few Ez snapshots are saved using the same minimal 2D setup.

    # Unit convention: 1 [meep length] = 1 mm
    c0 = 299_792_458.0  # [m/s]
    L0 = 1e-3  # [m]

    f0 = 2.45e9 * (L0 / c0)
    fwidth = 2.0e9 * (L0 / c0)

    sx, sy, dpml = 12.0, 8.0, 1.0
    resolution = 20
    cell = mp.Vector3(sx, sy, 0)

    src_x = -0.5 * sx + dpml + 1.0
    src = mp.Source(
        src=mp.GaussianSource(frequency=f0, fwidth=fwidth),
        component=mp.Ez,
        center=mp.Vector3(src_x, 0, 0),
        size=mp.Vector3(0, sy - 2 * dpml, 0),
    )

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(dpml)],
        sources=[src],
        resolution=resolution,
    )

    snapshot_times = [40.0, 80.0, 120.0, 160.0]
    snapshots: list[np.ndarray] = []
    times: list[float] = []

    sim.init_sim()
    prev_t = 0.0
    for t in snapshot_times:
        sim.run(until=t - prev_t)
        prev_t = t
        ez = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
        snapshots.append(np.array(ez, copy=True))
        times.append(sim.meep_time())

    if not mp.am_master():
        return

    vmax = max(float(np.max(np.abs(s))) for s in snapshots)
    vmax = max(vmax, 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, sharey=True, constrained_layout=True)
    extent = (-0.5 * sx, 0.5 * sx, -0.5 * sy, 0.5 * sy)

    for ax, ez, t in zip(axes.ravel(), snapshots, times):
        im = ax.imshow(
            ez.T,
            extent=extent,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )

        # Mark the interior (analysis region) inside PML with dashed lines
        ax.add_patch(
            Rectangle(
                (-0.5 * sx + dpml, -0.5 * sy + dpml),
                sx - 2 * dpml,
                sy - 2 * dpml,
                fill=False,
                linestyle="--",
                linewidth=0.8,
                edgecolor="black",
                alpha=0.6,
            )
        )

        ax.set_title(f"t = {t:.0f}")
        ax.set_xlabel("x [meep]")
        ax.set_ylabel("y [meep]")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92, label=r"$E_z$")

    repo_dir = Path(__file__).resolve().parents[2]
    figs_dir = repo_dir / "figs" / "ch06"
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(figs_dir / "ch06_wave2d_snapshots.png"), dpi=200)
    plt.close(fig)

    print("done")


if __name__ == "__main__":
    main()
