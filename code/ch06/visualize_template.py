import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp
import numpy as np


def main() -> None:
    # Minimal skeleton for the “check geometry and fields first” workflow.
    # plot2D is used to confirm PML/geometry placement, then ε and fields are written to HDF5, and finally fields are plotted.

    # Unit convention: 1 [meep length] = 1 mm
    c0 = 299_792_458.0  # [m/s]
    L0 = 1e-3  # [m]

    f0 = 2.45e9 * (L0 / c0)
    fwidth = 2.0e9 * (L0 / c0)

    sx, sy, dpml = 14.0, 8.0, 1.0
    resolution = 20
    cell = mp.Vector3(sx, sy, 0)

    # Example: dielectric slab with a local “notch” (later object overwrites earlier one)
    w = 2.0
    slab = mp.Block(
        material=mp.Medium(index=2.0),
        center=mp.Vector3(),
        size=mp.Vector3(mp.inf, w, mp.inf),
    )
    notch = mp.Block(
        material=mp.Medium(index=1.0),
        center=mp.Vector3(0.0, +0.6, 0.0),
        size=mp.Vector3(1.2, 1.2, mp.inf),
    )
    geometry = [slab, notch]

    x_src = -0.5 * sx + dpml + 1.0
    src = mp.Source(
        src=mp.GaussianSource(frequency=f0, fwidth=fwidth),
        component=mp.Ez,
        center=mp.Vector3(x_src, 0, 0),
        size=mp.Vector3(0, w, 0),
    )

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=[src],
        resolution=resolution,
    )

    # Write outputs under this repository, independent of the current working directory.
    repo_dir = Path(__file__).resolve().parents[2]
    output_dir = repo_dir / "out" / "ch06_visualize"
    figs_dir = repo_dir / "figs" / "ch06"
    output_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    sim.use_output_directory(str(output_dir))
    sim.filename_prefix = "viz"

    # Confirm geometry before running
    sim.init_sim()
    if mp.am_master():
        sim.plot2D()
        plt.savefig(str(output_dir / "viz_structure.png"), dpi=200, bbox_inches="tight")
        plt.savefig(str(figs_dir / "ch06_visualize_structure.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # Export as NumPy arrays for plotting
    eps = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    if mp.am_master():
        print(f"eps: shape={eps.shape}, min={np.min(eps):.3g}, max={np.max(eps):.3g}")

    # Write ε and Ez to HDF5, then plot Ez
    sim.run(
        mp.at_beginning(mp.output_epsilon),
        mp.at_end(mp.output_efield_z),
        until=200,
    )

    if mp.am_master():
        sim.plot2D(fields=mp.Ez)
        plt.savefig(str(output_dir / "viz_ez.png"), dpi=200, bbox_inches="tight")
        plt.savefig(str(figs_dir / "ch06_visualize_ez.png"), dpi=200, bbox_inches="tight")
        plt.close()

    if mp.am_master():
        print("done")


if __name__ == "__main__":
    main()
