import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp
import numpy as np


def main() -> None:
    # Helper script to visualize a minimal N2F example.
    # The far field of a point source (z polarization) is swept by angle to check the pattern shape.

    fcen = 0.15
    df = 0.10
    nfreq = 1

    sx = sy = sz = 10.0
    dpml = 1.0
    resolution = 8

    cell = mp.Vector3(sx, sy, sz)
    pml_layers = [mp.PML(dpml)]

    src = mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(),
    )

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell,
        boundary_layers=pml_layers,
        sources=[src],
    )

    d = 3.0
    l = 2 * d
    n2f = sim.add_near2far(
        fcen,
        df,
        nfreq,
        mp.Near2FarRegion(center=mp.Vector3(+d, 0, 0), size=mp.Vector3(0, l, l), direction=mp.X, weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(-d, 0, 0), size=mp.Vector3(0, l, l), direction=mp.X, weight=-1),
        mp.Near2FarRegion(center=mp.Vector3(0, +d, 0), size=mp.Vector3(l, 0, l), direction=mp.Y, weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(0, -d, 0), size=mp.Vector3(l, 0, l), direction=mp.Y, weight=-1),
        mp.Near2FarRegion(center=mp.Vector3(0, 0, +d), size=mp.Vector3(l, l, 0), direction=mp.Z, weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(0, 0, -d), size=mp.Vector3(l, l, 0), direction=mp.Z, weight=-1),
    )

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            60,
            mp.Ez,
            mp.Vector3(d, 0, 0),
            1e-6,
        ),
    )

    # For a z-polarized point source, intensity ideally follows a sin^2(theta) pattern.
    # The sweep is performed in the x-z plane (y=0) with theta measured from the +z direction.
    r = 1000.0
    thetas = np.linspace(0.0, np.pi, 181)  # theta=0 is +z
    intensities: list[float] = []
    for theta in thetas:
        x = r * np.sin(theta)
        z = r * np.cos(theta)
        ff = sim.get_farfield(n2f, mp.Vector3(x, 0, z))
        ex, ey, ez, *_ = ff
        e2 = float(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)
        intensities.append(e2)

    u = np.array(intensities)
    u = u / (np.max(u) + 1e-30)
    u_ideal = np.sin(thetas) ** 2

    if not mp.am_master():
        return

    fig, ax = plt.subplots(figsize=(6.6, 3.2), constrained_layout=True)
    ax.plot(np.degrees(thetas), u, label="N2F (normalized)")
    ax.plot(np.degrees(thetas), u_ideal, linestyle="--", color="black", alpha=0.6, label=r"$\sin^2\theta$")
    ax.set_xlabel(r"$\theta$ [deg] (from +z)")
    ax.set_ylabel("normalized intensity")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    repo_dir = Path(__file__).resolve().parents[2]
    figs_dir = repo_dir / "figs" / "ch06"
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(figs_dir / "ch06_n2f_dipole_pattern.png"), dpi=200)
    plt.close(fig)

    print("done")


if __name__ == "__main__":
    main()
