from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp
import numpy as np

# Unit convention: 1 [meep length] = 1 mm (Chapter 5)
c0 = 299_792_458.0  # [m/s]
L0 = 1e-3  # [m]


def ghz_to_meep(f_ghz: float) -> float:
    return f_ghz * 1e9 * (L0 / c0)


def meep_to_ghz(f_meep: np.ndarray) -> np.ndarray:
    return np.asarray(f_meep) * (c0 / L0) / 1e9


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass(frozen=True)
class CrossDipoleParams:
    f0_ghz: float = 2.45
    df_ghz: float = 0.60
    nfreq: int = 41

    length: float = 58.0  # [mm]
    radius: float = 1.0  # [mm]
    gap: float = 2.0  # [mm]
    z_offset: float = 1.5  # [mm] (avoid overlap at the cross)

    dpml: float = 12.0  # [mm]
    d_buf: float = 8.0  # [mm]
    d_n2f: float = 35.0  # [mm]

    resolution: int = 1

    src_size_xy: float = 2.0  # [mm]
    src_amp: float = 1.0
    phase_deg: float = 90.0  # phase difference between orthogonal excitations


def make_cross_dipole_geometry(*, length: float, radius: float, gap: float, z_offset: float) -> list[mp.GeometricObject]:
    arm_len = 0.5 * (length - gap)
    xc = 0.5 * gap + 0.5 * arm_len
    yc = xc
    return [
        # x-oriented dipole (shift upward slightly)
        mp.Cylinder(
            radius=radius,
            height=arm_len,
            center=mp.Vector3(+xc, 0, +z_offset),
            axis=mp.Vector3(1, 0, 0),
            material=mp.metal,
        ),
        mp.Cylinder(
            radius=radius,
            height=arm_len,
            center=mp.Vector3(-xc, 0, +z_offset),
            axis=mp.Vector3(1, 0, 0),
            material=mp.metal,
        ),
        # y-oriented dipole (shift downward slightly)
        mp.Cylinder(
            radius=radius,
            height=arm_len,
            center=mp.Vector3(0, +yc, -z_offset),
            axis=mp.Vector3(0, 1, 0),
            material=mp.metal,
        ),
        mp.Cylinder(
            radius=radius,
            height=arm_len,
            center=mp.Vector3(0, -yc, -z_offset),
            axis=mp.Vector3(0, 1, 0),
            material=mp.metal,
        ),
    ]


def rhcp_lhcp_from_exey(ex: np.ndarray, ey: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Under the Chapter 3 convention (E_RHCP=(Eθ-jEφ)/√2), boresight (+z) can be approximated as Eθ=Ex and Eφ=Ey for phi=0.
    er = (ex - 1j * ey) / np.sqrt(2.0)
    el = (ex + 1j * ey) / np.sqrt(2.0)
    return er, el


def axial_ratio_db(er: np.ndarray, el: np.ndarray) -> np.ndarray:
    # With co=RHCP and cross=LHCP: AR=(1+ρ)/(1-ρ), ρ=|cross|/|co|
    eps = 1e-30
    rho = np.minimum(np.abs(el) / np.maximum(np.abs(er), eps), 0.999999)
    ar = (1.0 + rho) / (1.0 - rho)
    return 20 * np.log10(np.maximum(ar, 1.0))


def main() -> None:
    p = CrossDipoleParams()

    repo_dir = Path(__file__).resolve().parents[2]
    figs_dir = repo_dir / "figs" / "ch08"
    ensure_dir(str(figs_dir))

    out_ar = str(figs_dir / "ch08_cross_dipole_ar.png")
    out_circ = str(figs_dir / "ch08_cross_dipole_rhcp_lhcp.png")
    out_setup = str(figs_dir / "ch08_cross_dipole_setup.png")

    outputs = [out_ar, out_circ, out_setup]
    if os.environ.get("CH08_FORCE_REGEN") != "1" and all(os.path.exists(p) for p in outputs):
        return

    f0 = ghz_to_meep(p.f0_ghz)
    fcen = ghz_to_meep(p.f0_ghz)
    df = ghz_to_meep(p.df_ghz)
    freqs = np.linspace(fcen - 0.5 * df, fcen + 0.5 * df, p.nfreq)

    # Cell: N2F surface + buffer + PML
    s = 2 * (p.d_n2f + p.d_buf + p.dpml)
    cell = mp.Vector3(s, s, s)

    geometry = make_cross_dipole_geometry(length=p.length, radius=p.radius, gap=p.gap, z_offset=p.z_offset)

    # Two orthogonal sources: x is phase 0, y is +90° (handedness depends on convention)
    phase = np.deg2rad(p.phase_deg)
    src_x = mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),
        component=mp.Ex,
        center=mp.Vector3(0, 0, +p.z_offset),
        size=mp.Vector3(p.gap, p.src_size_xy, p.src_size_xy),
        amplitude=p.src_amp,
    )
    src_y = mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),
        component=mp.Ey,
        center=mp.Vector3(0, 0, -p.z_offset),
        size=mp.Vector3(p.src_size_xy, p.gap, p.src_size_xy),
        amplitude=p.src_amp * np.exp(1j * phase),
    )

    sim = mp.Simulation(
        resolution=p.resolution,
        cell_size=cell,
        boundary_layers=[mp.PML(p.dpml)],
        geometry=geometry,
        sources=[src_x, src_y],
    )

    n2f = sim.add_near2far(
        fcen,
        df,
        p.nfreq,
        mp.Near2FarRegion(center=mp.Vector3(+p.d_n2f, 0, 0), size=mp.Vector3(0, 2 * p.d_n2f, 2 * p.d_n2f), direction=mp.X, weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(-p.d_n2f, 0, 0), size=mp.Vector3(0, 2 * p.d_n2f, 2 * p.d_n2f), direction=mp.X, weight=-1),
        mp.Near2FarRegion(center=mp.Vector3(0, +p.d_n2f, 0), size=mp.Vector3(2 * p.d_n2f, 0, 2 * p.d_n2f), direction=mp.Y, weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(0, -p.d_n2f, 0), size=mp.Vector3(2 * p.d_n2f, 0, 2 * p.d_n2f), direction=mp.Y, weight=-1),
        mp.Near2FarRegion(center=mp.Vector3(0, 0, +p.d_n2f), size=mp.Vector3(2 * p.d_n2f, 2 * p.d_n2f, 0), direction=mp.Z, weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(0, 0, -p.d_n2f), size=mp.Vector3(2 * p.d_n2f, 2 * p.d_n2f, 0), direction=mp.Z, weight=-1),
    )

    # Circular polarization should ideally be evaluated in a well-converged steady state, but 3D runs are expensive.
    # As a baseline, run for a fixed time after the sources stop so that figures can be produced.
    # For tighter convergence, increase this value or use stop_when_fields_decayed.
    t_after = float(os.environ.get("CH08_CROSS_T_AFTER", "2200"))
    sim.run(until_after_sources=t_after)

    # Extract Ex and Ey at boresight (+z)
    r_far = 1500.0
    ff = sim.get_farfield(n2f, mp.Vector3(0, 0, r_far))

    ex = np.array([ff[6 * i + 0] for i in range(p.nfreq)], dtype=np.complex128)
    ey = np.array([ff[6 * i + 1] for i in range(p.nfreq)], dtype=np.complex128)

    er, el = rhcp_lhcp_from_exey(ex, ey)
    ar_db = axial_ratio_db(er, el)

    if mp.am_master():
        f_ghz = meep_to_ghz(freqs)
        fig, ax = plt.subplots(figsize=(7.0, 3.3))
        ax.plot(f_ghz, ar_db, lw=2)
        ax.axhline(3.0, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("f [GHz]")
        ax.set_ylabel("AR [dB]")
        ax.set_ylim(0, 15)
        ax.grid(True, alpha=0.3)
        ax.set_title("Cross-dipole circular polarization (boresight)")
        fig.tight_layout()
        fig.savefig(out_ar, dpi=200)
        plt.close(fig)

        eps = 1e-30
        rhcp_db = 20 * np.log10(np.maximum(np.abs(er), eps))
        lhcp_db = 20 * np.log10(np.maximum(np.abs(el), eps))

        fig, ax = plt.subplots(figsize=(7.0, 3.3))
        ax.plot(f_ghz, rhcp_db, lw=2, label="RHCP")
        ax.plot(f_ghz, lhcp_db, lw=2, ls="--", label="LHCP")
        ax.set_xlabel("f [GHz]")
        ax.set_ylabel(r"$|E|$ [dB a.u.]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.set_title("RHCP/LHCP split at boresight")
        fig.tight_layout()
        fig.savefig(out_circ, dpi=200)
        plt.close(fig)

        # Setup figure (top-view schematic)
        fig, ax = plt.subplots(figsize=(5.6, 5.6))
        ax.set_aspect("equal", adjustable="box")
        ax.add_patch(plt.Circle((0, 0), 0.5 * s, fill=False, lw=1.0, color="k", alpha=0.5))
        ax.add_patch(plt.Rectangle((-p.d_n2f, -p.d_n2f), 2 * p.d_n2f, 2 * p.d_n2f, fill=False, lw=2, ls="--", color="tab:orange"))
        ax.text(-p.d_n2f, p.d_n2f + 3, "N2F box", color="tab:orange")
        ax.plot([+0.5 * p.gap, 0.5 * p.length], [0, 0], color="tab:purple", lw=4)
        ax.plot([-0.5 * p.length, -0.5 * p.gap], [0, 0], color="tab:purple", lw=4)
        ax.plot([0, 0], [+0.5 * p.gap, 0.5 * p.length], color="tab:blue", lw=4)
        ax.plot([0, 0], [-0.5 * p.length, -0.5 * p.gap], color="tab:blue", lw=4)
        ax.plot([0], [0], marker="o", color="red")
        ax.text(4, 4, "feed", color="red")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title("Cross-dipole setup (top view)")
        ax.set_xlim(-0.5 * s, 0.5 * s)
        ax.set_ylim(-0.5 * s, 0.5 * s)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_setup, dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
