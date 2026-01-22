from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp
import numpy as np
from matplotlib import font_manager

# Unit convention: 1 [meep length] = 1 mm (Chapter 5)
c0 = 299_792_458.0  # [m/s]
L0 = 1e-3  # [m]
Z0_OHM = 376.730313668  # vacuum wave impedance [ohm]


def ghz_to_meep(f_ghz: float) -> float:
    return f_ghz * 1e9 * (L0 / c0)


def meep_to_ghz(f_meep: np.ndarray) -> np.ndarray:
    return np.asarray(f_meep) * (c0 / L0) / 1e9


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def configure_plot_fonts() -> None:
    # Prefer system fonts that render reliably on common environments.
    candidates = [
        "Hiragino Sans",
        "Hiragino Kaku Gothic ProN",
        "Yu Gothic",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
        "IPAGothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class DipoleParams:
    f0_ghz: float = 2.45
    df_ghz: float = 1.20
    nfreq: int = 61

    length: float = 58.0  # [mm] (initial guess including end effects)
    radius: float = 1.0  # [mm] (thicker conductors typically shift resonance slightly lower)
    gap: float = 2.0  # [mm]

    dpml: float = 15.0  # [mm]
    d_buf: float = 10.0  # [mm] (distance between PML and N2F surface)
    d_n2f_xy: float = 35.0  # [mm] (distance from center to N2F surface)
    d_n2f_z: float = 45.0  # [mm]

    resolution: int = 1  # [px/mm] (start coarse for 3D)

    src_size_xy: float = 2.0  # [mm] (effective feed size)
    src_amp: complex = 1.0  # Jz amplitude (phase can be set via complex values)
    zref_ohm: float = 50.0  # reference impedance


def make_dipole_geometry(*, length: float, radius: float, gap: float) -> list[mp.GeometricObject]:
    arm_len = 0.5 * (length - gap)
    zc = 0.5 * gap + 0.5 * arm_len
    return [
        mp.Cylinder(
            radius=radius,
            height=arm_len,
            center=mp.Vector3(0, 0, +zc),
            axis=mp.Vector3(0, 0, 1),
            material=mp.metal,
        ),
        mp.Cylinder(
            radius=radius,
            height=arm_len,
            center=mp.Vector3(0, 0, -zc),
            axis=mp.Vector3(0, 0, 1),
            material=mp.metal,
        ),
    ]


def spherical_basis(theta: float, phi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    e_r = np.array([st * cp, st * sp, ct])
    e_theta = np.array([ct * cp, ct * sp, -st])
    e_phi = np.array([-sp, cp, 0.0])
    return e_r, e_theta, e_phi


def make_setup_figure(*, p: DipoleParams, sx: float, sz: float, out_setup: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.set_aspect("equal", adjustable="box")

    # cell boundary
    ax.add_patch(plt.Rectangle((-0.5 * sx, -0.5 * sz), sx, sz, fill=False, lw=1.5, color="k"))
    # PML
    ax.add_patch(
        plt.Rectangle(
            (-0.5 * sx, -0.5 * sz),
            sx,
            sz,
            fill=False,
            lw=0,
            edgecolor="none",
            hatch="///",
            alpha=0.08,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (-0.5 * sx + p.dpml, -0.5 * sz + p.dpml),
            sx - 2 * p.dpml,
            sz - 2 * p.dpml,
            fill=False,
            lw=0,
            edgecolor="none",
            hatch="///",
            alpha=0.08,
        )
    )

    # near2far box
    ax.add_patch(
        plt.Rectangle(
            (-p.d_n2f_xy, -p.d_n2f_z),
            2 * p.d_n2f_xy,
            2 * p.d_n2f_z,
            fill=False,
            lw=2,
            ls="--",
            color="tab:orange",
        )
    )
    ax.annotate(
        "N2F surface",
        xy=(-p.d_n2f_xy, p.d_n2f_z),
        xytext=(-p.d_n2f_xy + 10, p.d_n2f_z + 10),
        color="tab:orange",
        fontsize=11,
        arrowprops=dict(arrowstyle="-", color="tab:orange", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
    )

    # dipole (z-axis)
    ax.plot([0, 0], [+0.5 * p.gap, 0.5 * p.length], color="tab:purple", lw=4)
    ax.plot([0, 0], [-0.5 * p.length, -0.5 * p.gap], color="tab:purple", lw=4)
    ax.annotate(
        "Conductor",
        xy=(0, 0.5 * p.length - 3),
        xytext=(10, 0.5 * p.length - 8),
        color="tab:purple",
        fontsize=11,
        arrowprops=dict(arrowstyle="-", color="tab:purple", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
    )

    # source (gap)
    ax.plot([0], [0], marker="o", color="red")
    ax.annotate(
        "Source (feed)",
        xy=(0, 0),
        xytext=(10, -12),
        color="red",
        fontsize=11,
        arrowprops=dict(arrowstyle="-", color="red", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
    )

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("Dipole setup (x-z plane)")
    ax.set_xlim(-0.5 * sx, 0.5 * sx)
    ax.set_ylim(-0.5 * sz, 0.5 * sz)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_setup, dpi=200)
    plt.close(fig)


def main() -> None:
    configure_plot_fonts()
    p = DipoleParams()

    repo_dir = Path(__file__).resolve().parents[2]
    figs_dir = repo_dir / "figs" / "ch08"
    ensure_dir(str(figs_dir))

    out_s11 = str(figs_dir / "ch08_dipole_s11.png")
    out_pattern = str(figs_dir / "ch08_dipole_pattern.png")
    out_ez_line = str(figs_dir / "ch08_dipole_ez_along_arm.png")
    out_ez_xz = str(figs_dir / "ch08_dipole_ez_xz.png")
    out_setup = str(figs_dir / "ch08_dipole_setup.png")

    out_paths = {
        "s11": out_s11,
        "pattern": out_pattern,
        "ez_line": out_ez_line,
        "ez_xz": out_ez_xz,
        "setup": out_setup,
    }
    force_regen = os.environ.get("CH08_FORCE_REGEN") == "1"
    missing = {k: (force_regen or (not os.path.exists(p))) for k, p in out_paths.items()}
    if not any(missing.values()):
        return

    f0 = ghz_to_meep(p.f0_ghz)
    fcen = ghz_to_meep(p.f0_ghz)
    df = ghz_to_meep(p.df_ghz)
    freqs = np.linspace(fcen - 0.5 * df, fcen + 0.5 * df, p.nfreq)

    # Set the cell from “N2F surface + buffer + PML” as a working baseline.
    sx = 2 * (p.d_n2f_xy + p.d_buf + p.dpml)
    sy = sx
    sz = 2 * (p.d_n2f_z + p.d_buf + p.dpml)
    cell = mp.Vector3(sx, sy, sz)

    need_run = any(missing[k] for k in ["s11", "pattern", "ez_line", "ez_xz"])
    if not need_run and missing["setup"] and mp.am_master():
        make_setup_figure(p=p, sx=sx, sz=sz, out_setup=out_setup)
        return

    geometry = make_dipole_geometry(length=p.length, radius=p.radius, gap=p.gap)
    src = mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(),
        size=mp.Vector3(p.src_size_xy, p.src_size_xy, p.gap),
        amplitude=p.src_amp,
    )

    sim = mp.Simulation(
        resolution=p.resolution,
        cell_size=cell,
        boundary_layers=[mp.PML(p.dpml)],
        geometry=geometry,
        sources=[src],
    )

    # Sample complex Ez at the feed point (per frequency). Approximate voltage as V ≈ Ez * gap.
    dft_feed = sim.add_dft_fields([mp.Ez], fcen, df, p.nfreq, center=mp.Vector3(), size=mp.Vector3())

    # Sample Ez along z slightly outside the conductor surface to view a proxy of current distribution.
    probe_x = p.radius + 1.0
    dft_line = sim.add_dft_fields(
        [mp.Ez],
        fcen,
        df,
        p.nfreq,
        center=mp.Vector3(probe_x, 0, 0),
        size=mp.Vector3(0, 0, p.length + 20.0),
    )

    # Sample complex Ez on an x-z cross section (single frequency near resonance).
    dft_xz = sim.add_dft_fields(
        [mp.Ez],
        f0,
        0.0,
        1,
        center=mp.Vector3(),
        size=mp.Vector3(sx - 2 * p.dpml, 0, sz - 2 * p.dpml),
    )

    # N2F: do not place regions inside PML. Use 6 faces to form a closed surface.
    n2f = sim.add_near2far(
        fcen,
        df,
        p.nfreq,
        mp.Near2FarRegion(
            center=mp.Vector3(+p.d_n2f_xy, 0, 0),
            size=mp.Vector3(0, 2 * p.d_n2f_xy, 2 * p.d_n2f_z),
            direction=mp.X,
            weight=+1,
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(-p.d_n2f_xy, 0, 0),
            size=mp.Vector3(0, 2 * p.d_n2f_xy, 2 * p.d_n2f_z),
            direction=mp.X,
            weight=-1,
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(0, +p.d_n2f_xy, 0),
            size=mp.Vector3(2 * p.d_n2f_xy, 0, 2 * p.d_n2f_z),
            direction=mp.Y,
            weight=+1,
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(0, -p.d_n2f_xy, 0),
            size=mp.Vector3(2 * p.d_n2f_xy, 0, 2 * p.d_n2f_z),
            direction=mp.Y,
            weight=-1,
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(0, 0, +p.d_n2f_z),
            size=mp.Vector3(2 * p.d_n2f_xy, 2 * p.d_n2f_xy, 0),
            direction=mp.Z,
            weight=+1,
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(0, 0, -p.d_n2f_z),
            size=mp.Vector3(2 * p.d_n2f_xy, 2 * p.d_n2f_xy, 0),
            direction=mp.Z,
            weight=-1,
        ),
    )

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            600,
            mp.Ez,
            mp.Vector3(p.d_n2f_xy, 0, 0),
            1e-5,
        )
    )

    # ---- S11（Z=V/I → Γ）----
    ez_feed = np.array([sim.get_dft_array(dft_feed, mp.Ez, i).item() for i in range(p.nfreq)])
    v = ez_feed * p.gap
    i0 = p.src_amp * (p.src_size_xy * p.src_size_xy)
    z_in = v / i0

    zref = p.zref_ohm / Z0_OHM
    gamma = (z_in - zref) / (z_in + zref)
    s11 = gamma

    if mp.am_master():
        f_ghz = meep_to_ghz(freqs)
        eps = 1e-12
        s11_db = 20 * np.log10(np.maximum(np.abs(s11), eps))

        fig, ax = plt.subplots(figsize=(7.0, 3.3))
        ax.plot(f_ghz, s11_db, lw=2)
        ax.axhline(-10, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("f [GHz]")
        ax.set_ylabel(r"$|S_{11}|$ [dB]")
        ax.set_ylim(-40, 0)
        ax.grid(True, alpha=0.3)
        ax.set_title("Half-wave dipole (gap-fed) : reflection")
        fig.tight_layout()
        fig.savefig(out_s11, dpi=200)
        plt.close(fig)

    # ---- Proxy of current distribution (Ez profile along z) ----
    idx0 = int(np.argmin(np.abs(freqs - f0)))
    _, _, z_line, _ = sim.get_array_metadata(dft_cell=dft_line)
    ez_line = sim.get_dft_array(dft_line, mp.Ez, idx0)
    ez_line_1d = np.squeeze(ez_line)

    if mp.am_master():
        fig, ax = plt.subplots(figsize=(6.6, 3.2))
        ax.plot(z_line, np.abs(ez_line_1d), lw=2)
        ax.axvline(-0.5 * p.gap, color="k", lw=1, alpha=0.3)
        ax.axvline(+0.5 * p.gap, color="k", lw=1, alpha=0.3)
        ax.set_xlabel("z [mm]")
        ax.set_ylabel(r"$|E_z|$ [a.u.]")
        ax.grid(True, alpha=0.3)
        ax.set_title(r"$|E_z|$ along the arm (near conductor surface)")
        fig.tight_layout()
        fig.savefig(out_ez_line, dpi=200)
        plt.close(fig)

    # ---- Ez on x-z cross section (near resonance) ----
    x_xz, _, z_xz, _ = sim.get_array_metadata(dft_cell=dft_xz)
    ez_xz = sim.get_dft_array(dft_xz, mp.Ez, 0)
    ez_xz_2d = np.squeeze(ez_xz).T  # to (z, x)

    if mp.am_master():
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        im = ax.pcolormesh(x_xz, z_xz, np.real(ez_xz_2d), shading="auto", cmap="RdBu_r")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_title(r"$E_z(x,z)$ (real part, near resonance)")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(im, ax=ax, label=r"$E_z$ [a.u.]")
        fig.tight_layout()
        fig.savefig(out_ez_xz, dpi=200)
        plt.close(fig)

    # ---- Radiation pattern (x-z cut) ----
    # theta: angle from +z. phi=0 corresponds to the x-z plane.
    ntheta = 181
    thetas = np.linspace(0.0, np.pi, ntheta)
    r_far = 2000.0
    e2 = np.zeros_like(thetas)
    for k, th in enumerate(thetas):
        x = r_far * np.sin(th)
        z = r_far * np.cos(th)
        ff = sim.get_farfield(n2f, mp.Vector3(x, 0, z))
        ex, ey, ez, _, _, _ = ff[6 * idx0 : 6 * idx0 + 6]
        e2[k] = (abs(ex) ** 2 + abs(ey) ** 2 + abs(ez) ** 2)
    e2 /= np.max(e2) + 1e-30

    # Ideal half-wave dipole theta pattern (for comparison)
    with np.errstate(divide="ignore", invalid="ignore"):
        ideal = np.cos(0.5 * np.pi * np.cos(thetas)) / np.sin(thetas)
    ideal = np.where(np.isfinite(ideal), ideal, 0.0) ** 2
    ideal /= np.max(ideal) + 1e-30

    if mp.am_master():
        fig = plt.figure(figsize=(6.3, 4.0))
        ax = fig.add_subplot(111, projection="polar")
        ax.plot(thetas, e2, label="MEEP", lw=2)
        ax.plot(thetas, ideal, label="ideal (half-wave)", lw=1.5, ls="--")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rmax(1.0)
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.set_title("Radiation pattern (x-z cut, normalized)")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.15), ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(out_pattern, dpi=200)
        plt.close(fig)

    # ---- Setup figure (schematic) ----
    if mp.am_master():
        make_setup_figure(p=p, sx=sx, sz=sz, out_setup=out_setup)


if __name__ == "__main__":
    main()
