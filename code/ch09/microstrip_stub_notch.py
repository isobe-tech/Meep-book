from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp
import numpy as np
from matplotlib.patches import Rectangle

# Unit convention: 1 [meep length] = 1 mm (Chapter 5)
c0 = 299_792_458.0  # [m/s]
L0 = 1e-3  # [m]


def ghz_to_meep(f_ghz: float) -> float:
    return f_ghz * 1e9 * (L0 / c0)


def meep_to_ghz(f_meep: np.ndarray) -> np.ndarray:
    return np.asarray(f_meep) * (c0 / L0) / 1e9


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def microstrip_eps_eff(eps_r: float, w: float, h: float) -> float:
    wh = w / h
    if wh <= 0:
        return 1.0
    # Representative approximation (simplified Hammerstad–Jensen)
    return 0.5 * (eps_r + 1.0) + 0.5 * (eps_r - 1.0) * (1.0 / np.sqrt(1.0 + 12.0 / wh))


@dataclass(frozen=True)
class StubParams:
    # Frequency
    f0_ghz: float = 2.45
    df_ghz: float = 2.0
    nfreq: int = 51

    # Microstrip (values comparable to Chapter 7)
    eps_r: float = 4.2
    h: float = 1.6  # [mm]
    t_metal: float = 0.25  # [mm]
    w0: float = 3.0  # [mm]

    # Open-circuited stub (extend toward +y)
    l_stub: float = 18.0  # [mm] (initial λg/4 guess)
    x_stub: float = 0.0  # [mm]

    # Simulation domain
    dpml: float = 2.0  # [mm]
    sx: float = 40.0  # [mm]
    sy: float = 48.0  # [mm]
    sz: float = 12.0  # [mm]
    resolution: int = 4  # [px/mm]

    # Reference planes (keep distance from PML and the discontinuity)
    ref_margin: float = 6.0  # [mm]

    # Stopping condition (resonance tails exist; start from a not-too-strict value)
    decay_target: float = 1e-3
    z_probe: float = 0.6  # [mm] (above the conductor)


def make_microstrip_geometry(*, eps_r: float, h: float, t_metal: float, w0: float) -> list[mp.GeometricObject]:
    # Let z=0 be the substrate top surface, and let +x be the propagation direction.
    substrate = mp.Block(
        material=mp.Medium(epsilon=eps_r),
        center=mp.Vector3(0, 0, -0.5 * h),
        size=mp.Vector3(mp.inf, mp.inf, h),
    )
    gnd = mp.Block(
        material=mp.metal,
        center=mp.Vector3(0, 0, -h - 0.5 * t_metal),
        size=mp.Vector3(mp.inf, mp.inf, t_metal),
    )

    z_sig = 0.5 * t_metal
    x_long = 1e3  # [mm] (this value only needs to be much larger than the cell)
    sig = mp.Block(
        material=mp.metal,
        center=mp.Vector3(0, 0, z_sig),
        size=mp.Vector3(x_long, w0, t_metal),
    )
    return [substrate, gnd, sig]


def make_microstrip_stub_geometry(
    *,
    eps_r: float,
    h: float,
    t_metal: float,
    w0: float,
    l_stub: float,
    x_stub: float,
) -> list[mp.GeometricObject]:
    geom = make_microstrip_geometry(eps_r=eps_r, h=h, t_metal=t_metal, w0=w0)

    # Extend the stub from the main line end toward +y. Use a slight overlap to ensure connectivity.
    overlap = 0.5  # [mm]
    y_center = 0.5 * w0 + 0.5 * (l_stub - overlap)
    z_sig = 0.5 * t_metal
    stub = mp.Block(
        material=mp.metal,
        center=mp.Vector3(x_stub, y_center, z_sig),
        size=mp.Vector3(w0, l_stub + overlap, t_metal),
    )
    geom.append(stub)
    return geom


def add_flux_monitor(
    sim: mp.Simulation,
    *,
    fcen: float,
    df: float,
    nfreq: int,
    x_pos: float,
    sy: float,
    sz: float,
    dpml: float,
) -> mp.DftFlux:
    y_span = sy - 2 * dpml
    z_span = sz - 2 * dpml
    return sim.add_flux(
        fcen,
        df,
        nfreq,
        mp.FluxRegion(center=mp.Vector3(x_pos, 0, 0), size=mp.Vector3(0, y_span, z_span), direction=mp.X),
    )


def run_until_decayed(sim: mp.Simulation, *, x_probe: float, z_probe: float, decay_target: float) -> None:
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            80,
            mp.Ez,
            mp.Vector3(x_probe, 0, z_probe),
            decay_target,
        )
    )


def run_baseline(
    *,
    p: StubParams,
    fcen: float,
    df: float,
    df_src: float,
    x_src: float,
    x_in: float,
    x_out: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, mp.FluxData]:
    geom = make_microstrip_geometry(eps_r=p.eps_r, h=p.h, t_metal=p.t_metal, w0=p.w0)
    src = mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=df_src),
        component=mp.Ez,
        center=mp.Vector3(x_src, 0, -0.5 * p.h),
        size=mp.Vector3(0, p.w0, 0),
    )
    sim = mp.Simulation(
        resolution=p.resolution,
        cell_size=mp.Vector3(p.sx, p.sy, p.sz),
        boundary_layers=[mp.PML(p.dpml)],
        geometry=geom,
        sources=[src],
    )
    flux_in = add_flux_monitor(sim, fcen=fcen, df=df, nfreq=p.nfreq, x_pos=x_in, sy=p.sy, sz=p.sz, dpml=p.dpml)
    flux_out = add_flux_monitor(sim, fcen=fcen, df=df, nfreq=p.nfreq, x_pos=x_out, sy=p.sy, sz=p.sz, dpml=p.dpml)
    run_until_decayed(sim, x_probe=x_out, z_probe=p.z_probe, decay_target=p.decay_target)

    freqs = np.array(mp.get_flux_freqs(flux_in))
    p_inc = np.array(mp.get_fluxes(flux_in))
    p_out = np.array(mp.get_fluxes(flux_out))
    inc_flux_data = sim.get_flux_data(flux_in)
    sim.reset_meep()
    return freqs, p_inc, p_out, inc_flux_data


def run_with_stub(
    *,
    p: StubParams,
    fcen: float,
    df: float,
    df_src: float,
    x_src: float,
    x_in: float,
    x_out: float,
    inc_flux_data_in: mp.FluxData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geom = make_microstrip_stub_geometry(
        eps_r=p.eps_r,
        h=p.h,
        t_metal=p.t_metal,
        w0=p.w0,
        l_stub=p.l_stub,
        x_stub=p.x_stub,
    )
    src = mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=df_src),
        component=mp.Ez,
        center=mp.Vector3(x_src, 0, -0.5 * p.h),
        size=mp.Vector3(0, p.w0, 0),
    )
    sim = mp.Simulation(
        resolution=p.resolution,
        cell_size=mp.Vector3(p.sx, p.sy, p.sz),
        boundary_layers=[mp.PML(p.dpml)],
        geometry=geom,
        sources=[src],
    )
    flux_in = add_flux_monitor(sim, fcen=fcen, df=df, nfreq=p.nfreq, x_pos=x_in, sy=p.sy, sz=p.sz, dpml=p.dpml)
    flux_out = add_flux_monitor(sim, fcen=fcen, df=df, nfreq=p.nfreq, x_pos=x_out, sy=p.sy, sz=p.sz, dpml=p.dpml)
    sim.load_minus_flux_data(flux_in, inc_flux_data_in)

    # 2D view of fields near the notch frequency (plane at z=z_probe)
    x_span = p.sx - 2 * p.dpml
    y_span = p.sy - 2 * p.dpml
    dft_xy = sim.add_dft_fields(
        [mp.Ez],
        fcen,
        0.0,
        1,
        center=mp.Vector3(0, 0, p.z_probe),
        size=mp.Vector3(x_span, y_span, 0),
    )

    run_until_decayed(sim, x_probe=x_out, z_probe=p.z_probe, decay_target=p.decay_target)

    refl_flux = np.array(mp.get_fluxes(flux_in))
    tran_flux = np.array(mp.get_fluxes(flux_out))
    p_ref = -refl_flux
    p_tran = tran_flux

    ez_xy = sim.get_dft_array(dft_xy, mp.Ez, 0)
    sim.reset_meep()
    return p_ref, p_tran, ez_xy


def main() -> None:
    p = StubParams()

    repo_dir = Path(__file__).resolve().parents[2]
    figs_dir = repo_dir / "figs" / "ch09"
    ensure_dir(str(figs_dir))

    out_sparams = str(figs_dir / "ch09_stub_sparams.png")
    out_ez_xy = str(figs_dir / "ch09_stub_ez_xy.png")
    outputs = [out_sparams, out_ez_xy]
    if os.environ.get("CH09_FORCE_REGEN") != "1" and all(os.path.exists(p) for p in outputs):
        return

    fcen = ghz_to_meep(p.f0_ghz)
    df = ghz_to_meep(p.df_ghz)
    df_src = 5.0 * df

    # Reference-plane and source locations
    x_src = -0.5 * p.sx + p.dpml + 0.8
    x_in = x_src + p.ref_margin
    x_out = 0.5 * p.sx - p.dpml - p.ref_margin

    eps_eff = microstrip_eps_eff(p.eps_r, p.w0, p.h)
    f_notch_est = (c0 / L0) / (4.0 * p.l_stub * np.sqrt(eps_eff)) / 1e9
    if mp.am_master():
        print(f"eps_eff≈{eps_eff:.3g}, f_notch_est≈{f_notch_est:.3g} GHz")

    freqs, p_inc, p_out_base, inc_flux_data_in = run_baseline(
        p=p,
        fcen=fcen,
        df=df,
        df_src=df_src,
        x_src=x_src,
        x_in=x_in,
        x_out=x_out,
    )
    p_ref, p_tran, ez_xy = run_with_stub(
        p=p,
        fcen=fcen,
        df=df,
        df_src=df_src,
        x_src=x_src,
        x_in=x_in,
        x_out=x_out,
        inc_flux_data_in=inc_flux_data_in,
    )

    if not mp.am_master():
        return

    f_ghz = meep_to_ghz(freqs)
    eps = 1e-30

    s11_db = 10 * np.log10(np.maximum(p_ref / np.maximum(p_inc, eps), eps))
    s21_db = 10 * np.log10(np.maximum(p_tran / np.maximum(p_inc, eps), eps))
    s21_base_db = 10 * np.log10(np.maximum(p_out_base / np.maximum(p_inc, eps), eps))

    fig, axes = plt.subplots(2, 1, figsize=(6.9, 5.4), sharex=True)
    ax = axes[0]
    ax.plot(f_ghz, s11_db, lw=1.9, label="with open stub")
    ax.set_ylabel(r"$|S_{11}|$ [dB]")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-60, 0)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ax.plot(f_ghz, s21_base_db, lw=1.6, label="baseline (no stub)")
    ax.plot(f_ghz, s21_db, lw=1.9, ls="--", label="with open stub")
    ax.set_xlabel("f [GHz]")
    ax.set_ylabel(r"$|S_{21}|$ [dB]")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-40, 1)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_sparams, dpi=200)
    plt.close(fig)

    # Top view of DFT |Ez| (z=z_probe)
    x_span = p.sx - 2 * p.dpml
    y_span = p.sy - 2 * p.dpml
    x = np.linspace(-0.5 * x_span, 0.5 * x_span, ez_xy.shape[0])
    y = np.linspace(-0.5 * y_span, 0.5 * y_span, ez_xy.shape[1])

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    im = ax.imshow(
        np.abs(ez_xy).T,
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect="auto",
        cmap="magma",
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(rf"$|E_z(x,y)|$ at $f={p.f0_ghz:.2f}\,\mathrm{{GHz}}$")
    ax.set_ylim(-20, 20)
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label(r"$|E_z|$ [a.u.]")

    # Main line and stub locations (for plotting)
    ax.add_patch(
        Rectangle(
            (x[0], -0.5 * p.w0),
            x[-1] - x[0],
            p.w0,
            fill=False,
            lw=1.0,
            ec="white",
            alpha=0.8,
        )
    )
    ax.add_patch(
        Rectangle(
            (-0.5 * p.w0, 0.5 * p.w0),
            p.w0,
            p.l_stub,
            fill=False,
            lw=1.2,
            ec="white",
            alpha=0.9,
        )
    )
    ax.text(
        2.5,
        14.0,
        "open stub",
        color="white",
        ha="left",
        va="bottom",
        fontsize=9,
        clip_on=False,
        bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 1.5},
    )
    fig.tight_layout()
    fig.savefig(out_ez_xy, dpi=200)
    plt.close(fig)

    print("generated:", out_sparams)
    print("generated:", out_ez_xy)


if __name__ == "__main__":
    main()
