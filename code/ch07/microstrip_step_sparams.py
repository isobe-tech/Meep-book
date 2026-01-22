from __future__ import annotations

import os
from pathlib import Path

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


def make_microstrip_geometry(
    *,
    eps_r: float,
    h: float,
    t_metal: float,
    w0: float,
    w_step: float,
    l_step: float,
) -> list:
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

    # Signal trace: width is w0 on both sides and w_step only in the center (step discontinuity).
    x_long = 1e3  # [meep] (= mm). This value only needs to be much larger than the cell.
    sig_left = mp.Block(
        material=mp.metal,
        center=mp.Vector3(-0.5 * (l_step + x_long), 0, z_sig),
        size=mp.Vector3(x_long, w0, t_metal),
    )
    sig_mid = mp.Block(
        material=mp.metal,
        center=mp.Vector3(0, 0, z_sig),
        size=mp.Vector3(l_step, w_step, t_metal),
    )
    sig_right = mp.Block(
        material=mp.metal,
        center=mp.Vector3(0.5 * (l_step + x_long), 0, z_sig),
        size=mp.Vector3(x_long, w0, t_metal),
    )

    return [substrate, gnd, sig_left, sig_mid, sig_right]


def add_flux_monitors(
    sim: mp.Simulation,
    *,
    fcen: float,
    df: float,
    nfreq: int,
    x_in: float,
    x_out: float,
    sy: float,
    sz: float,
    dpml: float,
) -> tuple[mp.DftFlux, mp.DftFlux]:
    y_span = sy - 2 * dpml
    z_span = sz - 2 * dpml
    flux_in = sim.add_flux(
        fcen,
        df,
        nfreq,
        mp.FluxRegion(
            center=mp.Vector3(x_in, 0, 0),
            size=mp.Vector3(0, y_span, z_span),
            direction=mp.X,
        ),
    )
    flux_out = sim.add_flux(
        fcen,
        df,
        nfreq,
        mp.FluxRegion(
            center=mp.Vector3(x_out, 0, 0),
            size=mp.Vector3(0, y_span, z_span),
            direction=mp.X,
        ),
    )
    return flux_in, flux_out


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
        mp.FluxRegion(
            center=mp.Vector3(x_pos, 0, 0),
            size=mp.Vector3(0, y_span, z_span),
            direction=mp.X,
        ),
    )


def plot_eps_yz(
    sim: mp.Simulation,
    *,
    x_plane: float,
    sy: float,
    sz: float,
    dpml: float,
    eps_r: float,
    out_path: str,
) -> None:
    y_span = sy - 2 * dpml
    z_span = sz - 2 * dpml
    eps = sim.get_array(
        center=mp.Vector3(x_plane, 0, 0),
        size=mp.Vector3(0, y_span, z_span),
        component=mp.Dielectric,
    )
    y = np.linspace(-0.5 * y_span, 0.5 * y_span, eps.shape[0])
    z = np.linspace(-0.5 * z_span, 0.5 * z_span, eps.shape[1])

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    metal_mask = eps < 0
    eps_plot = np.ma.masked_where(metal_mask, eps)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="black")
    im = ax.imshow(
        eps_plot.T,
        origin="lower",
        extent=[y[0], y[-1], z[0], z[-1]],
        aspect="auto",
        cmap=cmap,
        vmin=1.0,
        vmax=eps_r,
    )
    ax.set_xlabel("y [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title(r"$\varepsilon(y,z)$")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("relative permittivity (metal masked)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_eps_xz_geometry(
    sim: mp.Simulation,
    *,
    y_plane: float,
    sx: float,
    sz: float,
    dpml: float,
    eps_r: float,
    x_src: float,
    z_src: float,
    x_in: float,
    x_out: float,
    l_step: float,
    out_path: str,
) -> None:
    eps = sim.get_array(
        center=mp.Vector3(0, y_plane, 0),
        size=mp.Vector3(sx, 0, sz),
        component=mp.Dielectric,
    )
    x = np.linspace(-0.5 * sx, 0.5 * sx, eps.shape[0])
    z = np.linspace(-0.5 * sz, 0.5 * sz, eps.shape[1])

    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    metal_mask = eps < 0
    eps_plot = np.ma.masked_where(metal_mask, eps)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="black")
    im = ax.imshow(
        eps_plot.T,
        origin="lower",
        extent=[x[0], x[-1], z[0], z[-1]],
        aspect="auto",
        cmap=cmap,
        vmin=1.0,
        vmax=eps_r,
    )

    # Lightly shade the PML region (thickness dpml)
    ax.axvspan(-0.5 * sx, -0.5 * sx + dpml, color="gray", alpha=0.12, lw=0)
    ax.axvspan(0.5 * sx - dpml, 0.5 * sx, color="gray", alpha=0.12, lw=0)
    ax.axhspan(-0.5 * sz, -0.5 * sz + dpml, color="gray", alpha=0.12, lw=0)
    ax.axhspan(0.5 * sz - dpml, 0.5 * sz, color="gray", alpha=0.12, lw=0)

    # Reference planes (flux planes)
    ax.axvline(x_in, color="tab:blue", ls="--", lw=1.4)
    ax.axvline(x_out, color="tab:blue", ls="--", lw=1.4)

    # Step-section boundaries (width-change locations)
    ax.axvline(-0.5 * l_step, color="tab:red", ls=":", lw=1.4)
    ax.axvline(0.5 * l_step, color="tab:red", ls=":", lw=1.4)

    ax.plot([x_src], [z_src], marker="o", ms=5.5, color="tab:red")

    ax.text(x_in, z[-1] - 0.1 * sz, "ref", color="tab:blue", ha="center", va="top")
    ax.text(x_out, z[-1] - 0.1 * sz, "ref", color="tab:blue", ha="center", va="top")
    ax.text(0.0, z[-1] - 0.1 * sz, "step", color="tab:red", ha="center", va="top")
    ax.text(x_src, z_src + 0.15 * sz, "src", color="tab:red", ha="center", va="bottom")

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title(r"$\varepsilon(x,z)$ (slice)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("relative permittivity (metal masked)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



def plot_ez_yz(
    sim: mp.Simulation,
    *,
    x_plane: float,
    sy: float,
    sz: float,
    dpml: float,
    out_path: str,
) -> None:
    y_span = sy - 2 * dpml
    z_span = sz - 2 * dpml
    ez = sim.get_array(
        center=mp.Vector3(x_plane, 0, 0),
        size=mp.Vector3(0, y_span, z_span),
        component=mp.Ez,
    )
    y = np.linspace(-0.5 * y_span, 0.5 * y_span, ez.shape[0])
    z = np.linspace(-0.5 * z_span, 0.5 * z_span, ez.shape[1])

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    vmax = np.max(np.abs(ez)) + 1e-12
    im = ax.imshow(
        ez.T,
        origin="lower",
        extent=[y[0], y[-1], z[0], z[-1]],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xlabel("y [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title(r"$E_z(y,z)$ snapshot")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Ez [a.u.]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ez_xy(
    sim: mp.Simulation,
    *,
    z_plane: float,
    sx: float,
    sy: float,
    dpml: float,
    out_path: str,
) -> None:
    x_span = sx - 2 * dpml
    y_span = sy - 2 * dpml
    ez = sim.get_array(
        center=mp.Vector3(0, 0, z_plane),
        size=mp.Vector3(x_span, y_span, 0),
        component=mp.Ez,
    )
    x = np.linspace(-0.5 * x_span, 0.5 * x_span, ez.shape[0])
    y = np.linspace(-0.5 * y_span, 0.5 * y_span, ez.shape[1])

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    im = ax.imshow(
        np.abs(ez).T,
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect="auto",
        cmap="magma",
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(r"$|E_z(x,y)|$ above the line")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("|Ez| [a.u.]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_until_decayed(
    sim: mp.Simulation,
    *,
    x_probe: float,
    z_probe: float,
    decay_target: float = 1e-5,
    min_time: float = 50,
) -> None:
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            min_time,
            mp.Ez,
            mp.Vector3(x_probe, 0, z_probe),
            decay_target,
        )
    )


def run_baseline_case(
    *,
    fcen: float,
    df: float,
    df_src: float,
    nfreq: int,
    sx: float,
    sy: float,
    sz: float,
    dpml: float,
    resolution: int,
    eps_r: float,
    h: float,
    t_metal: float,
    w0: float,
    l_step: float,
    x_in: float,
    x_out: float,
    x_in_list: list[float] | None = None,
    out_dir: str | None = None,
    make_eps_plots: bool = False,
    make_geom_plot: bool = False,
    make_field_plot: bool = False,
    decay_target: float = 1e-5,
    decay_min_time: float = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, mp.FluxData, list[np.ndarray], list[mp.FluxData]]:
    geometry = make_microstrip_geometry(
        eps_r=eps_r,
        h=h,
        t_metal=t_metal,
        w0=w0,
        w_step=w0,
        l_step=l_step,
    )
    x_src = -0.5 * sx + dpml + 0.5
    z_src = -0.5 * h
    src = mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=df_src),
        component=mp.Ez,
        center=mp.Vector3(x_src, 0, z_src),
        size=mp.Vector3(0, w0, 0),
    )
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=mp.Vector3(sx, sy, sz),
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=[src],
    )
    if x_in_list is None:
        x_in_list = [x_in]
    flux_in_list = [
        add_flux_monitor(
            sim,
            fcen=fcen,
            df=df,
            nfreq=nfreq,
            x_pos=xi,
            sy=sy,
            sz=sz,
            dpml=dpml,
        )
        for xi in x_in_list
    ]
    flux_out = add_flux_monitor(
        sim,
        fcen=fcen,
        df=df,
        nfreq=nfreq,
        x_pos=x_out,
        sy=sy,
        sz=sz,
        dpml=dpml,
    )

    sim.init_sim()
    if mp.am_master() and out_dir is not None:
        if make_eps_plots:
            plot_eps_yz(
                sim,
                x_plane=x_in,
                sy=sy,
                sz=sz,
                dpml=dpml,
                eps_r=eps_r,
                out_path=os.path.join(out_dir, "ch07_microstrip_eps.png"),
            )
        if make_geom_plot:
            plot_eps_xz_geometry(
                sim,
                y_plane=0.0,
                sx=sx,
                sz=sz,
                dpml=dpml,
                eps_r=eps_r,
                x_src=x_src,
                z_src=z_src,
                x_in=x_in,
                x_out=x_out,
                l_step=l_step,
                out_path=os.path.join(out_dir, "ch07_microstrip_geometry_xz.png"),
            )

    if make_field_plot and out_dir is not None:
        sim.run(until=160)
        if mp.am_master():
            plot_ez_yz(
                sim,
                x_plane=x_in,
                sy=sy,
                sz=sz,
                dpml=dpml,
                out_path=os.path.join(out_dir, "ch07_microstrip_ez_yz.png"),
            )

    run_until_decayed(sim, x_probe=x_out, z_probe=0.6, decay_target=decay_target, min_time=decay_min_time)

    freqs = np.array(mp.get_flux_freqs(flux_in_list[0]))
    p_inc = np.array(mp.get_fluxes(flux_in_list[0]))
    p_inc_list = [np.array(mp.get_fluxes(f)) for f in flux_in_list]
    p_out = np.array(mp.get_fluxes(flux_out))
    inc_flux_data_in = sim.get_flux_data(flux_in_list[0])
    inc_flux_data_in_list = [sim.get_flux_data(f) for f in flux_in_list]
    sim.reset_meep()
    # For compatibility, reference-plane sweep information is returned in addition to the single-plane return values.
    # If reference-plane sweep is not needed, ignore p_inc_list / inc_flux_data_in_list.
    return freqs, p_inc, p_out, inc_flux_data_in, p_inc_list, inc_flux_data_in_list


def compute_center_s11(
    *,
    w_step: float,
    fcen: float,
    df_src: float,
    sx: float,
    sy: float,
    sz: float,
    dpml: float,
    resolution: int,
    eps_r: float,
    h: float,
    t_metal: float,
    w0: float,
    l_step: float,
    x_in: float,
    x_out: float,
    z_probe: float = 0.6,
    decay_target: float = 1e-4,
    decay_min_time: float = 50,
) -> float:
    freqs, p_inc, _, inc_flux_data_in, _, _ = run_baseline_case(
        fcen=fcen,
        df=0.0,
        df_src=df_src,
        nfreq=1,
        sx=sx,
        sy=sy,
        sz=sz,
        dpml=dpml,
        resolution=resolution,
        eps_r=eps_r,
        h=h,
        t_metal=t_metal,
        w0=w0,
        l_step=l_step,
        x_in=x_in,
        x_out=x_out,
        out_dir=None,
        make_eps_plots=False,
        make_geom_plot=False,
        make_field_plot=False,
        decay_target=decay_target,
        decay_min_time=decay_min_time,
    )

    p_ref_list, _ = run_step_case(
        w_step=w_step,
        fcen=fcen,
        df=0.0,
        df_src=df_src,
        nfreq=1,
        sx=sx,
        sy=sy,
        sz=sz,
        dpml=dpml,
        resolution=resolution,
        eps_r=eps_r,
        h=h,
        t_metal=t_metal,
        w0=w0,
        l_step=l_step,
        x_in_list=[x_in],
        x_out=x_out,
        inc_flux_data_in_list=[inc_flux_data_in],
        snapshot=False,
        figs_dir=None,
        decay_target=decay_target,
        decay_min_time=decay_min_time,
    )
    i0 = int(np.argmin(np.abs(freqs - fcen)))
    p_ref = p_ref_list[0]
    return float(np.sqrt(np.maximum(p_ref[i0], 0.0) / np.maximum(p_inc[i0], 1e-30)))


def plot_power_budget(
    *,
    f_ghz: np.ndarray,
    p_inc: np.ndarray,
    p_out_base: np.ndarray,
    p_ref_step: np.ndarray,
    p_tran_step: np.ndarray,
    out_path: str,
) -> None:
    eps = 1e-30
    s21_sq_base = np.maximum(p_out_base / np.maximum(p_inc, eps), 0.0)
    l_base = np.clip(1.0 - s21_sq_base, 0.0, 1.0)

    s11_sq_step = np.maximum(p_ref_step / np.maximum(p_inc, eps), 0.0)
    s21_sq_step = np.maximum(p_tran_step / np.maximum(p_inc, eps), 0.0)
    l_step = np.clip(1.0 - s11_sq_step - s21_sq_step, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(f_ghz, s11_sq_step, lw=2.0, label=r"$|S_{11}|^2$ (step)")
    ax.plot(f_ghz, s21_sq_step, lw=2.0, label=r"$|S_{21}|^2$ (step)")
    ax.plot(f_ghz, l_step, lw=2.0, label=r"$L$ (step)")

    ax.plot(f_ghz, s21_sq_base, lw=1.6, ls="--", color="tab:blue", alpha=0.8, label=r"$|S_{21}|^2$ (no step)")
    ax.plot(f_ghz, l_base, lw=1.6, ls="--", color="tab:green", alpha=0.8, label=r"$L$ (no step)")

    ax.set_xlabel("f [GHz]")
    ax.set_ylabel("power ratio")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_refplane_sensitivity(
    *,
    x_in_list: list[float],
    s11_list: list[float],
    l_step: float,
    out_path: str,
) -> None:
    dist_from_step = [abs(x_in) - 0.5 * l_step for x_in in x_in_list]
    pairs = sorted(zip(dist_from_step, s11_list))
    dist_from_step_sorted = [p[0] for p in pairs]
    s11_sorted = [p[1] for p in pairs]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(dist_from_step_sorted, s11_sorted, marker="o")
    ax.set_xlabel("distance from discontinuity [mm]")
    ax.set_ylabel(r"$|S_{11}(f_0)|$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_convergence_figure(
    *,
    fcen: float,
    df_src: float,
    sx: float,
    sy: float,
    sz: float,
    dpml: float,
    resolution: int,
    eps_r: float,
    h: float,
    t_metal: float,
    w0: float,
    w_step_demo: float,
    l_step: float,
    x_in: float,
    x_out: float,
    out_path: str,
    decay_target: float = 1e-3,
    s11_main: float | None = None,
) -> None:
    fast = os.environ.get("CH07_CONV_FAST", "0") == "1"
    decay_target_conv = 1e-2 if fast else decay_target
    decay_min_time_conv = 20 if fast else 50

    if s11_main is None:
        s11_main = compute_center_s11(
            w_step=w_step_demo,
            fcen=fcen,
            df_src=df_src,
            sx=sx,
            sy=sy,
            sz=sz,
            dpml=dpml,
            resolution=resolution,
            eps_r=eps_r,
            h=h,
            t_metal=t_metal,
            w0=w0,
            l_step=l_step,
            x_in=x_in,
            x_out=x_out,
            decay_target=decay_target_conv,
            decay_min_time=decay_min_time_conv,
        )

    # Convergence (center frequency only): effects of resolution and PML thickness
    res_list = [6, resolution]
    s11_res = []
    for r in res_list:
        if r == resolution:
            s11_res.append(s11_main)
        else:
            s11_res.append(
                compute_center_s11(
                    w_step=w_step_demo,
                    fcen=fcen,
                    df_src=df_src,
                    sx=sx,
                    sy=sy,
                    sz=sz,
                    dpml=dpml,
                    resolution=r,
                    eps_r=eps_r,
                    h=h,
                    t_metal=t_metal,
                    w0=w0,
                    l_step=l_step,
                    x_in=x_in,
                    x_out=x_out,
                    decay_target=decay_target_conv,
                    decay_min_time=decay_min_time_conv,
                )
            )

    dpml_list = [1.0, dpml]
    s11_dpml = []
    for dp in dpml_list:
        if abs(dp - dpml) < 1e-12:
            s11_dpml.append(s11_main)
        else:
            x_in_dp = -0.5 * sx + dp + 2.0
            x_out_dp = 0.5 * sx - dp - 2.0
            s11_dpml.append(
                compute_center_s11(
                    w_step=w_step_demo,
                    fcen=fcen,
                    df_src=df_src,
                    sx=sx,
                    sy=sy,
                    sz=sz,
                    dpml=dp,
                    resolution=resolution,
                    eps_r=eps_r,
                    h=h,
                    t_metal=t_metal,
                    w0=w0,
                    l_step=l_step,
                    x_in=x_in_dp,
                    x_out=x_out_dp,
                    decay_target=decay_target_conv,
                    decay_min_time=decay_min_time_conv,
                )
            )

    def _disable_y_offset(ax: plt.Axes) -> None:
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        fmt = ax.yaxis.get_major_formatter()
        if hasattr(fmt, "set_useOffset"):
            fmt.set_useOffset(False)
        if hasattr(fmt, "set_scientific"):
            fmt.set_scientific(False)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax = axes[0]
    ax.plot(res_list, s11_res, marker="o")
    ax.set_xlabel("resolution")
    ax.set_ylabel(r"$|S_{11}(f_0)|$")
    ax.grid(True, alpha=0.3)
    _disable_y_offset(ax)

    ax = axes[1]
    ax.plot(dpml_list, s11_dpml, marker="o")
    ax.set_xlabel(r"$d_{\mathrm{PML}}$ [mm]")
    ax.set_ylabel(r"$|S_{11}(f_0)|$")
    ax.grid(True, alpha=0.3)
    _disable_y_offset(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def microstrip_eps_eff(eps_r: float, w: float, h: float) -> float:
    wh = w / h
    if wh <= 0:
        return 1.0
    # Representative approximation (simplified Hammerstad–Jensen)
    return 0.5 * (eps_r + 1.0) + 0.5 * (eps_r - 1.0) * (1.0 / np.sqrt(1.0 + 12.0 / wh))


def plot_time_trace_data(
    t: np.ndarray,
    ez: np.ndarray,
    *,
    t_marks: list[tuple[float, str]],
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(t, ez, lw=1.4)
    ax.set_xlabel("t [meep]")
    ax.set_ylabel(r"$E_z$ [a.u.]")
    ax.grid(True, alpha=0.3)
    x0, x1 = float(t[0]), float(t[-1])
    marks = sorted(t_marks, key=lambda x: x[0])
    y_slots = [0.98, 0.90, 0.82, 0.74]
    for i, (tm, label) in enumerate(marks):
        ax.axvline(tm, color="tab:red", ls="--", lw=1.0, alpha=0.8)
        y = y_slots[i % len(y_slots)]
        near_right = tm > x0 + 0.88 * (x1 - x0)
        dx = -6 if near_right else 6
        ha = "right" if near_right else "left"
        ax.annotate(
            label,
            xy=(tm, y),
            xycoords=("data", "axes fraction"),
            xytext=(dx, 0),
            textcoords="offset points",
            ha=ha,
            va="top",
            fontsize=9,
            color="tab:red",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.2},
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_step_case(
    *,
    w_step: float,
    fcen: float,
    df: float,
    df_src: float,
    nfreq: int,
    sx: float,
    sy: float,
    sz: float,
    dpml: float,
    resolution: int,
    eps_r: float,
    h: float,
    t_metal: float,
    w0: float,
    l_step: float,
    x_in_list: list[float],
    x_out: float,
    inc_flux_data_in_list: list[mp.FluxData],
    snapshot: bool = False,
    figs_dir: str | None = None,
    time_trace_out_path: str | None = None,
    time_trace_dt: float = 1.0,
    time_trace_x_probe: float | None = None,
    time_trace_z_probe: float = 0.6,
    time_trace_marks: list[tuple[float, str]] | None = None,
    decay_target: float = 1e-5,
    decay_min_time: float = 50,
) -> tuple[list[np.ndarray], np.ndarray]:
    geometry = make_microstrip_geometry(
        eps_r=eps_r,
        h=h,
        t_metal=t_metal,
        w0=w0,
        w_step=w_step,
        l_step=l_step,
    )
    src = mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=df_src),
        component=mp.Ez,
        center=mp.Vector3(-0.5 * sx + dpml + 0.5, 0, -0.5 * h),
        size=mp.Vector3(0, w0, 0),
    )
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=mp.Vector3(sx, sy, sz),
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=[src],
    )
    if len(x_in_list) != len(inc_flux_data_in_list):
        raise ValueError("x_in_list and inc_flux_data_in_list must have the same length")
    flux_in_list = [
        add_flux_monitor(
            sim,
            fcen=fcen,
            df=df,
            nfreq=nfreq,
            x_pos=xi,
            sy=sy,
            sz=sz,
            dpml=dpml,
        )
        for xi in x_in_list
    ]
    flux_out = add_flux_monitor(
        sim,
        fcen=fcen,
        df=df,
        nfreq=nfreq,
        x_pos=x_out,
        sy=sy,
        sz=sz,
        dpml=dpml,
    )
    for flux_in, inc_flux_data_in in zip(flux_in_list, inc_flux_data_in_list):
        sim.load_minus_flux_data(flux_in, inc_flux_data_in)

    trace_ts: list[float] = []
    trace_ez: list[float] = []

    do_trace = time_trace_out_path is not None
    x_probe_t = x_in_list[0] if time_trace_x_probe is None else time_trace_x_probe

    def sample(sim: mp.Simulation) -> None:
        if not mp.am_master():
            return
        trace_ts.append(float(sim.meep_time()))
        ez = sim.get_field_point(mp.Ez, mp.Vector3(x_probe_t, 0, time_trace_z_probe))
        trace_ez.append(float(np.real(ez)))

    if snapshot and figs_dir is not None:
        if do_trace:
            sim.run(mp.at_every(time_trace_dt, sample), until=80)
        else:
            sim.run(until=80)
        if mp.am_master():
            plot_ez_xy(
                sim,
                z_plane=0.6,  # above the conductor
                sx=sx,
                sy=sy,
                dpml=dpml,
                out_path=os.path.join(figs_dir, "ch07_microstrip_leak_xy.png"),
            )

    if do_trace:
        sim.run(
            mp.at_every(time_trace_dt, sample),
            until_after_sources=mp.stop_when_fields_decayed(
                decay_min_time,
                mp.Ez,
                mp.Vector3(x_out, 0, time_trace_z_probe),
                decay_target,
            ),
        )
    else:
        run_until_decayed(
            sim,
            x_probe=x_out,
            z_probe=time_trace_z_probe,
            decay_target=decay_target,
            min_time=decay_min_time,
        )

    refl_flux_list = [np.array(mp.get_fluxes(f)) for f in flux_in_list]  # reflected flux often becomes negative
    tran_flux = np.array(mp.get_fluxes(flux_out))
    p_ref_list = [-refl_flux for refl_flux in refl_flux_list]
    p_tran = tran_flux

    if do_trace and mp.am_master():
        plot_time_trace_data(
            np.asarray(trace_ts),
            np.asarray(trace_ez),
            t_marks=[] if time_trace_marks is None else time_trace_marks,
            out_path=time_trace_out_path,
        )
    return p_ref_list, p_tran


def main() -> None:
    # Transmission-line example: microstrip (on a substrate)
    eps_r = 4.2
    h = 1.6  # [mm]
    t_metal = 0.25  # [mm] (round to match grid resolution when needed)

    w0 = 3.0  # [mm] (initial ~50-ohm guess from Chapter 4 approximations)
    l_step = 6.0  # [mm]

    f0_ghz = 2.45
    df_ghz = 2.0
    fcen = ghz_to_meep(f0_ghz)
    df = ghz_to_meep(df_ghz)
    df_src = 5.0 * df  # shorten the pulse (can be wider than the DFT band df)
    nfreq = 21

    dpml = 2.0
    sx, sy, sz = 28.0, 14.0, 11.0
    resolution = 8

    # Reference planes (keep distance from PML and the discontinuity)
    x_in = -0.5 * sx + dpml + 2.0
    x_out = 0.5 * sx - dpml - 2.0

    repo_dir = Path(__file__).resolve().parents[2]
    figs_dir = repo_dir / "figs" / "ch07"
    ensure_dir(str(figs_dir))

    force_regen = os.environ.get("CH07_FORCE_REGEN", "0") == "1"
    out_paths = {
        "eps": str(figs_dir / "ch07_microstrip_eps.png"),
        "geom": str(figs_dir / "ch07_microstrip_geometry_xz.png"),
        "ez_yz": str(figs_dir / "ch07_microstrip_ez_yz.png"),
        "leak": str(figs_dir / "ch07_microstrip_leak_xy.png"),
        "time": str(figs_dir / "ch07_microstrip_time_trace.png"),
        "power": str(figs_dir / "ch07_microstrip_power_budget.png"),
        "refplane": str(figs_dir / "ch07_microstrip_refplane_sensitivity.png"),
        "conv": str(figs_dir / "ch07_microstrip_convergence.png"),
        "sparams": str(figs_dir / "ch07_microstrip_sparams.png"),
        "sweep": str(figs_dir / "ch07_microstrip_sweep_wstep.png"),
    }
    missing = {k: (force_regen or (not os.path.exists(p))) for k, p in out_paths.items()}
    if not any(missing.values()):
        return

    # Candidate input reference planes for sensitivity checks. Move toward the discontinuity and check stabilization.
    x_in_list = [x_in, x_in + 2.0, x_in + 4.0, x_in + 6.0]
    x_in_limit = -0.5 * l_step - 0.5  # keep at least 0.5 mm away from the step boundary (-l_step/2)
    x_in_list = [xi for xi in x_in_list if xi <= x_in_limit]

    need_full_run = any(missing[k] for k in ["eps", "geom", "ez_yz", "leak", "time", "power", "refplane", "sparams", "sweep"])
    if not need_full_run and missing["conv"]:
        generate_convergence_figure(
            fcen=fcen,
            df_src=df_src,
            sx=sx,
            sy=sy,
            sz=sz,
            dpml=dpml,
            resolution=resolution,
            eps_r=eps_r,
            h=h,
            t_metal=t_metal,
            w0=w0,
            w_step_demo=5.0,
            l_step=l_step,
            x_in=x_in,
            x_out=x_out,
            out_path=out_paths["conv"],
            decay_target=1e-3,
            s11_main=None,
        )
        return

    # (1) Normalization (no step): store incident power and DFT data at the reference plane
    freqs, p_inc, p_out_base, inc_flux_data_in, p_inc_list, inc_flux_data_in_list = run_baseline_case(
        fcen=fcen,
        df=df,
        df_src=df_src,
        nfreq=nfreq,
        sx=sx,
        sy=sy,
        sz=sz,
        dpml=dpml,
        resolution=resolution,
        eps_r=eps_r,
        h=h,
        t_metal=t_metal,
        w0=w0,
        l_step=l_step,
        x_in=x_in,
        x_out=x_out,
        x_in_list=x_in_list,
        out_dir=figs_dir,
        make_eps_plots=missing["eps"],
        make_geom_plot=missing["geom"],
        make_field_plot=missing["ez_yz"],
    )
    f_ghz = meep_to_ghz(freqs)

    # (2) With step: evaluate reflected/transmitted power and visualize field leakage
    w_step_demo = 5.0

    # Rough time-domain schedule (incident, step reflection, boundary reflection)
    eps_eff = microstrip_eps_eff(eps_r, w0, h)
    v_eff = 1.0 / np.sqrt(eps_eff)
    x_src = -0.5 * sx + dpml + 0.5
    t_inc = abs(x_in - x_src) / v_eff
    t_step = t_inc + 2.0 * (abs(x_in) - 0.5 * l_step) / v_eff
    t_bnd = t_inc + 2.0 * ((0.5 * sx - dpml) - x_in) / v_eff

    p_ref_list_step, p_tran_step = run_step_case(
        w_step=w_step_demo,
        fcen=fcen,
        df=df,
        df_src=df_src,
        nfreq=nfreq,
        sx=sx,
        sy=sy,
        sz=sz,
        dpml=dpml,
        resolution=resolution,
        eps_r=eps_r,
        h=h,
        t_metal=t_metal,
        w0=w0,
        l_step=l_step,
        x_in_list=x_in_list,
        x_out=x_out,
        inc_flux_data_in_list=inc_flux_data_in_list,
        snapshot=missing["leak"],
        figs_dir=figs_dir,
        time_trace_out_path=(out_paths["time"] if missing["time"] else None),
        time_trace_dt=1.0,
        time_trace_x_probe=x_in,
        time_trace_z_probe=0.6,
        time_trace_marks=[
            (t_inc, "incident"),
            (t_step, "step refl"),
            (t_bnd, "boundary refl"),
        ],
    )
    p_ref_step = p_ref_list_step[0]

    i0 = int(np.argmin(np.abs(freqs - fcen)))
    s11_main = float(np.sqrt(np.maximum(p_ref_step[i0], 0.0) / np.maximum(p_inc[i0], 1e-30)))

    if mp.am_master():
        if missing["power"]:
            plot_power_budget(
                f_ghz=f_ghz,
                p_inc=p_inc,
                p_out_base=p_out_base,
                p_ref_step=p_ref_step,
                p_tran_step=p_tran_step,
                out_path=out_paths["power"],
            )

    # Reference-plane location sensitivity (center frequency only): evaluate multiple planes within the same simulation
    s11_refplane = []
    for p_ref, p_inc_i in zip(p_ref_list_step, p_inc_list):
        s11_refplane.append(float(np.sqrt(np.maximum(p_ref[i0], 0.0) / np.maximum(p_inc_i[i0], 1e-30))))
    if mp.am_master():
        if missing["refplane"]:
            plot_refplane_sensitivity(
                x_in_list=x_in_list,
                s11_list=s11_refplane,
                l_step=l_step,
                out_path=out_paths["refplane"],
            )

    if mp.am_master() and missing["conv"]:
        generate_convergence_figure(
            fcen=fcen,
            df_src=df_src,
            sx=sx,
            sy=sy,
            sz=sz,
            dpml=dpml,
            resolution=resolution,
            eps_r=eps_r,
            h=h,
            t_metal=t_metal,
            w0=w0,
            w_step_demo=w_step_demo,
            l_step=l_step,
            x_in=x_in,
            x_out=x_out,
            out_path=out_paths["conv"],
            decay_target=1e-3,
            s11_main=s11_main,
        )

    # Step-width sweep: inspect |S11| at the center frequency
    sweep_w = [2.0, 3.0, 4.0, 5.0]
    s11_center = []
    for w_step in sweep_w:
        if abs(w_step - w0) < 1e-12:
            # Without a step, reflection is almost zero. The sweep skips execution and treats it as 0.
            s11_center.append(0.0)
            continue
        if abs(w_step - w_step_demo) < 1e-12:
            p_ref = p_ref_step
            p_tran = p_tran_step
        else:
            p_ref_list, p_tran = run_step_case(
                w_step=w_step,
                fcen=fcen,
                df=df,
                df_src=df_src,
                nfreq=nfreq,
                sx=sx,
                sy=sy,
                sz=sz,
                dpml=dpml,
                resolution=resolution,
                eps_r=eps_r,
                h=h,
                t_metal=t_metal,
                w0=w0,
                l_step=l_step,
                x_in_list=[x_in],
                x_out=x_out,
                inc_flux_data_in_list=[inc_flux_data_in],
                snapshot=False,
                figs_dir=None,
            )
            p_ref = p_ref_list[0]
        s11_center.append(float(np.sqrt(np.maximum(p_ref[i0], 0.0) / p_inc[i0])))

    if not mp.am_master():
        return

    # Power ratio → |S| magnitude (dB). Use 20log10|S| = 10log10(P_ratio).
    eps = 1e-30
    s21_base_db = 10 * np.log10(np.maximum(p_out_base / p_inc, eps))
    s11_base_db = np.full_like(s21_base_db, -60.0)
    s21_step_db = 10 * np.log10(np.maximum(p_tran_step / p_inc, eps))
    s11_step_db = 10 * np.log10(np.maximum(p_ref_step / p_inc, eps))

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True)
    ax = axes[0]
    ax.plot(f_ghz, s11_base_db, label="no step", lw=1.8)
    ax.plot(f_ghz, s11_step_db, label=f"step (w={w_step_demo:.1f} mm)", lw=1.8, ls="--")
    ax.set_ylabel(r"$|S_{11}|$ [dB]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(-60, 0)

    ax = axes[1]
    ax.plot(f_ghz, s21_base_db, label="no step", lw=1.8)
    ax.plot(f_ghz, s21_step_db, label=f"step (w={w_step_demo:.1f} mm)", lw=1.8, ls="--")
    ax.set_xlabel("f [GHz]")
    ax.set_ylabel(r"$|S_{21}|$ [dB]")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-10, 1)

    fig.tight_layout()
    if missing["sparams"]:
        fig.savefig(out_paths["sparams"], dpi=200)
    plt.close(fig)

    # Plot: step-width sweep
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(sweep_w, s11_center, marker="o")
    ax.set_xlabel(r"$w_{\mathrm{step}}$ [mm]")
    ax.set_ylabel(r"$|S_{11}(f_0)|$")
    ax.set_title(f"S11 sensitivity at f0={f0_ghz:.2f} GHz")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if missing["sweep"]:
        fig.savefig(out_paths["sweep"], dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
