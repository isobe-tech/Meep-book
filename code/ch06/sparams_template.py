import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp
import numpy as np


def main() -> None:
    # Template showing the basic structure of S-parameter extraction.
    # The example is a simple 2D waveguide with a small discontinuity; later chapters replace it with lines/waveguides.

    fcen = 0.15  # center frequency (Meep units)
    df = 0.10  # bandwidth (Meep units)
    nfreq = 41

    sx, sy = 16.0, 8.0
    dpml = 1.0
    resolution = 20

    cell = mp.Vector3(sx, sy, 0)
    pml_layers = [mp.PML(dpml)]

    # In 2D, ODD_Z corresponds to Ez polarization (see MEEP mode decomposition documentation).
    eig_parity = mp.ODD_Z

    # Waveguide (slab): infinite in x, width w in y
    n_wg = 2.0
    w = 2.0
    waveguide = mp.Block(
        material=mp.Medium(index=n_wg),
        center=mp.Vector3(),
        size=mp.Vector3(mp.inf, w, mp.inf),
    )

    # Example discontinuity: change the refractive index locally in a short section
    n_dev = 1.0
    dev_len = 2.0
    device = mp.Block(
        material=mp.Medium(index=n_dev),
        center=mp.Vector3(),
        size=mp.Vector3(dev_len, w, mp.inf),
    )

    # Port reference planes (keep enough distance from PML)
    x_in = -0.5 * sx + dpml + 2.0
    x_out = 0.5 * sx - dpml - 2.0

    # Excitation (eigenmode source)
    src = mp.EigenModeSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=df),
        center=mp.Vector3(-0.5 * sx + dpml + 0.5, 0, 0),
        size=mp.Vector3(0, w, 0),
        direction=mp.X,
        eig_band=1,
        eig_parity=eig_parity,
        eig_match_freq=True,
    )

    def add_port_monitors(sim: mp.Simulation):
        mode_in = sim.add_mode_monitor(
            fcen,
            df,
            nfreq,
            mp.ModeRegion(
                center=mp.Vector3(x_in, 0, 0),
                size=mp.Vector3(0, sy - 2 * dpml, 0),
                direction=mp.X,
            ),
        )
        mode_out = sim.add_mode_monitor(
            fcen,
            df,
            nfreq,
            mp.ModeRegion(
                center=mp.Vector3(x_out, 0, 0),
                size=mp.Vector3(0, sy - 2 * dpml, 0),
                direction=mp.X,
            ),
        )
        return mode_in, mode_out

    def run_until_decayed(sim: mp.Simulation):
        sim.run(
            until_after_sources=mp.stop_when_fields_decayed(
                50,
                mp.Ez,
                mp.Vector3(x_out, 0, 0),
                1e-6,
            )
        )

    # (1) Normalization (no device): store the incident-wave coefficient and DFT data of the incident field
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=[waveguide],
        sources=[src],
    )
    mon_in, mon_out = add_port_monitors(sim)
    run_until_decayed(sim)

    res_in = sim.get_eigenmode_coefficients(mon_in, bands=[1], eig_parity=eig_parity)
    a1 = res_in.alpha[0, :, 0]  # forward (incident)
    inc_flux_data_in = sim.get_flux_data(mon_in)
    freqs = np.array(mp.get_flux_freqs(mon_in))

    sim.reset_meep()

    # (2) With device: subtract the incident field at the reference plane and keep reflection only (for S11)
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=[waveguide, device],
        sources=[src],
    )
    mon_in, mon_out = add_port_monitors(sim)
    sim.load_minus_flux_data(mon_in, inc_flux_data_in)
    run_until_decayed(sim)

    res_refl = sim.get_eigenmode_coefficients(mon_in, bands=[1], eig_parity=eig_parity)
    res_tran = sim.get_eigenmode_coefficients(mon_out, bands=[1], eig_parity=eig_parity)
    b1 = res_refl.alpha[0, :, 1]  # backward (reflected)
    b2 = res_tran.alpha[0, :, 0]  # forward (transmitted)

    s11 = b1 / a1
    s21 = b2 / a1

    if mp.am_master():
        print("# f, |S11|, |S21|")
        for f, x, y in zip(freqs, np.abs(s11), np.abs(s21)):
            print(f"{f:.6e}, {x:.6f}, {y:.6f}")

        # Visualization (to reproduce figures used in the book)
        repo_dir = Path(__file__).resolve().parents[2]
        figs_dir = repo_dir / "figs" / "ch06"
        figs_dir.mkdir(parents=True, exist_ok=True)

        eps = 1e-12
        s11_db = 20 * np.log10(np.maximum(np.abs(s11), eps))
        s21_db = 20 * np.log10(np.maximum(np.abs(s21), eps))

        fig, ax = plt.subplots(figsize=(6.6, 3.4))
        ax.plot(freqs, s11_db, label=r"$S_{11}$")
        ax.plot(freqs, s21_db, label=r"$S_{21}$")
        ax.set_xlabel("f [meep]")
        ax.set_ylabel("magnitude [dB]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(str(figs_dir / "ch06_sparams_example.png"), dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
