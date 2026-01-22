import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp
import numpy as np


def main() -> None:
    # Helper script that connects “time waveform” and “frequency content” in figures.
    # Ez(t) at a probe point is sampled and its spectrum is plotted using a simple FFT.

    # Unit convention: 1 [meep length] = 1 mm
    c0 = 299_792_458.0  # [m/s]
    L0 = 1e-3  # [m]

    f0 = 2.45e9 * (L0 / c0)
    fwidth = 2.0e9 * (L0 / c0)

    sx, sy, dpml = 10.0, 6.0, 1.0
    resolution = 20
    cell = mp.Vector3(sx, sy, 0)

    src_x = -0.5 * sx + dpml + 1.0
    probe_x = 0.5 * sx - dpml - 1.0

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

    dt_sample = 0.5
    t_end = 300.0
    times: list[float] = []
    values: list[complex] = []

    def sample(sim: mp.Simulation) -> None:
        times.append(sim.meep_time())
        values.append(sim.get_field_point(mp.Ez, mp.Vector3(probe_x, 0, 0)))

    sim.run(mp.at_every(dt_sample, sample), until=t_end)

    if not mp.am_master():
        return

    t = np.array(times)
    x = np.array(values)
    x_re = np.real(x)
    x_re = x_re - np.mean(x_re)

    window = np.hanning(len(x_re))
    xw = x_re * window

    freqs = np.fft.rfftfreq(len(xw), d=dt_sample)
    spectrum = np.fft.rfft(xw)
    mag = np.abs(spectrum)
    mag = mag / (np.max(mag) + 1e-30)
    mag_db = 20 * np.log10(np.maximum(mag, 1e-12))

    t_span = float(t[-1] - t[0])
    df_est = 1.0 / t_span if t_span > 0 else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

    ax = axes[0]
    ax.plot(t, x_re, color="C0")
    ax.set_xlabel("t [meep]")
    ax.set_ylabel(r"$\mathrm{Re}\,E_z(t)$")
    ax.grid(True, alpha=0.3)
    ax.set_title("time waveform")

    ax = axes[1]
    ax.plot(freqs, mag_db, color="C1")
    ax.axvline(f0, color="black", linewidth=0.8, alpha=0.7, linestyle="--", label=r"$f_0$")
    ax.set_xlabel("f [meep]")
    ax.set_ylabel("normalized magnitude [dB]")
    ax.set_ylim(-80, 5)
    ax.grid(True, alpha=0.3)
    ax.set_title(rf"spectrum ($\Delta f \approx {df_est:.3g}$)")
    ax.legend(loc="best")

    fig.tight_layout()

    repo_dir = Path(__file__).resolve().parents[2]
    figs_dir = repo_dir / "figs" / "ch06"
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(figs_dir / "ch06_wave2d_probe_spectrum.png"), dpi=200)
    plt.close(fig)

    print("done")


if __name__ == "__main__":
    main()
