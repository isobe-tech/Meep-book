import meep as mp


def main() -> None:
    # Template showing the basic structure of N2F (near-to-far).
    # Radiation from a point source is collected on a near surface, and a complex far field is obtained at an arbitrary point.

    fcen = 0.15
    df = 0.10
    nfreq = 1

    sx = sy = sz = 10.0
    dpml = 1.0
    resolution = 10

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

    # Place the near surface outside PML, in a homogeneous medium (vacuum here).
    # A closed surface with 6 faces captures radiation in all directions.
    d = 3.0
    L = 2 * d
    n2f = sim.add_near2far(
        fcen,
        df,
        nfreq,
        mp.Near2FarRegion(center=mp.Vector3(+d, 0, 0), size=mp.Vector3(0, L, L), direction=mp.X, weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(-d, 0, 0), size=mp.Vector3(0, L, L), direction=mp.X, weight=-1),
        mp.Near2FarRegion(center=mp.Vector3(0, +d, 0), size=mp.Vector3(L, 0, L), direction=mp.Y, weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(0, -d, 0), size=mp.Vector3(L, 0, L), direction=mp.Y, weight=-1),
        mp.Near2FarRegion(center=mp.Vector3(0, 0, +d), size=mp.Vector3(L, L, 0), direction=mp.Z, weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(0, 0, -d), size=mp.Vector3(L, L, 0), direction=mp.Z, weight=-1),
    )

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50,
            mp.Ez,
            mp.Vector3(d, 0, 0),
            1e-6,
        ),
    )

    # The far field is returned as 6 complex components (Ex, Ey, Ez, Hx, Hy, Hz).
    # Circular-polarization evaluation depends on the phase difference between two orthogonal components.
    r = 1000.0
    ff = sim.get_farfield(n2f, mp.Vector3(r, 0, 0))

    if mp.am_master():
        ex, ey, ez, hx, hy, hz = ff
        print(f"Ex = {ex}")
        print(f"Ey = {ey}")
        print(f"Ez = {ez}")
        print("done")


if __name__ == "__main__":
    main()
