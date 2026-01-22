import meep as mp

# Unit convention: 1 [meep length] = 1 mm
c0 = 299_792_458.0  # [m/s]
L0 = 1e-3  # [m]

f0 = 2.45e9 * (L0 / c0)
fwidth = 2.0e9 * (L0 / c0)

sx, sy, dpml = 10.0, 6.0, 1.0  # [meep]（= mm）
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
    resolution=20,
)
sim.run(until=200)

if mp.am_master():
    ez = sim.get_field_point(mp.Ez, mp.Vector3(probe_x, 0, 0))
    print(f"Ez@probe = {ez}")
    print("done")
