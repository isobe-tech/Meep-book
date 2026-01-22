import meep as mp

# Unit convention: 1 [meep length] = 1 mm
c0 = 299_792_458.0  # [m/s]
L0 = 1e-3  # [m]

f0 = 2.45e9 * (L0 / c0)
fwidth = 2.0e9 * (L0 / c0)

sx = sy = sz = 6.0  # [meep]（= mm）
dpml = 1.0

cell = mp.Vector3(sx, sy, sz)
src = mp.Source(mp.GaussianSource(frequency=f0, fwidth=fwidth), component=mp.Ez, center=mp.Vector3())

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=[mp.PML(dpml)],
    sources=[src],
    resolution=8,
)
sim.run(until=120)

if mp.am_master():
    ez = sim.get_field_point(mp.Ez, mp.Vector3(1.0, 0, 0))
    print(f"Ez@probe = {ez}")
    print("done")
