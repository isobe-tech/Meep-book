from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

# In numpy 2.x, some aliases were removed. Older dependencies may fail to import because of that.
# Figure generation does not require strict aliasing, so minimal compatibility aliases are provided.
_NP_ALIASES: dict[str, str] = {
    "bool8": "bool_",
    "object0": "object_",
    "int0": "intp",
    "uint0": "uintp",
    "float_": "float64",
    "complex_": "complex128",
    "cfloat": "complex128",
    "singlecomplex": "complex64",
    "string_": "bytes_",
    "unicode_": "str_",
    "str0": "str_",
    "bytes0": "bytes_",
    "longfloat": "longdouble",
    "clongfloat": "clongdouble",
    "longcomplex": "clongdouble",
    "void0": "void",
}
for _dst, _src in _NP_ALIASES.items():
    if not hasattr(np, _dst) and hasattr(np, _src):
        setattr(np, _dst, getattr(np, _src))  # type: ignore[attr-defined]

try:
    import cadquery as cq
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "cadquery was not found. Install cadquery in the conda environment before running.\n"
        "Example: mamba env create -f environment_cad.yml"
    ) from e


@dataclass(frozen=True)
class Part:
    name: str
    wp: cq.Workplane
    color: tuple[float, float, float]
    alpha: float = 1.0


@dataclass(frozen=True)
class Annotation:
    label: str
    point: tuple[float, float, float]
    color: str = "tab:red"


@dataclass(frozen=True)
class CadFigure:
    base: str
    parts: tuple[Part, ...]
    annotations: tuple[Annotation, ...]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _tessellate(solid: cq.Shape, tol: float) -> tuple[np.ndarray, np.ndarray]:
    verts, tris = solid.tessellate(tol)
    v = np.asarray([(p.x, p.y, p.z) for p in verts], dtype=float)
    t = np.asarray(tris, dtype=int)
    return v, t


def _set_axes_equal(ax) -> None:
    # Not needed for 2D projection (kept for compatibility)
    return


def _rotation_matrix(*, elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)

    cz, sz = np.cos(azim), np.sin(azim)
    cx, sx = np.cos(elev), np.sin(elev)

    # Rotate around z first, then around x (right-handed system)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    return rx @ rz


def _project(points: np.ndarray, *, elev_deg: float, azim_deg: float) -> np.ndarray:
    r = _rotation_matrix(elev_deg=elev_deg, azim_deg=azim_deg)
    return points @ r.T


def _apply_perspective(
    points_view: np.ndarray,
    *,
    z_cam: float,
    focal: float,
) -> np.ndarray:
    # Simple perspective projection: place the camera at (0,0,z_cam) and look toward -z.
    # Choose z_cam large enough in view coordinates (z_cam > max(z)).
    out = points_view.copy()
    denom = z_cam - points_view[:, 2]
    denom = np.maximum(denom, 1e-6)
    s = focal / denom
    out[:, 0] = out[:, 0] * s
    out[:, 1] = out[:, 1] * s
    return out


def _tri_area_2d(p2: np.ndarray) -> float:
    a = p2[0]
    b = p2[1]
    c = p2[2]
    # Absolute value of the z component of a 2D cross product
    return 0.5 * float(abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])))


def _subdivide_triangle(
    p: np.ndarray,
    *,
    max_edge: float,
    min_area_2d: float,
    min_depth_range: float,
    max_depth: int,
    _depth: int = 0,
) -> list[np.ndarray]:
    if _depth >= max_depth:
        return [p]

    p2 = p[:, :2]
    if _tri_area_2d(p2) < min_area_2d:
        return [p]

    e01 = float(np.linalg.norm(p[0] - p[1]))
    e12 = float(np.linalg.norm(p[1] - p[2]))
    e20 = float(np.linalg.norm(p[2] - p[0]))
    depth_range = float(np.max(p[:, 2]) - np.min(p[:, 2]))
    if max(e01, e12, e20) <= max_edge and depth_range <= min_depth_range:
        return [p]

    # When a large polygon (such as a plane) is tessellated into only a few triangles,
    # the average triangle depth is pulled toward a nearby corner, which can break ordering near the center.
    # Subdivide only large-area triangles to localize the depth gradient.
    m01 = 0.5 * (p[0] + p[1])
    m12 = 0.5 * (p[1] + p[2])
    m20 = 0.5 * (p[2] + p[0])

    children = [
        np.stack([p[0], m01, m20], axis=0),
        np.stack([m01, p[1], m12], axis=0),
        np.stack([m20, m12, p[2]], axis=0),
        np.stack([m01, m12, m20], axis=0),
    ]

    out: list[np.ndarray] = []
    for ch in children:
        out.extend(
            _subdivide_triangle(
                ch,
                max_edge=max_edge,
                min_area_2d=min_area_2d,
                min_depth_range=min_depth_range,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
        )
    return out


def _render_isometric_2d(
    fig: CadFigure,
    *,
    out_png: str,
    tol: float,
    elev_deg: float,
    azim_deg: float,
) -> None:
    tris_2d: list[np.ndarray] = []
    face_depth: list[float] = []
    face_fc: list[tuple[float, float, float, float]] = []
    face_ec: list[tuple[float, float, float, float]] = []
    face_lw: list[float] = []

    # NOTE: mplot3d z-ordering tends to be unstable, so triangles are sorted manually.
    # Sorting uses depth along the camera direction (view z) with a per-part bias to stabilize layering (GND < substrate < patch/conductor).
    depth_mode = "view"

    depth_bias_by_part: dict[str, float] = {
        # far
        "gnd": -10_000.0,
        "substrate": -5_000.0,
        # near-ish
        "patch": 0.0,
        "via": 500.0,
        "wire": 500.0,
        "feed": 600.0,
    }

    # For depth_mode="view", subdivide large planes to reduce ordering artifacts.
    split_max_edge = 8.0
    split_min_area_2d = 200.0
    split_min_depth_range = 3.0
    split_max_depth = 5

    # Project all vertices first to determine the camera position for perspective projection.
    projected_parts: list[tuple[int, Part, np.ndarray, np.ndarray, np.ndarray]] = []
    zmax = -1.0e30
    max_xy = 0.0
    for part_idx, part in enumerate(fig.parts):
        v3, t = _tessellate(part.wp.val(), tol=tol)
        vr = _project(v3, elev_deg=elev_deg, azim_deg=azim_deg)
        zmax = max(zmax, float(np.max(vr[:, 2])))
        max_xy = max(max_xy, float(np.max(np.abs(vr[:, :2]))))
        projected_parts.append((part_idx, part, v3, t, vr))

    # Set focal based on the view-coordinate scale. Larger values reduce perspective distortion.
    # For structure figures, a CAD-like isometric feel is prioritized, so perspective distortion is kept weak.
    focal = max(1.0, 12.0 * max_xy)
    z_cam = zmax + focal

    # Simple Lambert-like lighting for appearance.
    light_dir = np.array([0.25, 0.35, 1.0], dtype=float)
    light_dir /= float(np.linalg.norm(light_dir))
    ambient = 0.60
    diffuse = 0.40

    for part_idx, part, v3, t, vr in projected_parts:
        # Visible internal triangle edges tend to look “mesh-like”.
        # Hide edges on planes (gnd/substrate/patch) and keep faint edges only on thin solids (wire/via/feed).
        if part.name in {"gnd", "substrate", "patch"}:
            edge_rgba = (0.0, 0.0, 0.0, 0.0)
            lw = 0.0
        else:
            edge_rgba = (0.0, 0.0, 0.0, 0.12)
            lw = 0.25

        do_split = depth_mode == "view" and part.name == "gnd"
        for tri in t:
            p_view = vr[tri]  # (3,3) in view-coordinates
            p_src = v3[tri]  # (3,3) in original coordinates
            sub_tris = (
                _subdivide_triangle(
                    p_view,
                    max_edge=split_max_edge,
                    min_area_2d=split_min_area_2d,
                    min_depth_range=split_min_depth_range,
                    max_depth=split_max_depth,
                )
                if do_split
                else [p_view]
            )
            for p_sub in sub_tris:
                p_sub_p = _apply_perspective(p_sub, z_cam=z_cam, focal=focal)
                tris_2d.append(p_sub_p[:, :2])
                depth = float(np.mean(p_sub[:, 2])) if depth_mode == "view" else float(np.mean(p_src[:, 2]))
                depth += depth_bias_by_part.get(part.name, 0.0)
                # For equal depth, nudge later parts slightly forward.
                face_depth.append(depth + 1e-3 * part_idx)
                # Add depth via per-triangle shading (constant on planes, smoothly varying on curved surfaces).
                n = np.cross(p_sub[1] - p_sub[0], p_sub[2] - p_sub[0])
                n_norm = float(np.linalg.norm(n))
                if n_norm < 1e-12:
                    lam = 1.0
                else:
                    n = n / n_norm
                    lam = float(abs(np.dot(n, light_dir)))
                shade = ambient + diffuse * lam
                rgb = tuple(float(np.clip(c * shade, 0.0, 1.0)) for c in part.color)
                face_fc.append((*rgb, part.alpha))
                face_ec.append(edge_rgba)
                face_lw.append(lw)

    order = np.argsort(face_depth)  # far -> near
    polys = [tris_2d[i] for i in order]
    fcs = [face_fc[i] for i in order]
    ecs = [face_ec[i] for i in order]
    lws = [face_lw[i] for i in order]

    all_xy = np.vstack(polys)
    xmin, ymin = np.min(all_xy, axis=0)
    xmax, ymax = np.max(all_xy, axis=0)
    span = max(xmax - xmin, ymax - ymin)
    pad = 0.02 * span
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    # Choose the drawing region based on the post-projection aspect ratio to avoid stretched-looking figures.
    xspan = float(xmax - xmin)
    yspan = float(ymax - ymin)
    base_w = 6.0
    fig_h = base_w * (yspan / max(xspan, 1e-9))
    fig_h = float(np.clip(fig_h, 4.2, 6.0))
    fig_mpl, ax = plt.subplots(figsize=(base_w, fig_h), dpi=160)

    coll = PolyCollection(polys, facecolors=fcs, edgecolors=ecs, linewidths=lws, antialiaseds=False)
    ax.add_collection(coll)

    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Annotations (2D projection). Avoid non-English labels in the figure; explain details in the main text caption.
    text_bbox = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88)
    inset = 0.045 * span
    for ann in fig.annotations:
        p3 = np.asarray([ann.point], dtype=float)
        pr = _project(p3, elev_deg=elev_deg, azim_deg=azim_deg)
        pr = _apply_perspective(pr, z_cam=z_cam, focal=focal)[0]
        x2, y2 = float(pr[0]), float(pr[1])
        ax.scatter([x2], [y2], s=18, color=ann.color, zorder=3)
        # With bbox_inches="tight", labels outside the figure enlarge margins.
        # Keep labels inside and point with short arrows in directions that avoid overlap.
        label_lower = ann.label.strip().lower()
        preferred_dir: dict[str, tuple[float, float]] = {
            # A small number of labels makes rule-based overlap avoidance sufficient.
            "gnd": (0.10, -1.0),
            "feed": (1.0, 0.15),
            "corner cut": (-1.0, 0.55),
        }
        if label_lower in preferred_dir:
            ux, uy = preferred_dir[label_lower]
            n = (ux * ux + uy * uy) ** 0.5
            ux, uy = ux / n, uy / n
        else:
            vx = cx - x2
            vy = cy - y2
            n = (vx * vx + vy * vy) ** 0.5
            if n < 1e-9:
                ux, uy = 0.7, 0.7
            else:
                ux, uy = vx / n, vy / n

        # Keep annotation leader lines short to avoid a “tacked on” look.
        offset_by_label: dict[str, float] = {
            "gnd": 0.080,
            "corner cut": 0.075,
            "feed": 0.060,
        }
        offset = offset_by_label.get(label_lower, 0.060) * span
        lx = x2 + offset * ux
        ly = y2 + offset * uy
        lx = min(max(lx, xmin + inset), xmax - inset)
        ly = min(max(ly, ymin + inset), ymax - inset)

        # Align label text so that it extends along the leader-line direction (short lines reduce overlap).
        ha = "left" if ux >= 0 else "right"
        va = "bottom" if uy >= 0 else "top"

        ax.annotate(
            ann.label,
            xy=(x2, y2),
            xytext=(lx, ly),
            textcoords="data",
            ha=ha,
            va=va,
            color=ann.color,
            bbox=text_bbox,
            arrowprops=dict(arrowstyle="-", lw=0.9, color=ann.color, shrinkA=2, shrinkB=0),
            zorder=4,
        )

    ensure_dir(os.path.dirname(out_png))
    fig_mpl.savefig(out_png, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(fig_mpl)


def render_isometric(
    fig: CadFigure,
    *,
    out_png: str,
    tol: float = 0.2,
    elev: float = 35.264,
    azim: float = 45.0,
) -> None:
    # mplot3d depth sorting is unstable; annotations and thin conductors can slip behind surfaces.
    # Use a custom isometric projection and draw triangles back-to-front to avoid artifacts.
    _render_isometric_2d(fig, out_png=out_png, tol=tol, elev_deg=elev, azim_deg=azim)


def export_step(parts: Iterable[Part], out_step: str) -> None:
    compound = cq.Compound.makeCompound([p.wp.val() for p in parts])
    ensure_dir(os.path.dirname(out_step))
    cq.exporters.export(compound, out_step)


def make_monopole(*, gnd_xy: float = 60.0, gnd_t: float = 0.5, h: float = 25.0) -> CadFigure:
    # z points upward. The origin is placed near the feed point.
    gnd = cq.Workplane("XY").box(gnd_xy, gnd_xy, gnd_t, centered=(True, True, False))
    wire_r = 0.5
    wire = cq.Workplane("XY").circle(wire_r).extrude(h).translate((0, 0, gnd_t))
    feed = cq.Workplane("XY").sphere(1.0).translate((0, 0, gnd_t))
    return CadFigure(
        base="ch08_monopole",
        parts=(
            Part("gnd", gnd, (0.55, 0.55, 0.55), 1.0),
            Part("wire", wire, (0.25, 0.25, 0.25), 1.0),
            Part("feed", feed, (0.85, 0.1, 0.1), 1.0),
        ),
        annotations=(
            Annotation("feed", (0.0, 0.0, gnd_t)),
            Annotation("GND", (0.45 * gnd_xy, 0.45 * gnd_xy, 0.5 * gnd_t), color="tab:blue"),
        ),
    )


def make_patch(
    *,
    sub_xy: float = 60.0,
    h: float = 1.6,
    gnd_t: float = 0.15,
    patch_w: float = 38.0,
    patch_l: float = 30.0,
    patch_t: float = 0.15,
    feed_x: float = -6.0,
) -> CadFigure:
    # Use z=0 as the substrate top surface, and place GND at z=-h.
    substrate = (
        cq.Workplane("XY")
        .box(sub_xy, sub_xy, h, centered=(True, True, False))
        .translate((0, 0, -h))
    )
    gnd = (
        cq.Workplane("XY")
        .box(sub_xy, sub_xy, gnd_t, centered=(True, True, False))
        .translate((0, 0, -h - gnd_t))
    )
    patch = cq.Workplane("XY").box(patch_l, patch_w, patch_t, centered=(True, True, False))

    via_r = 0.35
    via = (
        cq.Workplane("XY")
        .circle(via_r)
        .extrude(h + gnd_t + patch_t)
        .translate((feed_x, 0, -h - gnd_t))
    )
    feed = cq.Workplane("XY").sphere(0.9).translate((feed_x, 0, -h))

    return CadFigure(
        base="ch08_patch",
        parts=(
            Part("gnd", gnd, (0.55, 0.55, 0.55), 1.0),
            Part("substrate", substrate, (0.95, 0.85, 0.55), 0.70),
            Part("patch", patch, (0.25, 0.25, 0.25), 1.0),
            Part("via", via, (0.25, 0.25, 0.25), 1.0),
            Part("feed", feed, (0.85, 0.1, 0.1), 1.0),
        ),
        annotations=(
            Annotation("feed", (feed_x, 0.0, -h)),
            Annotation("GND", (0.45 * sub_xy, 0.45 * sub_xy, -h - 0.5 * gnd_t), color="tab:blue"),
        ),
    )


def make_cp_patch(
    *,
    sub_xy: float = 60.0,
    h: float = 1.6,
    gnd_t: float = 0.05,
    patch_w: float = 38.0,
    patch_l: float = 30.0,
    patch_t: float = 0.05,
    corner_cut: float = 5.0,
    feed_x: float = -6.0,
) -> CadFigure:
    base = make_patch(
        sub_xy=sub_xy,
        h=h,
        gnd_t=gnd_t,
        patch_w=patch_w,
        patch_l=patch_l,
        patch_t=patch_t,
        feed_x=feed_x,
    )

    # Replace the patch with a corner-cut variant (mounted on z=0)
    wp = cq.Workplane("XY").rect(patch_l, patch_w).extrude(patch_t)
    for sx in (-1, 1):
        for sy in (-1, 1):
            wp = wp.cut(
                cq.Workplane("XY")
                .box(corner_cut, corner_cut, 10 * patch_t, centered=(True, True, False))
                .translate(
                    (
                        sx * 0.5 * (patch_l - corner_cut),
                        sy * 0.5 * (patch_w - corner_cut),
                        0,
                    )
                )
            )

    parts = tuple(
        Part(p.name, (wp if p.name == "patch" else p.wp), p.color, p.alpha) for p in base.parts
    )
    return CadFigure(
        base="ch08_cp_patch",
        parts=parts,
        annotations=base.annotations + (Annotation("corner cut", (0.5 * patch_l, 0.5 * patch_w, 0.0)),),
    )


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[2]
    figs_dir = repo_dir / "figs" / "ch08"
    cad_dir = repo_dir / "cad" / "ch08"

    figures = (
        # The main goal of structure figures is quick shape recognition, so excessively large base solids are avoided.
        make_monopole(gnd_xy=32.0, gnd_t=1.0),
        make_patch(sub_xy=50.0),
        make_cp_patch(sub_xy=50.0),
    )

    for f in figures:
        out_png = str(figs_dir / f"{f.base}_structure_iso.png")
        out_step = str(cad_dir / f"{f.base}.step")
        # Use a consistent CAD-like isometric viewpoint across structure figures to avoid viewpoint drift within a chapter.
        render_isometric(f, out_png=out_png)
        export_step(f.parts, out_step=out_step)

    print("generated: figs/ch08/*_structure_iso.png")
    print("generated: cad/ch08/*.step")


if __name__ == "__main__":
    main()
