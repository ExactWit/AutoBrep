"""B-Rep STEP post-process: analytic surface replacement + ShapeFix."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

_LOG = logging.getLogger(__name__)

# GeomAbs surface types (avoid import-time OCC for unit tests that skip)
_ANALYTIC_TYPES = None


def _occ_imports():
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
    from OCC.Core.BRepTools import BRepTools_ReShape
    from OCC.Core.Geom import Geom_CylindricalSurface, Geom_Plane, Geom_SphericalSurface
    from OCC.Core.GeomAbs import (
        GeomAbs_BezierSurface,
        GeomAbs_BSplineSurface,
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_Plane,
        GeomAbs_Sphere,
        GeomAbs_Torus,
    )
    from OCC.Core.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.ShapeFix import ShapeFix_Shape
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    return {
        "BRepAdaptor_Surface": BRepAdaptor_Surface,
        "BRepBuilderAPI_MakeFace": BRepBuilderAPI_MakeFace,
        "BRepBuilderAPI_Sewing": BRepBuilderAPI_Sewing,
        "BRepTools_ReShape": BRepTools_ReShape,
        "Geom_Plane": Geom_Plane,
        "Geom_CylindricalSurface": Geom_CylindricalSurface,
        "Geom_SphericalSurface": Geom_SphericalSurface,
        "GeomAbs_Plane": GeomAbs_Plane,
        "GeomAbs_Cylinder": GeomAbs_Cylinder,
        "GeomAbs_Cone": GeomAbs_Cone,
        "GeomAbs_Sphere": GeomAbs_Sphere,
        "GeomAbs_BSplineSurface": GeomAbs_BSplineSurface,
        "GeomAbs_BezierSurface": GeomAbs_BezierSurface,
        "GeomAbs_Torus": GeomAbs_Torus,
        "gp_Ax3": gp_Ax3,
        "gp_Dir": gp_Dir,
        "gp_Pln": gp_Pln,
        "gp_Pnt": gp_Pnt,
        "IFSelect_RetDone": IFSelect_RetDone,
        "ShapeFix_Shape": ShapeFix_Shape,
        "STEPControl_AsIs": STEPControl_AsIs,
        "STEPControl_Reader": STEPControl_Reader,
        "STEPControl_Writer": STEPControl_Writer,
        "TopAbs_FACE": TopAbs_FACE,
        "TopExp_Explorer": TopExp_Explorer,
        "topods": topods,
    }


def read_step(path: Path | str):
    occ = _occ_imports()
    reader = occ["STEPControl_Reader"]()
    status = reader.ReadFile(str(path))
    if status != occ["IFSelect_RetDone"]:
        raise RuntimeError(f"STEP read failed: {path}")
    reader.TransferRoots()
    return reader.OneShape()


def write_step(shape, path: Path | str) -> None:
    occ = _occ_imports()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = occ["STEPControl_Writer"]()
    writer.Transfer(shape, occ["STEPControl_AsIs"])
    status = writer.Write(str(path))
    if status != occ["IFSelect_RetDone"]:
        raise RuntimeError(f"STEP write failed: {path}")


def _sample_face_points(face, *, n_u: int = 8, n_v: int = 8) -> np.ndarray:
    occ = _occ_imports()
    adaptor = occ["BRepAdaptor_Surface"](face, True)
    u0, u1 = adaptor.FirstUParameter(), adaptor.LastUParameter()
    v0, v1 = adaptor.FirstVParameter(), adaptor.LastVParameter()
    pts = []
    for iu in range(n_u):
        u = u0 + (u1 - u0) * (iu + 0.5) / n_u
        for iv in range(n_v):
            v = v0 + (v1 - v0) * (iv + 0.5) / n_v
            p = adaptor.Value(u, v)
            pts.append([p.X(), p.Y(), p.Z()])
    return np.asarray(pts, dtype=np.float64)


def _fit_plane(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    if len(pts) < 3:
        return None
    c = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
    normal = vh[-1]
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-12:
        return None
    normal = normal / nrm
    dist = np.abs((pts - c) @ normal)
    return c, normal, float(dist.max())


def _fit_sphere(pts: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    if len(pts) < 4:
        return None
    # Algebraic fit: ||x||^2 = 2 c·x + k
    A = np.concatenate([2 * pts, np.ones((len(pts), 1))], axis=1)
    b = (pts ** 2).sum(axis=1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    center = sol[:3]
    r2 = float(sol[3] + (center ** 2).sum())
    if r2 <= 0:
        return None
    radius = float(np.sqrt(r2))
    err = float(np.abs(np.linalg.norm(pts - center, axis=1) - radius).max())
    return center, radius, err


def _fit_cylinder(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Rough cylinder: PCA axis + mean radial distance."""
    if len(pts) < 6:
        return None
    c = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
    axis = vh[0]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    # radial vectors
    d = pts - c
    axial = (d @ axis)[:, None] * axis[None, :]
    radial = d - axial
    radii = np.linalg.norm(radial, axis=1)
    r = float(np.median(radii))
    if r < 1e-8:
        return None
    err = float(np.abs(radii - r).max())
    return c, axis, r, err


def _make_analytic_geom(face, *, tol: float):
    """Return Geom_Surface or None if freeform should be kept."""
    occ = _occ_imports()
    adaptor = occ["BRepAdaptor_Surface"](face, True)
    st = adaptor.GetType()
    already = {
        occ["GeomAbs_Plane"],
        occ["GeomAbs_Cylinder"],
        occ["GeomAbs_Cone"],
        occ["GeomAbs_Sphere"],
        occ["GeomAbs_Torus"],
    }
    if st in already:
        return None  # keep as-is (no reshape needed)

    pts = _sample_face_points(face)
    # Prefer plane → cylinder → sphere by error
    candidates = []
    plane = _fit_plane(pts)
    if plane is not None and plane[2] <= tol:
        c, n, err = plane
        pln = occ["gp_Pln"](
            occ["gp_Pnt"](float(c[0]), float(c[1]), float(c[2])),
            occ["gp_Dir"](float(n[0]), float(n[1]), float(n[2])),
        )
        candidates.append((err, occ["Geom_Plane"](pln), "plane"))
    cyl = _fit_cylinder(pts)
    if cyl is not None and cyl[3] <= tol * 2:
        c, axis, r, err = cyl
        # Build Ax3: location, Z=axis, X arbitrary orthogonal
        z = axis / (np.linalg.norm(axis) + 1e-12)
        tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(tmp, z)
        x = x / (np.linalg.norm(x) + 1e-12)
        ax3 = occ["gp_Ax3"](
            occ["gp_Pnt"](float(c[0]), float(c[1]), float(c[2])),
            occ["gp_Dir"](float(z[0]), float(z[1]), float(z[2])),
            occ["gp_Dir"](float(x[0]), float(x[1]), float(x[2])),
        )
        candidates.append((err, occ["Geom_CylindricalSurface"](ax3, float(r)), "cylinder"))
    sph = _fit_sphere(pts)
    if sph is not None and sph[2] <= tol * 2:
        c, r, err = sph
        ax3 = occ["gp_Ax3"](occ["gp_Pnt"](float(c[0]), float(c[1]), float(c[2])))
        candidates.append((err, occ["Geom_SphericalSurface"](ax3, float(r)), "sphere"))

    if not candidates:
        # Freeform smoothing: OCC GeomPlate / Approx not always available — skip (documented).
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def replace_analytic_surfaces(shape, *, tol: float = 1e-3) -> tuple[Any, dict[str, int]]:
    """
    Replace BSpline/Bezier faces with fitted plane/cylinder/sphere when RMSE ≤ tol.

    Uses ``BRep_Builder.UpdateFace`` on a copied face so wires stay attached.
    """
    occ = _occ_imports()
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy

    reshape = occ["BRepTools_ReShape"]()
    stats = {"faces": 0, "replaced": 0, "kept_freeform": 0, "already_analytic": 0, "failed": 0}
    exp = occ["TopExp_Explorer"](shape, occ["TopAbs_FACE"])
    while exp.More():
        face = occ["topods"].Face(exp.Current())
        stats["faces"] += 1
        adaptor = occ["BRepAdaptor_Surface"](face, True)
        st = adaptor.GetType()
        analytic = {
            occ["GeomAbs_Plane"],
            occ["GeomAbs_Cylinder"],
            occ["GeomAbs_Cone"],
            occ["GeomAbs_Sphere"],
            occ["GeomAbs_Torus"],
        }
        if st in analytic:
            stats["already_analytic"] += 1
            exp.Next()
            continue
        geom = _make_analytic_geom(face, tol=tol)
        if geom is None:
            stats["kept_freeform"] += 1
            exp.Next()
            continue
        try:
            copied = occ["topods"].Face(BRepBuilderAPI_Copy(face).Shape())
            BRep_Builder().UpdateFace(copied, geom, float(tol))
            reshape.Replace(face, copied)
            stats["replaced"] += 1
        except Exception:  # noqa: BLE001
            stats["failed"] += 1
        exp.Next()
    return reshape.Apply(shape), stats


def shape_fix_and_sew(
    shape,
    *,
    sew_tolerance: float = 0.005,
    do_sew: bool = True,
) -> Any:
    occ = _occ_imports()
    fixer = occ["ShapeFix_Shape"](shape)
    fixer.Perform()
    fixed = fixer.Shape()
    if not do_sew:
        return fixed
    sewing = occ["BRepBuilderAPI_Sewing"](float(sew_tolerance))
    exp = occ["TopExp_Explorer"](fixed, occ["TopAbs_FACE"])
    n_faces = 0
    while exp.More():
        sewing.Add(exp.Current())
        n_faces += 1
        exp.Next()
    if n_faces == 0:
        return fixed
    sewing.Perform()
    return sewing.SewedShape()


def postprocess_shape(
    shape,
    *,
    analytic: bool = True,
    analytic_tol: float = 1e-3,
    sew_tolerance: float = 0.005,
    shape_fix: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """
    Full STEP post-process pipeline.

    Freeform fairing: not applied (OCC fairing APIs vary); documented skip.
    """
    info: dict[str, Any] = {"analytic": bool(analytic), "fairing": "skipped"}
    out = shape
    if analytic:
        out, stats = replace_analytic_surfaces(out, tol=analytic_tol)
        info["replace_stats"] = stats
    if shape_fix:
        out = shape_fix_and_sew(out, sew_tolerance=sew_tolerance, do_sew=True)
        info["shape_fix"] = True
        info["sew_tolerance"] = float(sew_tolerance)
    return out, info


def postprocess_step_file(
    src: Path | str,
    dst: Path | str | None = None,
    *,
    analytic: bool = True,
    analytic_tol: float = 1e-3,
    sew_tolerance: float = 0.005,
) -> dict[str, Any]:
    src = Path(src)
    dst = Path(dst) if dst is not None else src
    shape = read_step(src)
    out, info = postprocess_shape(
        shape,
        analytic=analytic,
        analytic_tol=analytic_tol,
        sew_tolerance=sew_tolerance,
    )
    write_step(out, dst)
    info.update({"src": str(src), "dst": str(dst), "ok": True})
    return info


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline analytic STEP post-process batch")
    p.add_argument("--pred-dir", type=str, required=True, help="Directory of *.step")
    p.add_argument("--out-dir", type=str, default="", help="Output dir (default: overwrite / sibling)")
    p.add_argument("--analytic", type=int, default=1)
    p.add_argument("--analytic-tol", type=float, default=1e-3)
    p.add_argument("--sew-tolerance", type=float, default=0.005)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pred_dir / "postprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = sorted(pred_dir.glob("*.step"))
    if args.limit > 0:
        steps = steps[: args.limit]
    report = []
    for i, sp in enumerate(steps):
        dst = out_dir / sp.name
        try:
            info = postprocess_step_file(
                sp,
                dst,
                analytic=bool(args.analytic),
                analytic_tol=float(args.analytic_tol),
                sew_tolerance=float(args.sew_tolerance),
            )
        except Exception as exc:  # noqa: BLE001
            info = {"src": str(sp), "ok": False, "error": str(exc)}
        report.append(info)
        if (i + 1) % 20 == 0 or i + 1 == len(steps):
            print(f"[postprocess] {i + 1}/{len(steps)} last_ok={info.get('ok')}", flush=True)
    (out_dir / "postprocess_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    n_ok = sum(1 for r in report if r.get("ok"))
    print(f"[postprocess] done ok={n_ok}/{len(report)} → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
