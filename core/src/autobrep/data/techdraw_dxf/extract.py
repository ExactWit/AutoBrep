"""Extract structured primitives from TechDraw DXF files."""

from __future__ import annotations

from pathlib import Path

import ezdxf
import numpy as np

from autobrep.data.techdraw_dxf.schema import DxfIR, PrimIR


def _linetype_category(name: str) -> str:
    upper = str(name).upper()
    if upper in {"", "BYLAYER", "BYBLOCK", "CONTINUOUS"}:
        return "solid"
    if "HIDDEN" in upper or "DASH" in upper:
        return "hidden"
    if "CENTER" in upper:
        return "center"
    return "other"


def _entity_linetype(entity) -> str:
    value = getattr(entity.dxf, "linetype", None)
    if value is None:
        return "solid"
    return _linetype_category(str(value))


def _xy(point) -> list[float]:
    return [float(point[0]), float(point[1])]


def _collect_points(prim: PrimIR) -> list[list[float]]:
    params = prim.params
    points: list[list[float]] = []
    if prim.type == "line":
        points.extend([params["start"][:2], params["end"][:2]])
    elif prim.type in {"arc", "circle"}:
        c = params["center"]
        r = float(params["radius"])
        points.extend([[c[0] - r, c[1] - r], [c[0] + r, c[1] + r]])
    elif prim.type == "ellipse":
        c = params["center"]
        maj = params["major_axis"]
        ratio = abs(float(params.get("ratio", 1.0)))
        points.append([c[0], c[1]])
        points.append([c[0] + maj[0], c[1] + maj[1]])
        points.append([c[0] - maj[1] * ratio, c[1] + maj[0] * ratio])
    elif prim.type == "spline":
        for p in params.get("control_points") or []:
            points.append(p[:2])
    elif prim.type == "lwpolyline":
        for p in params.get("points") or []:
            points.append(p[:2])
    return points


def extract_dxf_primitives(path: Path | str, *, drop_insert: bool = True) -> DxfIR:
    path = Path(path)
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    prims: list[PrimIR] = []

    for entity in msp:
        etype = entity.dxftype()
        linetype = _entity_linetype(entity)
        try:
            if etype == "LINE":
                prims.append(
                    PrimIR(
                        type="line",
                        linetype=linetype,
                        params={"start": _xy(entity.dxf.start), "end": _xy(entity.dxf.end)},
                    )
                )
            elif etype == "CIRCLE":
                prims.append(
                    PrimIR(
                        type="circle",
                        linetype=linetype,
                        params={
                            "center": _xy(entity.dxf.center),
                            "radius": float(entity.dxf.radius),
                            "start_angle": 0.0,
                            "end_angle": 2.0 * float(np.pi),
                        },
                    )
                )
            elif etype == "ARC":
                # ezdxf angles are degrees
                start = float(np.deg2rad(entity.dxf.start_angle))
                end = float(np.deg2rad(entity.dxf.end_angle))
                prims.append(
                    PrimIR(
                        type="arc",
                        linetype=linetype,
                        params={
                            "center": _xy(entity.dxf.center),
                            "radius": float(entity.dxf.radius),
                            "start_angle": start,
                            "end_angle": end,
                        },
                    )
                )
            elif etype == "ELLIPSE":
                prims.append(
                    PrimIR(
                        type="ellipse",
                        linetype=linetype,
                        params={
                            "center": _xy(entity.dxf.center),
                            "major_axis": _xy(entity.dxf.major_axis),
                            "ratio": float(entity.dxf.ratio),
                            "start_param": float(getattr(entity.dxf, "start_param", 0.0)),
                            "end_param": float(getattr(entity.dxf, "end_param", 2.0 * np.pi)),
                        },
                    )
                )
            elif etype == "SPLINE":
                cps = [[float(p[0]), float(p[1])] for p in entity.control_points]
                prims.append(
                    PrimIR(
                        type="spline",
                        linetype=linetype,
                        params={
                            "control_points": cps,
                            "degree": int(getattr(entity.dxf, "degree", 3)),
                            "closed": bool(getattr(entity, "closed", False)),
                        },
                    )
                )
            elif etype == "LWPOLYLINE":
                pts = [[float(p[0]), float(p[1])] for p in entity.get_points("xy")]
                prims.append(
                    PrimIR(
                        type="lwpolyline",
                        linetype=linetype,
                        params={"points": pts, "closed": bool(entity.closed)},
                    )
                )
            elif etype == "INSERT":
                if drop_insert:
                    continue
                prims.append(
                    PrimIR(
                        type="other",
                        linetype=linetype,
                        params={"center": _xy(entity.dxf.insert), "name": str(entity.dxf.name)},
                    )
                )
        except Exception:
            continue

    points: list[list[float]] = []
    for prim in prims:
        points.extend(_collect_points(prim))
    if points:
        arr = np.asarray(points, dtype=np.float32)
        bbox_min = arr.min(axis=0)
        bbox_max = arr.max(axis=0)
        if np.allclose(bbox_min, bbox_max):
            bbox_max = bbox_min + 1.0
    else:
        bbox_min = np.zeros(2, dtype=np.float32)
        bbox_max = np.ones(2, dtype=np.float32)

    return DxfIR(
        n_prims=len(prims),
        prims=prims,
        bbox_min=bbox_min.astype(np.float32),
        bbox_max=bbox_max.astype(np.float32),
    )
