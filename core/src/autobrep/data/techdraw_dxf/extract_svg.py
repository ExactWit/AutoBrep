"""Geometric SVG → PrimIR (same schema as DXF)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from autobrep.data.techdraw_dxf.schema import DxfIR, PrimIR

_CMD_RE = re.compile(
    r"([MmLlHhVvCcSsQqTtAaZz])|([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)


def _tokenizeize_path(d: str, *, curve_samples: int = 4) -> list[list[float]]:
    """Approximate SVG path `d` as a polyline in absolute coords."""
    if not d:
        return []
    tokens: list[str] = []
    for m in _CMD_RE.finditer(d.replace(",", " ")):
        tokens.append(m.group(1) or m.group(2))
    pts: list[list[float]] = []
    i = 0
    cmd = "L"
    cx = cy = 0.0
    sx = sy = 0.0  # subpath start

    def _num() -> float:
        nonlocal i
        v = float(tokens[i])
        i += 1
        return v

    while i < len(tokens):
        t = tokens[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in "Zz":
                if pts and (pts[-1][0] != sx or pts[-1][1] != sy):
                    pts.append([sx, sy])
                cx, cy = sx, sy
                continue
        # implicit command repeat
        try:
            if cmd in "Mm":
                x, y = _num(), _num()
                if cmd == "m":
                    x, y = cx + x, cy + y
                cx, cy = x, y
                sx, sy = x, y
                pts.append([x, y])
                cmd = "l" if cmd == "m" else "L"
            elif cmd in "Ll":
                x, y = _num(), _num()
                if cmd == "l":
                    x, y = cx + x, cy + y
                cx, cy = x, y
                pts.append([x, y])
            elif cmd in "Hh":
                x = _num()
                if cmd == "h":
                    x = cx + x
                cx = x
                pts.append([cx, cy])
            elif cmd in "Vv":
                y = _num()
                if cmd == "v":
                    y = cy + y
                cy = y
                pts.append([cx, cy])
            elif cmd in "Cc":
                coords = [_num() for _ in range(6)]
                if cmd == "c":
                    coords = [
                        cx + coords[0],
                        cy + coords[1],
                        cx + coords[2],
                        cy + coords[3],
                        cx + coords[4],
                        cy + coords[5],
                    ]
                x0, y0 = cx, cy
                x1, y1, x2, y2, x3, y3 = coords
                for s in range(1, curve_samples + 1):
                    t = s / curve_samples
                    mt = 1 - t
                    x = (
                        mt**3 * x0
                        + 3 * mt**2 * t * x1
                        + 3 * mt * t**2 * x2
                        + t**3 * x3
                    )
                    y = (
                        mt**3 * y0
                        + 3 * mt**2 * t * y1
                        + 3 * mt * t**2 * y2
                        + t**3 * y3
                    )
                    pts.append([x, y])
                cx, cy = x3, y3
            else:
                # skip unsupported arc/quadratic: consume one number if possible
                if i < len(tokens) and not tokens[i].isalpha():
                    i += 1
                else:
                    break
        except (IndexError, ValueError):
            break
    return pts


def _linetype_from_style(style: str, stroke_dasharray: str | None) -> str:
    blob = f"{style or ''} {stroke_dasharray or ''}".lower()
    if "dash" in blob or "stroke-dasharray" in blob:
        # treat any dashed stroke as hidden (common in techdraw exports)
        if "dasharray:none" in blob.replace(" ", ""):
            return "solid"
        return "hidden"
    return "solid"


def extract_svg_primitives(path: Path | str) -> DxfIR:
    """Parse SVG paths into line/lwpolyline PrimIR (geometric, no raster)."""
    path = Path(path)
    tree = ET.parse(str(path))
    root = tree.getroot()
    prims: list[PrimIR] = []

    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if tag != "path":
            continue
        d = elem.get("d") or ""
        style = elem.get("style") or ""
        dash = elem.get("stroke-dasharray")
        # also pull dash from style
        if "stroke-dasharray" in style and dash is None:
            m = re.search(r"stroke-dasharray\s*:\s*([^;]+)", style)
            if m:
                dash = m.group(1).strip()
        linetype = _linetype_from_style(style, dash)
        pts = _polygonize_path(d)
        if len(pts) < 2:
            continue
        # collapse to segments / short polylines
        if len(pts) == 2:
            prims.append(
                PrimIR(
                    type="line",
                    linetype=linetype,
                    params={"start": pts[0][:2], "end": pts[1][:2]},
                )
            )
        else:
            # subsample long polylines
            if len(pts) > 32:
                idx = np.linspace(0, len(pts) - 1, num=32).astype(int)
                pts = [pts[i] for i in idx]
            prims.append(
                PrimIR(
                    type="lwpolyline",
                    linetype=linetype,
                    params={"points": pts, "closed": False},
                )
            )

    points: list[list[float]] = []
    for prim in prims:
        if prim.type == "line":
            points.extend([prim.params["start"][:2], prim.params["end"][:2]])
        else:
            points.extend([p[:2] for p in prim.params.get("points") or []])
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
