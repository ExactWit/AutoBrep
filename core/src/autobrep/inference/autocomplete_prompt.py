"""Build BOGEOM autocomplete prompts with real FSQ shape codes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import torch

from autobrep.data.serialize import deserialize_array
from autobrep.data.token_mapping import MMTokenIndex
from autobrep.utils import quantize_pos


@dataclass
class AutocompleteVocab:
    bit: int = 10
    max_face: int = 200
    surf_codebook_size: int = 1024
    edge_codebook_size: int = 1024

    def __post_init__(self) -> None:
        self.flag_pad = len(MMTokenIndex.__members__)
        self.id_pad = self.max_face
        self.pos_pad = 2**self.bit
        self.face_z_pad = self.pos_pad + self.id_pad + self.flag_pad
        self.edge_code_offset = self.face_z_pad + self.surf_codebook_size


COMPLEXITY_TO_TOKEN = {
    "easy": MMTokenIndex.GEN_EASY.value,
    "medium": MMTokenIndex.GEN_MID.value,
    "hard": MMTokenIndex.GEN_HARD.value,
    "random": MMTokenIndex.GEN_UNCOND.value,
    "uncond": MMTokenIndex.GEN_UNCOND.value,
}


def _as_array(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def encode_face_fsq(
    surface_fsq: torch.nn.Module,
    face_ncs: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """face_ncs (F,32,32,3) → int codes (F,4) in [0, codebook)."""
    if face_ncs.size == 0:
        return np.zeros((0, 4), dtype=np.int64)
    x = torch.from_numpy(face_ncs.astype(np.float32)).to(device)
    with torch.inference_mode():
        _, ids = surface_fsq.encode(x.permute(0, 3, 1, 2))
        ids = ids.flatten(-2, -1).long().cpu().numpy()
    return ids


def encode_edge_fsq(
    edge_fsq: torch.nn.Module,
    edge_ncs: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """edge_ncs (E,32,3) → int codes (E,2)."""
    if edge_ncs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    x = torch.from_numpy(edge_ncs.astype(np.float32)).to(device)
    with torch.inference_mode():
        _, ids = edge_fsq.encode(x.permute(0, 2, 1))
        ids = ids.long().cpu().numpy()
    return ids


def build_geom_tokens_from_arrays(
    face_pos: np.ndarray,
    face_ncs: np.ndarray,
    edge_pos: np.ndarray,
    edge_ncs: np.ndarray,
    face_edge_adj: np.ndarray,
    user_face_indices: Sequence[int],
    *,
    surface_fsq: torch.nn.Module,
    edge_fsq: torch.nn.Module,
    device: torch.device,
    vocab: Optional[AutocompleteVocab] = None,
) -> Tuple[List[int], List[int]]:
    """
    Build BOGEOM…EOGEOM token list with real FSQ codes for selected faces.

    Returns:
        geom_tokens, remapped_face_ids (local 0..n-1 in emission order)
    """
    vocab = vocab or AutocompleteVocab()
    user_face_indices = [int(i) for i in user_face_indices]
    if len(user_face_indices) < 1:
        raise ValueError("autocomplete requires at least one condition face")

    face_pos = _as_array(face_pos).reshape(-1, 6).astype(np.float32)
    edge_pos = _as_array(edge_pos).reshape(-1, 6).astype(np.float32)
    face_ncs = _as_array(face_ncs).astype(np.float32)
    edge_ncs = _as_array(edge_ncs).astype(np.float32)
    face_edge_adj = _as_array(face_edge_adj).astype(bool)

    face_pos_bit, edge_pos_bit = quantize_pos(face_pos, edge_pos, vocab.bit)

    # Sort selected faces by quantized bbox (same as geom_tokenization)
    user_face_pos_bit = face_pos_bit[user_face_indices]
    xyz_order = np.lexsort(
        tuple(user_face_pos_bit[:, i] for i in range(5, -1, -1))
    ).tolist()
    faces_sorted_orig = [user_face_indices[x] for x in xyz_order]
    # Local Face IDs 0..n-1 in emission order
    orig_to_local = {orig: loc for loc, orig in enumerate(faces_sorted_orig)}
    faces_sorted = list(range(len(faces_sorted_orig)))

    # Subset UV / pos for FSQ
    face_ncs_sel = face_ncs[faces_sorted_orig]
    face_codes = encode_face_fsq(surface_fsq, face_ncs_sel, device)  # (n,4)

    face_graph = nx.Graph()
    face_graph.add_nodes_from(range(len(face_pos)))
    for col in face_edge_adj.T:
        hits = np.where(col)[0]
        if len(hits) == 2:
            face_graph.add_edge(int(hits[0]), int(hits[1]))

    selected = set(faces_sorted_orig)
    data_seq: List[int] = [
        MMTokenIndex.BOGEOM.value,
        MMTokenIndex.BOL.value,
    ]

    # Face header matches post-map_func training form: BOF + bbox×6 + FSQ×4
    # (Face ID after BOF is stripped before the model sees the sequence).
    for loc_idx, orig_face in enumerate(faces_sorted_orig):
        data_seq += (
            [MMTokenIndex.BOF.value]
            + (face_pos_bit[orig_face] + vocab.id_pad + vocab.flag_pad).tolist()
            + (face_codes[loc_idx] + vocab.face_z_pad).tolist()
        )

        prev_face_edges: List[int] = []
        prev_face_ids: List[int] = []
        for prev_orig in faces_sorted_orig[:loc_idx]:
            connected = np.where(face_edge_adj[[orig_face, prev_orig]].sum(0) == 2)[0]
            if len(connected) > 0:
                prev_face_edges += list(map(int, connected))
                prev_face_ids += [orig_to_local[prev_orig]] * len(connected)

        edges_on_face = np.where(face_edge_adj[orig_face] == 1)[0]
        neighbor_faces = [
            x for x in face_graph.neighbors(orig_face) if x in selected
        ]
        co_edges = []
        if neighbor_faces:
            co_edges = edges_on_face[
                np.where(
                    face_edge_adj[:, edges_on_face][np.array(neighbor_faces)].sum(0)
                    == 1
                )[0]
            ]
        dangling_edges = list(set(map(int, edges_on_face)) - set(map(int, co_edges)))

        all_edges = dangling_edges + prev_face_edges
        all_face_ids = [loc_idx] * len(dangling_edges) + prev_face_ids

        if all_edges:
            xyz_e = np.lexsort(
                tuple(edge_pos_bit[all_edges][:, i] for i in range(5, -1, -1))
            ).tolist()
            all_edges = [all_edges[x] for x in xyz_e]
            all_face_ids = [all_face_ids[x] for x in xyz_e]

            edge_ncs_batch = edge_ncs[all_edges]
            edge_codes = encode_edge_fsq(edge_fsq, edge_ncs_batch, device)

            for j, (edge_index, connect_local) in enumerate(
                zip(all_edges, all_face_ids)
            ):
                connect_tok = (
                    MMTokenIndex.DUMMYID.value
                    if connect_local == loc_idx
                    else (connect_local + vocab.flag_pad)
                )
                data_seq += (
                    [connect_tok]
                    + (edge_pos_bit[edge_index] + vocab.id_pad + vocab.flag_pad).tolist()
                    + (edge_codes[j] + vocab.edge_code_offset).tolist()
                )

        data_seq += [MMTokenIndex.EOF.value]

    data_seq += [MMTokenIndex.EOL.value, MMTokenIndex.EOGEOM.value]
    return data_seq, faces_sorted


def build_autocomplete_prompt(
    geom_tokens: Sequence[int],
    *,
    complexity: str = "medium",
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """BOS BOM GEN EOM + BOGEOM…EOGEOM + BOC  → (B, L)."""
    gen = COMPLEXITY_TO_TOKEN.get(complexity, MMTokenIndex.GEN_MID.value)
    prefix = [
        MMTokenIndex.BOS.value,
        MMTokenIndex.BOM.value,
        gen,
        MMTokenIndex.EOM.value,
    ]
    suffix = [MMTokenIndex.BOC.value]
    seq = list(prefix) + list(geom_tokens) + suffix
    t = torch.tensor([seq] * batch_size, dtype=torch.long)
    if device is not None:
        t = t.to(device)
    return t


def condition_to_jsonable(
    face_pos: np.ndarray,
    face_ncs: np.ndarray,
    edge_pos: np.ndarray,
    edge_ncs: np.ndarray,
    face_edge_adj: np.ndarray,
    user_face_indices: Sequence[int],
) -> Dict[str, Any]:
    """Export selected faces/edges as autocomplete JSON."""
    user_face_indices = [int(i) for i in user_face_indices]
    selected = set(user_face_indices)
    faces = []
    for new_id, fi in enumerate(user_face_indices):
        faces.append(
            {
                "id": new_id,
                "bbox": face_pos[fi].reshape(6).astype(float).tolist(),
                "uv": face_ncs[fi].astype(float).tolist(),
            }
        )
    # edges with both ends in selection, or dangling on a selected face
    edges = []
    for e in range(face_edge_adj.shape[1]):
        faces_hit = np.where(face_edge_adj[:, e])[0].tolist()
        if not faces_hit:
            continue
        if not any(f in selected for f in faces_hit):
            continue
        if len(faces_hit) == 2 and faces_hit[0] in selected and faces_hit[1] in selected:
            a = user_face_indices.index(faces_hit[0])
            b = user_face_indices.index(faces_hit[1])
        elif len(faces_hit) == 2:
            inside = [f for f in faces_hit if f in selected]
            if len(inside) != 1:
                continue
            a = user_face_indices.index(inside[0])
            b = None
        elif len(faces_hit) == 1 and faces_hit[0] in selected:
            a = user_face_indices.index(faces_hit[0])
            b = None
        else:
            continue
        edges.append(
            {
                "face_a": a,
                "face_b": b,
                "bbox": edge_pos[e].reshape(6).astype(float).tolist(),
                "uv": edge_ncs[e].astype(float).tolist(),
            }
        )
    return {"faces": faces, "edges": edges}


def load_condition_from_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    faces = data["faces"]
    edges = data.get("edges") or []
    n = len(faces)
    face_pos = np.stack([_as_array(f["bbox"]).reshape(6) for f in faces], axis=0)
    face_ncs = np.stack([_as_array(f["uv"]) for f in faces], axis=0)
    if face_ncs.ndim != 4 or face_ncs.shape[1:] != (32, 32, 3):
        raise ValueError(f"face uv must be (32,32,3), got {face_ncs.shape}")

    if not edges:
        # bbox-only: fabricate empty edge set (faces with no edges — weak condition)
        edge_pos = np.zeros((0, 6), dtype=np.float32)
        edge_ncs = np.zeros((0, 32, 3), dtype=np.float32)
        face_edge_adj = np.zeros((n, 0), dtype=bool)
    else:
        edge_pos = np.stack([_as_array(e["bbox"]).reshape(6) for e in edges], axis=0)
        edge_ncs = np.stack([_as_array(e["uv"]) for e in edges], axis=0)
        face_edge_adj = np.zeros((n, len(edges)), dtype=bool)
        for ei, e in enumerate(edges):
            a = int(e["face_a"])
            face_edge_adj[a, ei] = True
            b = e.get("face_b", None)
            if b is not None:
                face_edge_adj[int(b), ei] = True

    return {
        "face_pos": face_pos.astype(np.float32),
        "face_ncs": face_ncs.astype(np.float32),
        "edge_pos": edge_pos.astype(np.float32),
        "edge_ncs": edge_ncs.astype(np.float32),
        "face_edge_adj": face_edge_adj,
        "user_face_indices": list(range(n)),
        "meta": {"source": str(path), "num_faces": n, "num_edges": len(edges)},
    }


def find_abc_row_by_stem(
    data_root: Union[str, Path],
    stem: str,
    *,
    split: str = "train",
    max_files: int = 0,
) -> Dict[str, Any]:
    """Scan parquet shards for stem (exact match). max_files=0 → all files."""
    import pyarrow.parquet as pq

    root = Path(data_root)
    split_dir = root / split
    if not split_dir.is_dir():
        if (root.parent / split).is_dir():
            split_dir = root.parent / split
        else:
            raise FileNotFoundError(f"missing split dir: {split_dir}")

    cols = [
        "stem",
        "face_points_normalized",
        "edge_points_normalized",
        "face_bbox_world",
        "edge_bbox_world",
        "face_edge_incidence",
        "constraint_faces",
        "num_faces",
        "num_faces_after_splitting",
    ]
    files = sorted(split_dir.glob("*.parquet"))
    if max_files and max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no parquet under {split_dir}")

    for f in files:
        try:
            t = pq.read_table(f, columns=["stem"])
            stems = t.column(0).to_pylist()
            if stem not in stems:
                continue
            idx = stems.index(stem)
            # Only request columns that exist
            schema_names = set(pq.read_schema(f).names)
            use_cols = [c for c in cols if c in schema_names]
            full = pq.read_table(f, columns=use_cols)
            pdf = full.slice(idx, 1).to_pandas()
            return {c: pdf.iloc[0][c] for c in pdf.columns}
        except Exception:
            continue
    raise ValueError(f"stem not found under {split_dir}: {stem}")


def load_condition_from_abc_row(
    row: Dict[str, Any],
    *,
    face_ids: Optional[Sequence[int]] = None,
    num_faces: Optional[int] = None,
    mode: str = "random",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Unpickle ABC row and pick condition faces."""
    rng = rng or np.random.default_rng(0)
    face_pos = deserialize_array(row["face_bbox_world"]).astype(np.float32)
    edge_pos = deserialize_array(row["edge_bbox_world"]).astype(np.float32)
    face_ncs = deserialize_array(row["face_points_normalized"]).astype(np.float32)
    edge_ncs = deserialize_array(row["edge_points_normalized"]).astype(np.float32)
    face_edge_adj = deserialize_array(row["face_edge_incidence"]).astype(bool)
    constraint = None
    if "constraint_faces" in row and row["constraint_faces"] is not None:
        try:
            constraint = deserialize_array(row["constraint_faces"]).astype(bool)
        except Exception:
            constraint = None

    n_faces = face_pos.shape[0]
    if face_ids is not None:
        user = [int(i) for i in face_ids]
    elif mode == "constraint" and constraint is not None and constraint.any():
        user = np.where(constraint)[0].tolist()
        if len(user) > 20:
            user = rng.choice(user, size=20, replace=False).tolist()
    else:
        k = num_faces if num_faces is not None else min(4, max(2, n_faces // 4))
        k = min(k, n_faces)
        # Prefer two adjacent faces when possible
        if k >= 2 and face_edge_adj.size:
            # find an edge connecting two faces
            for e in range(face_edge_adj.shape[1]):
                hits = np.where(face_edge_adj[:, e])[0]
                if len(hits) == 2:
                    user = hits.tolist()
                    remaining = [i for i in range(n_faces) if i not in user]
                    if k > 2 and remaining:
                        extra = rng.choice(
                            remaining, size=min(k - 2, len(remaining)), replace=False
                        ).tolist()
                        user = user + extra
                    break
            else:
                user = rng.choice(n_faces, size=k, replace=False).tolist()
        else:
            user = rng.choice(n_faces, size=k, replace=False).tolist()

    for i in user:
        if i < 0 or i >= n_faces:
            raise ValueError(f"face id {i} out of range [0,{n_faces})")

    return {
        "face_pos": face_pos,
        "face_ncs": face_ncs,
        "edge_pos": edge_pos,
        "edge_ncs": edge_ncs,
        "face_edge_adj": face_edge_adj,
        "user_face_indices": user,
        "meta": {
            "stem": row.get("stem"),
            "face_ids": user,
            "mode": mode,
            "num_faces_solid": int(n_faces),
        },
    }


def pick_adjacent_face_pair(face_edge_adj: np.ndarray) -> List[int]:
    for e in range(face_edge_adj.shape[1]):
        hits = np.where(face_edge_adj[:, e])[0]
        if len(hits) == 2:
            return hits.tolist()
    return [0, min(1, face_edge_adj.shape[0] - 1)]
