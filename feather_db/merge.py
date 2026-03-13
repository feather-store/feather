"""
DB Merge — Feather DB v0.6.0
==============================
Merge two Feather DB files into one.

Usage
-----
    from feather_db.merge import merge

    stats = merge(
        target_db   = db,                   # open DB instance
        source_path = "other.feather",      # file to merge from
        dim         = 3072,
        conflict_policy = "keep_target",    # keep_target | keep_source | merge
        modalities  = ["text", "visual"],   # which modalities to merge
    )
    print(stats)  # {merged: N, skipped: N, conflicts: N}
"""

from __future__ import annotations

from typing import Optional


def merge(
    target_db,
    source_path: str,
    dim: Optional[int] = None,
    conflict_policy: str = "keep_target",
    modalities: Optional[list[str]] = None,
) -> dict:
    """
    Merge nodes from `source_path` into `target_db`.

    Parameters
    ----------
    target_db        Open feather_db.DB instance (the destination)
    source_path      Path to the source .feather file
    dim              Vector dimension for source DB (auto-detected if None)
    conflict_policy  What to do when the same ID exists in both DBs:
                       'keep_target'  — skip the source node (default)
                       'keep_source'  — overwrite with source metadata + vector
                       'merge'        — union attributes, keep higher importance,
                                        add edges, keep higher confidence
    modalities       Which modalities to merge (default: all in source)

    Returns
    -------
    dict with keys: merged, skipped, conflicts, edges_added
    """
    try:
        import feather_db as _fdb
    except ImportError:
        raise ImportError("feather_db must be installed")

    target_dim = dim or target_db.dim("text") or 768
    source_db  = _fdb.DB.open(source_path, dim=target_dim)

    stats = {"merged": 0, "skipped": 0, "conflicts": 0, "edges_added": 0}

    # Determine modalities to process
    if modalities is None:
        # Try common modalities — skip silently if source has no vectors there
        modalities = ["text", "visual", "audio", "video"]

    for mod in modalities:
        source_ids = source_db.get_all_ids(mod)
        if not source_ids:
            continue

        for sid in source_ids:
            src_vec  = source_db.get_vector(sid, mod)
            src_meta = source_db.get_metadata(sid)
            if src_meta is None or len(src_vec) == 0:
                continue

            src_vec_list = list(src_vec)
            existing = target_db.get_metadata(sid)

            if existing is None:
                # Node doesn't exist in target — insert
                import numpy as np
                v = np.array(src_vec_list, dtype=np.float32)
                target_db.add(id=sid, vec=v, meta=src_meta, modality=mod)
                stats["merged"] += 1
            else:
                # Conflict: node exists in both
                stats["conflicts"] += 1
                if conflict_policy == "keep_target":
                    stats["skipped"] += 1
                    continue
                elif conflict_policy == "keep_source":
                    import numpy as np
                    v = np.array(src_vec_list, dtype=np.float32)
                    target_db.add(id=sid, vec=v, meta=src_meta, modality=mod)
                    stats["merged"] += 1
                elif conflict_policy == "merge":
                    import numpy as np
                    # Merge metadata
                    merged_meta = existing
                    # Attributes: source wins on key collision
                    for k, v in src_meta.attributes.items():
                        merged_meta.set_attribute(k, v)
                    # Keep higher importance + confidence
                    if src_meta.importance > merged_meta.importance:
                        merged_meta.importance = src_meta.importance
                    if src_meta.confidence > merged_meta.confidence:
                        merged_meta.confidence = src_meta.confidence
                    target_db.update_metadata(sid, merged_meta)
                    stats["merged"] += 1

    # Merge edges from all source nodes (even if they were skipped by vector policy)
    all_source_ids: set[int] = set()
    for mod in modalities:
        all_source_ids.update(source_db.get_all_ids(mod))

    for sid in all_source_ids:
        src_meta = source_db.get_metadata(sid)
        if src_meta is None:
            continue
        if target_db.get_metadata(sid) is None:
            continue  # target doesn't have this node
        for edge in src_meta.edges:
            if target_db.get_metadata(edge.target_id) is not None:
                try:
                    target_db.link(from_id=sid, to_id=edge.target_id,
                                   rel_type=edge.rel_type, weight=edge.weight)
                    stats["edges_added"] += 1
                except Exception:
                    pass

    return stats
