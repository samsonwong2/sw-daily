"""CSV mapping exporter and JSON metadata writer."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_IDENTITY_COLS = ["name", "code", "cluster", "selected", "reason", "n"]


def write_json(path: str | os.PathLike, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def export_mapping(
    all_codes: Iterable[str],
    clusters: pd.Series | None,
    selected: Iterable[str],
    reasons: dict[str, str],
    cluster_df: pd.DataFrame | None,
    code_name_map: dict[str, str],
    universe: pd.DataFrame | None = None,
    out_csv: str | Path | None = None,
    selected_csv: str | Path | None = None,
) -> Path:
    """Write the mapping CSV: identity columns plus universe extras."""
    if out_csv is None:
        raise ValueError("out_csv is required")

    all_codes = list(dict.fromkeys(list(all_codes)))
    selected_set = set(selected)
    df = pd.DataFrame({"code": all_codes})
    if clusters is not None:
        df["cluster"] = df["code"].map(clusters)
    else:
        df["cluster"] = np.nan
    df["selected"] = df["code"].isin(selected_set)
    df["reason"] = df["code"].map(lambda c: reasons.get(c, ""))
    df["name"] = df["code"].map(lambda c: code_name_map.get(str(c).strip(), ""))
    df = df[["name", "code", "cluster", "selected", "reason"]]
    if cluster_df is not None and not cluster_df.empty and "n" in cluster_df.columns:
        cluster_n = cluster_df.set_index("cluster")[["n"]]
        df = df.join(cluster_n, on="cluster")
    else:
        df["n"] = np.nan

    extra_cols: list[str] = []
    if universe is not None:
        extra_cols = [c for c in universe.columns if c not in ("code", "name")]
        if extra_cols:
            extras = universe[["code", *extra_cols]].copy()
            df = df.merge(extras, on="code", how="left")
            for c in extra_cols:
                df[c] = df[c].fillna("")

    df = df[_IDENTITY_COLS + extra_cols]
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Wrote mapping to", out_path)
    if selected_csv:
        selected_path = Path(selected_csv)
        df[df["selected"]].to_csv(selected_path, index=False, encoding="utf-8-sig")
        print("Wrote selected-only mapping to", selected_path)
    return out_path
