"""Dendrogram plotting + CJK font detection."""
from __future__ import annotations

import os
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def detect_and_add_font() -> str | None:
    possible_files = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for p in possible_files:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                fp = fm.FontProperties(fname=p)
                return fp.get_name()
            except Exception:
                continue
    available = {f.name: f.fname for f in fm.fontManager.ttflist}
    for key in ["Noto", "WenQuan", "SimHei", "Microsoft YaHei", "DejaVu"]:
        for name, fname in available.items():
            if key.lower() in name.lower():
                try:
                    fm.fontManager.addfont(fname)
                    fp = fm.FontProperties(fname=fname)
                    return fp.get_name()
                except Exception:
                    continue
    return None


def plot_selected_dendrogram(
    Z,
    codes: Iterable[str],
    selected_set: set[str],
    code_name_map: dict[str, str],
    out_png: str,
    out_svg: str,
) -> None:
    codes = list(codes)
    labels = []
    for c in codes:
        if c in selected_set:
            name = code_name_map.get(str(c).strip())
            label = f"{name}_x" if name and name.strip() != "" else f"{c}_x"
        else:
            label = str(c)
        labels.append(label)

    font_name = detect_and_add_font()
    if font_name:
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    n = len(codes)
    fig_width = max(12, min(80, n * 0.35))
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.set_title("Shenwan second-level industry clusters (x = selected)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_svg)
    plt.close(fig)
    print("Wrote dendrogram to", out_png)
