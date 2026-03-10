"""
Generate figures/nlp4lp_instantiation_pipeline_v2.png

Publication-ready methodology figure for the NLP4LP paper.

Two-branch layout:
  Top branch    : Query text → Numeric mention extraction → Mention typing
  Bottom branch : Predicted schema → Eligible scalar slots → Slot typing
  Merge point   : Deterministic compatibility scoring / assignment
  Output        : Scalar slot–value assignments

Run:
    python figures/gen_instantiation_pipeline.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── Output path ─────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
OUT  = HERE / "nlp4lp_instantiation_pipeline_v2.png"

# ── Palette (clean academic — two muted colours for the two branches) ────────
COL_QUERY  = "#D6E4F0"   # light steel-blue   → query branch
COL_SCHEMA = "#D5E8D4"   # light sage-green   → schema branch
COL_MERGE  = "#E8DEF8"   # light lavender     → assignment block
COL_OUTPUT = "#FFF2CC"   # pale amber         → output block
EDGE       = "#555555"   # dark-grey borders & arrows
TEXT       = "#1A1A1A"   # near-black text

# DejaVu Sans is bundled with matplotlib and is always available —
# no system font installation needed.
FONT = "DejaVu Sans"
FS_BOX   = 9.5   # font size inside boxes
FS_LABEL = 8.0   # branch label font size

# ── Canvas ───────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 9.0, 4.4   # inches  (fits in a two-column journal page)
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ── Helper: draw a rounded box ───────────────────────────────────────────────
def box(cx, cy, w, h, label, color, pad=0.012, lw=1.1):
    rect = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={pad}",
        facecolor=color, edgecolor=EDGE, linewidth=lw,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, label,
            ha="center", va="center",
            fontsize=FS_BOX, fontfamily=FONT, color=TEXT,
            fontweight="normal", zorder=4,
            multialignment="center")

# ── Helper: draw a straight horizontal arrow ─────────────────────────────────
def arrow_h(x0, x1, y, color=EDGE, lw=1.2):
    ax.annotate(
        "", xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=12),
        zorder=2,
    )

# ── Helper: draw a diagonal / bent arrow (two-segment L-shape) ──────────────
def arrow_diag(x0, y0, x1, y1, color=EDGE, lw=1.2):
    """Draw an L-shaped connector: vertical segment then horizontal."""
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            mutation_scale=12,
            connectionstyle="angle,angleA=0,angleB=90,rad=4",
        ),
        zorder=2,
    )

# ── Layout geometry ──────────────────────────────────────────────────────────
# Horizontal positions (normalised 0–1)
X_GAP   = 0.015   # gap between box edge and arrow start/end
BOX_W   = 0.175   # standard box width
BOX_H   = 0.130   # standard box height

# Column x-centres for the three branch boxes
X1 = 0.11    # column 1
X2 = 0.32    # column 2
X3 = 0.53    # column 3

# Merge + output blocks
X4 = 0.72    # "Deterministic compatibility scoring / assignment"
X5 = 0.91    # "Scalar slot–value assignments"

WIDE_W  = 0.185  # slightly wider for the merge block (longer label)
OUT_W   = 0.160

# Row y-centres
Y_TOP = 0.72    # query branch
Y_BOT = 0.28    # schema branch
Y_MID = 0.50    # merge

# ── Draw boxes ───────────────────────────────────────────────────────────────

# — Query branch —
box(X1, Y_TOP, BOX_W, BOX_H, "Query text",                   COL_QUERY)
box(X2, Y_TOP, BOX_W, BOX_H, "Numeric mention\nextraction",  COL_QUERY)
box(X3, Y_TOP, BOX_W, BOX_H, "Mention\ntyping",              COL_QUERY)

# — Schema branch —
box(X1, Y_BOT, BOX_W, BOX_H, "Predicted\nschema",            COL_SCHEMA)
box(X2, Y_BOT, BOX_W, BOX_H, "Eligible\nscalar slots",       COL_SCHEMA)
box(X3, Y_BOT, BOX_W, BOX_H, "Slot\ntyping",                 COL_SCHEMA)

# — Merge block —
box(X4, Y_MID, WIDE_W, BOX_H + 0.04,
    "Deterministic compatibility\nscoring / assignment",      COL_MERGE)

# — Output block —
box(X5, Y_MID, OUT_W,  BOX_H,
    "Scalar slot–value\nassignments",                         COL_OUTPUT)

# ── Horizontal arrows within each branch ─────────────────────────────────────
hw = BOX_W / 2
# Query branch
arrow_h(X1 + hw + X_GAP, X2 - hw - X_GAP, Y_TOP)
arrow_h(X2 + hw + X_GAP, X3 - hw - X_GAP, Y_TOP)

# Schema branch
arrow_h(X1 + hw + X_GAP, X2 - hw - X_GAP, Y_BOT)
arrow_h(X2 + hw + X_GAP, X3 - hw - X_GAP, Y_BOT)

# ── Diagonal arrows from last branch box → merge block ───────────────────────
# Top branch: from right edge of "Mention typing" down-and-right to merge
arrow_diag(X3 + hw + X_GAP, Y_TOP,
           X4 - WIDE_W / 2 - X_GAP, Y_MID)

# Bottom branch: from right edge of "Slot typing" up-and-right to merge
arrow_diag(X3 + hw + X_GAP, Y_BOT,
           X4 - WIDE_W / 2 - X_GAP, Y_MID)

# ── Arrow from merge → output ─────────────────────────────────────────────────
arrow_h(X4 + WIDE_W / 2 + X_GAP, X5 - OUT_W / 2 - X_GAP, Y_MID)

# ── Branch labels (small, to the left of each first box) ─────────────────────
ax.text(0.005, Y_TOP, "Query\nbranch",
        ha="left", va="center", fontsize=FS_LABEL,
        fontstyle="italic", color="#336699", fontfamily=FONT, zorder=5)
ax.text(0.005, Y_BOT, "Schema\nbranch",
        ha="left", va="center", fontsize=FS_LABEL,
        fontstyle="italic", color="#336633", fontfamily=FONT, zorder=5)

# ── Light vertical dashed divider between branches and merge ─────────────────
ax.axvline(x=0.625, ymin=0.12, ymax=0.88,
           color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=1)

# ── Layout title (small) ─────────────────────────────────────────────────────
# (no title: figure captions handle that in the paper)

plt.tight_layout(pad=0.3)
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
