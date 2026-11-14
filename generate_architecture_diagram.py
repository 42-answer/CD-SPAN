import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = os.path.join(BASE_DIR, '.mpl_cache')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle
from PIL import Image
import numpy as np

# Set matplotlib fonts to standard publication sans-serif
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

# Canvas setup: ultra high-res 20.0 x 14.0 inches at 300 DPI (6000 x 4200 px)
fig = plt.figure(figsize=(20.0, 14.0), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 20.0)
ax.set_ylim(0, 14.0)
ax.axis('off')

# Color Palette inspired directly by Figure 2 of original SPAN paper + CVF standards
C_PANEL_BG = '#FBFDFF'
C_PANEL_BORDER = '#94A3B8'
C_HEADER_BG = '#0F172A'
C_TENSOR_BORDER = '#475569'

# 3D Module Colors (Front, Top, Side, Edge)
C_CONV_FRONT = '#D1FAE5'   # Sage Emerald
C_CONV_TOP   = '#ECFDF5'
C_CONV_SIDE  = '#A7F3D0'
C_CONV_EDGE  = '#059669'

C_SPAB_FRONT = '#E0E7FF'   # Soft Periwinkle
C_SPAB_TOP   = '#EEF2FF'
C_SPAB_SIDE  = '#C7D2FE'
C_SPAB_EDGE  = '#4338CA'

C_ACT_FRONT  = '#F1F5F9'   # Slate Slab (Activation)
C_ACT_TOP    = '#F8FAFC'
C_ACT_SIDE   = '#E2E8F0'
C_ACT_EDGE   = '#64748B'

C_CONCAT_FRONT = '#FEF3C7' # Warm Amber Ochre
C_CONCAT_TOP   = '#FFFBEB'
C_CONCAT_SIDE  = '#FDE68A'
C_CONCAT_EDGE  = '#D97706'

C_PS_FRONT   = '#FEF08A'   # Yellow (PixelShuffle)
C_PS_TOP     = '#FEF9C3'
C_PS_SIDE    = '#FDE047'
C_PS_EDGE    = '#CA8A04'

C_SIGMA_FRONT = '#FFE4E6'  # Coral / Rose
C_SIGMA_TOP   = '#FFF1F2'
C_SIGMA_SIDE  = '#FECDD3'
C_SIGMA_EDGE  = '#E11D48'

C_DIFF_FRONT  = '#EDE9FE'  # Purple / Violet
C_DIFF_TOP    = '#F5F3FF'
C_DIFF_SIDE   = '#DDD6FE'
C_DIFF_EDGE   = '#7C3AED'

# Helper: Draw Outer Panel with Title Bar
def draw_panel(ax, x, y, w, h, title, subtitle=''):
    panel = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.18",
                           linewidth=1.8, edgecolor=C_PANEL_BORDER, facecolor=C_PANEL_BG, zorder=1)
    ax.add_patch(panel)
    hbar = FancyBboxPatch((x, y + h - 0.58), w, 0.58, boxstyle="round,pad=0.02,rounding_size=0.15",
                          linewidth=0, facecolor=C_HEADER_BG, zorder=2)
    ax.add_patch(hbar)
    ax.text(x + 0.35, y + h - 0.29, title, ha='left', va='center',
            fontsize=10.8, fontweight='bold', color='#FFFFFF', zorder=3)
    if subtitle:
        ax.text(x + w - 0.35, y + h - 0.29, subtitle, ha='right', va='center',
                fontsize=8.5, fontfamily='monospace', color='#94A3B8', zorder=3)

# Helper: Draw 3D Isometric Shaded Cuboid
def draw_iso_cube(ax, x, y, w, h, d=0.35,
                  front_color='#E0E7FF', top_color='#EEF2FF', side_color='#C7D2FE', edge_color='#4338CA',
                  lw=1.4, title='', sub='', chan_label='', title_size=8.6, sub_size=7.0, zorder=4):
    dx = d * 0.707
    dy = d * 0.707

    # Front Face
    front = Rectangle((x, y), w, h, facecolor=front_color, edgecolor=edge_color, lw=lw, zorder=zorder)
    ax.add_patch(front)

    # Top Face
    top = Polygon([[x, y + h], [x + dx, y + h + dy], [x + w + dx, y + h + dy], [x + w, y + h]],
                  closed=True, facecolor=top_color, edgecolor=edge_color, lw=lw, zorder=zorder+1)
    ax.add_patch(top)

    # Right Side Face
    side = Polygon([[x + w, y], [x + w, y + h], [x + w + dx, y + h + dy], [x + w + dx, y + dy]],
                   closed=True, facecolor=side_color, edgecolor=edge_color, lw=lw, zorder=zorder)
    ax.add_patch(side)

    # Front Face Labels
    cy = y + h / 2
    if title and sub:
        ax.text(x + w/2, cy + 0.14, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color='#0F172A', zorder=zorder+2)
        ax.text(x + w/2, cy - 0.16, sub, ha='center', va='center', fontsize=sub_size, color='#334155', zorder=zorder+2)
    elif title:
        ax.text(x + w/2, cy, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color='#0F172A', zorder=zorder+2)

    # Channel depth label below front face
    if chan_label:
        ax.text(x + w/2, y - 0.22, chan_label, ha='center', va='top', fontsize=7.4, fontfamily='monospace', fontweight='bold', color='#1E293B', zorder=zorder+2)

    return (x, y, w, h, dx, dy)

# Helper: Draw Flat 2D Block
def draw_block(ax, x, y, w, h, title, sub='', dim='', color='#FFFFFF', border='#64748B', lw=1.4,
               title_size=8.6, sub_size=7.2, dim_size=6.6, title_color='#0F172A', zorder=3):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.10",
                         linewidth=lw, edgecolor=border, facecolor=color, zorder=zorder)
    ax.add_patch(box)
    cy = y + h/2
    if dim and sub:
        ax.text(x + w/2, cy + 0.22, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=title_color, zorder=zorder+1)
        ax.text(x + w/2, cy + 0.01, sub, ha='center', va='center', fontsize=sub_size, color='#334155', zorder=zorder+1)
        ax.text(x + w/2, cy - 0.22, dim, ha='center', va='center', fontsize=dim_size, fontfamily='monospace', fontweight='bold', color='#1D4ED8', zorder=zorder+1)
    elif sub:
        ax.text(x + w/2, cy + 0.14, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=title_color, zorder=zorder+1)
        ax.text(x + w/2, cy - 0.16, sub, ha='center', va='center', fontsize=sub_size, color='#334155', zorder=zorder+1)
    elif dim:
        ax.text(x + w/2, cy + 0.14, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=title_color, zorder=zorder+1)
        ax.text(x + w/2, cy - 0.16, dim, ha='center', va='center', fontsize=dim_size, fontfamily='monospace', fontweight='bold', color='#1D4ED8', zorder=zorder+1)
    else:
        ax.text(x + w/2, cy, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=title_color, zorder=zorder+1)

# Helper: Draw Mathematical Operation Circle
def draw_op(ax, x, y, symbol, r=0.20, color='#E0F2FE', border='#0284C7', text_color='#0369A1', size=13):
    circle = Circle((x, y), r, facecolor=color, edgecolor=border, linewidth=1.6, zorder=6)
    ax.add_patch(circle)
    ax.text(x, y, symbol, ha='center', va='center', fontsize=size, fontweight='bold', color=text_color, zorder=7)

# Helper: Draw Straight Arrow Connection
def draw_conn(ax, x1, y1, x2, y2, color='#334155', lw=1.5, text='', text_pos='top', style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=1.5, shrinkB=1.5), zorder=5)
    if text:
        tx = (x1 + x2)/2
        ty = (y1 + y2)/2 + (0.15 if text_pos=='top' else -0.17)
        ax.text(tx, ty, text, ha='center', va='center', fontsize=7.8, color='#0F172A', fontweight='bold', zorder=6)

# Helper: Draw 3x3 Discrete Kernel Matrix Grid
def draw_kernel_grid(ax, x, y, matrix, label, factor_label='', cell_size=0.16, font_size=5.8):
    title = f"{factor_label} {label}".strip() if factor_label else label
    ax.text(x + 1.5 * cell_size, y + 3 * cell_size + 0.08, title, ha='center', va='bottom',
            fontsize=7.6, fontweight='bold', color='#92400E', zorder=6)
    for r in range(3):
        for c in range(3):
            val = matrix[r][c]
            bg = '#FEF3C7' if val != 0 else '#FFFFFF'
            rect = Rectangle((x + c * cell_size, y + (2 - r) * cell_size), cell_size, cell_size,
                             facecolor=bg, edgecolor='#D97706', lw=0.8, zorder=5)
            ax.add_patch(rect)
            ax.text(x + (c + 0.5) * cell_size, y + (2.5 - r) * cell_size, str(val),
                    ha='center', va='center', fontsize=font_size, fontweight='bold', color='#78350F', zorder=6)

# =========================================================================
# PANEL (A): GLOBAL MACRO-ARCHITECTURE OF CAD-SPAN
# Inspired by Figure 2 of original SPAN paper (Hongyuan Yu et al., CVPRW 2024)
# =========================================================================
draw_panel(ax, 0.30, 7.90, 19.40, 5.85,
           '(a) Global Macro-Architecture of Proposed CAD-SPAN Framework',
           'Input LR -> Shallow Conv3XC -> 6 CAD-SPABs -> Refine Conv -> Concat -> PixelShuffle -> HR Output')

# 1. Real LR Input Baby Image (Set5)
lr_img_path = os.path.join(ASSETS_DIR, 'diag_lr_baby.png')
if os.path.exists(lr_img_path):
    lr_pil = Image.open(lr_img_path)
    ax.imshow(lr_pil, extent=[0.55, 1.75, 8.85, 10.45], aspect='auto', zorder=4)
    rect_lr = Rectangle((0.55, 8.85), 1.20, 1.60, linewidth=1.5, edgecolor='#059669', facecolor='none', zorder=5)
    ax.add_patch(rect_lr)
    ax.text(1.15, 8.60, 'LR Input $I_{LR}$\n$[B, 3, H, W]$', ha='center', va='top', fontsize=7.8, fontweight='bold', color='#065F46')
    ax.text(1.15, 8.16, 'Mean Shift:\n$(I - \mu) \\times 255$', ha='center', va='top', fontsize=6.8, color='#475569')

# Arrow: LR Input -> Shallow Conv
draw_conn(ax, 1.80, 9.65, 2.25, 9.65)

# 2. Shallow Conv3XC (conv_1 in code: 3 -> 48) - 3D Sage Green Prism
draw_iso_cube(ax, 2.25, 8.85, 0.85, 1.60, d=0.35,
              front_color=C_CONV_FRONT, top_color=C_CONV_TOP, side_color=C_CONV_SIDE, edge_color=C_CONV_EDGE,
              title='Conv3XC', sub='(conv_1)', chan_label='48')

# Output of conv_1 is Feat.Map 1 (F0)
draw_conn(ax, 3.40, 9.65, 3.90, 9.65)
ax.text(3.65, 9.88, r'$F_0$', ha='center', va='bottom', fontsize=8.6, fontweight='bold', color='#047857')

# 3. Cascaded CAD-SPAB Blocks (#1 through #6) - 3D Periwinkle Cuboids
# CAD-SPAB #1
draw_iso_cube(ax, 3.90, 8.85, 1.20, 1.60, d=0.35,
              front_color=C_SPAB_FRONT, top_color=C_SPAB_TOP, side_color=C_SPAB_SIDE, edge_color=C_SPAB_EDGE,
              title=r'$\mathbf{CAD\text{-}SPAB}_1$', sub='Stage 1', chan_label='48')

draw_conn(ax, 5.40, 9.65, 5.90, 9.65)
ax.text(5.65, 9.88, r'$F_{b1}$', ha='center', va='bottom', fontsize=8.6, fontweight='bold', color='#1D4ED8')

# CAD-SPAB #2
draw_iso_cube(ax, 5.90, 8.85, 1.20, 1.60, d=0.35,
              front_color=C_SPAB_FRONT, top_color=C_SPAB_TOP, side_color=C_SPAB_SIDE, edge_color=C_SPAB_EDGE,
              title=r'$\mathbf{CAD\text{-}SPAB}_2$', sub='Stage 2', chan_label='48')

# Cascaded Ellipsis Connection (Stages 3 & 4 in series)
ax.plot([7.40, 8.00], [9.65, 9.65], color='#2563EB', lw=2.0, linestyle='--', zorder=4)
ax.annotate('', xy=(8.00, 9.65), xytext=(7.85, 9.65), arrowprops=dict(arrowstyle='->', color='#2563EB', lw=2.0), zorder=5)
ax.text(7.70, 9.88, r'$\cdots$', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#1D4ED8')
ax.text(7.70, 9.28, 'Stages #3, #4\nin series', ha='center', va='top', fontsize=6.8, color='#64748B')

# CAD-SPAB #5
draw_iso_cube(ax, 8.00, 8.85, 1.20, 1.60, d=0.35,
              front_color=C_SPAB_FRONT, top_color=C_SPAB_TOP, side_color=C_SPAB_SIDE, edge_color=C_SPAB_EDGE,
              title=r'$\mathbf{CAD\text{-}SPAB}_5$', sub='Stage 5', chan_label='48')

draw_conn(ax, 9.50, 9.65, 9.95, 9.65)
ax.text(9.72, 9.88, r'$F_{b5}$', ha='center', va='bottom', fontsize=8.6, fontweight='bold', color='#B45309')

# CAD-SPAB #6
draw_iso_cube(ax, 9.95, 8.85, 1.20, 1.60, d=0.35,
              front_color=C_SPAB_FRONT, top_color=C_SPAB_TOP, side_color=C_SPAB_SIDE, edge_color=C_SPAB_EDGE,
              title=r'$\mathbf{CAD\text{-}SPAB}_6$', sub='Stage 6', chan_label='48')

# Refinement Conv3XC (conv_2 in code: 48 -> 48) - 3D Sage Green Prism
draw_conn(ax, 11.45, 9.65, 11.85, 9.65)
ax.text(11.65, 9.88, r'$F_{b6}$', ha='center', va='bottom', fontsize=8.6, fontweight='bold', color='#1D4ED8')

draw_iso_cube(ax, 11.85, 8.85, 0.85, 1.60, d=0.35,
              front_color=C_CONV_FRONT, top_color=C_CONV_TOP, side_color=C_CONV_SIDE, edge_color=C_CONV_EDGE,
              title='Conv3XC', sub='(conv_2)', chan_label='48')

# 4. ALL 4 HIERARCHICAL SKIP CONNECTIONS ROUTED ACROSS THE TOP
# Track 1: Feat.Map 1 (F0) from after conv_1
ax.plot([3.65, 3.65, 13.55], [9.65, 12.80, 12.80], color='#059669', lw=1.6, zorder=3)
ax.plot([13.55, 13.55], [12.80, 10.80], color='#059669', lw=1.6, zorder=3)
ax.annotate('', xy=(13.55, 10.60), xytext=(13.55, 10.80), arrowprops=dict(arrowstyle='->', color='#059669', lw=1.6), zorder=5)
ax.text(5.20, 12.92, r'$\mathrm{Feat.Map}_1$ ($F_0$, Shallow Features)', fontsize=7.8, color='#047857', fontweight='bold', va='bottom')

# Track 2: Feat.Map 2 (Fb1) from after CAD-SPAB #1
ax.plot([5.65, 5.65, 13.85], [9.65, 12.35, 12.35], color='#2563EB', lw=1.6, zorder=3)
ax.plot([13.85, 13.85], [12.35, 10.80], color='#2563EB', lw=1.6, zorder=3)
ax.annotate('', xy=(13.85, 10.60), xytext=(13.85, 10.80), arrowprops=dict(arrowstyle='->', color='#2563EB', lw=1.6), zorder=5)
ax.text(6.60, 12.47, r'$\mathrm{Feat.Map}_2$ ($F_{b1}$, Early Stage Details)', fontsize=7.8, color='#1D4ED8', fontweight='bold', va='bottom')

# Track 3: Feat.Map 6 (Fb5) intermediate tap from Stage 5 / Block 6
ax.plot([9.72, 9.72, 14.15], [9.65, 11.90, 11.90], color='#B45309', lw=1.6, zorder=3)
ax.plot([14.15, 14.15], [11.90, 10.80], color='#B45309', lw=1.6, zorder=3)
ax.annotate('', xy=(14.15, 10.60), xytext=(14.15, 10.80), arrowprops=dict(arrowstyle='->', color='#B45309', lw=1.6), zorder=5)
ax.text(10.20, 12.02, r'$\mathrm{Feat.Map}_6$ ($F_{b5}$, Deep Residual)', fontsize=7.8, color='#92400E', fontweight='bold', va='bottom')

# Track 4: Feat.Map 7 (Fb6_conv) from refine conv_2
ax.plot([12.95, 12.95, 14.45], [9.65, 11.45, 11.45], color='#0D9488', lw=1.6, zorder=3)
ax.plot([14.45, 14.45], [11.45, 10.80], color='#0D9488', lw=1.6, zorder=3)
ax.annotate('', xy=(14.45, 10.60), xytext=(14.45, 10.80), arrowprops=dict(arrowstyle='->', color='#0D9488', lw=1.6), zorder=5)
ax.text(13.65, 11.57, r'$\mathrm{Feat.Map}_7$ ($F_{b6\_conv}$)', fontsize=7.8, color='#0F766E', fontweight='bold', va='bottom')

# 5. Concatenation Layer - 3D Warm Ochre Prism (192 channels)
draw_iso_cube(ax, 13.40, 8.85, 1.40, 1.60, d=0.35,
              front_color=C_CONCAT_FRONT, top_color=C_CONCAT_TOP, side_color=C_CONCAT_SIDE, edge_color=C_CONCAT_EDGE,
              title='Concat.', sub=r'$[F_0, F_7, F_1, F_6]$', chan_label='192')

# 6. Conv 1x1 Fusion (conv_cat in code: 192 -> 48) - 3D Sage Green Prism
draw_conn(ax, 15.10, 9.65, 15.55, 9.65)
draw_iso_cube(ax, 15.55, 8.85, 0.85, 1.60, d=0.35,
              front_color=C_CONV_FRONT, top_color=C_CONV_TOP, side_color=C_CONV_SIDE, edge_color=C_CONV_EDGE,
              title=r'Conv $1\times 1$', sub='(conv_cat)', chan_label='48')

# 7. PixelShuffle Upsampler - 3D Pastel Yellow Prism
draw_conn(ax, 16.70, 9.65, 17.15, 9.65)
draw_iso_cube(ax, 17.15, 8.85, 0.95, 1.60, d=0.35,
              front_color=C_PS_FRONT, top_color=C_PS_TOP, side_color=C_PS_SIDE, edge_color=C_PS_EDGE,
              title='Pixel', sub='Shuffle', chan_label='3 (RGB)')

# 8. Perspective Expansion Lines to HR Output Image
ax.plot([18.35, 18.75], [10.55, 10.65], color='#16A34A', lw=1.2, linestyle='--', zorder=4)
ax.plot([18.35, 18.75], [8.95, 8.85], color='#16A34A', lw=1.2, linestyle='--', zorder=4)

# Real HR Output Baby Image (Set5)
hr_img_path = os.path.join(ASSETS_DIR, 'diag_hr_baby.png')
if os.path.exists(hr_img_path):
    hr_pil = Image.open(hr_img_path)
    ax.imshow(hr_pil, extent=[18.65, 19.55, 8.85, 10.75], aspect='auto', zorder=4)
    rect_hr = Rectangle((18.65, 8.85), 0.90, 1.90, linewidth=1.8, edgecolor='#16A34A', facecolor='none', zorder=5)
    ax.add_patch(rect_hr)
    ax.text(19.10, 8.65, 'HR Output $I_{SR}$\n$[B, 3, 4H, 4W]$', ha='center', va='top', fontsize=7.8, fontweight='bold', color='#14532D')

# Component Color Legend (Upper Right Corner of Panel a)
leg_box = FancyBboxPatch((15.40, 11.20), 4.10, 1.80, boxstyle="round,pad=0.03,rounding_size=0.08",
                         linewidth=1.2, edgecolor='#CBD5E1', facecolor='#FFFFFF', zorder=3)
ax.add_patch(leg_box)
ax.text(15.55, 12.75, 'COMPONENT COLOR LEGEND', fontsize=7.8, fontweight='bold', color='#334155', zorder=4)

def draw_legend_iso(ax, x, y, fc, ec, text):
    p = Rectangle((x, y), 0.28, 0.20, facecolor=fc, edgecolor=ec, lw=1.0, zorder=4)
    ax.add_patch(p)
    ax.text(x + 0.38, y + 0.10, text, ha='left', va='center', fontsize=7.0, color='#1E293B', fontweight='bold', zorder=4)

draw_legend_iso(ax, 15.55, 12.38, C_CONV_FRONT, C_CONV_EDGE, 'Conv3XC (Re-param)')
draw_legend_iso(ax, 15.55, 12.02, C_SPAB_FRONT, C_SPAB_EDGE, 'CAD-SPAB Block')
draw_legend_iso(ax, 15.55, 11.66, C_CONCAT_FRONT, C_CONCAT_EDGE, 'Multi-Scale Concat')
draw_legend_iso(ax, 17.55, 12.38, C_PS_FRONT, C_PS_EDGE, 'PixelShuffle (x4)')
draw_legend_iso(ax, 17.55, 12.02, C_SIGMA_FRONT, C_SIGMA_EDGE, 'Origin-Symmetric Gate')
draw_legend_iso(ax, 17.55, 11.66, C_DIFF_FRONT, C_DIFF_EDGE, 'Dual-Differential Prior')

# Zoom-in Projection Lines from CAD-SPAB #1 down to Panel (b) container
ax.plot([4.20, 0.50], [8.75, 7.60], color='#2563EB', lw=1.2, linestyle='--', zorder=2)
ax.plot([5.00, 12.90], [8.75, 7.60], color='#2563EB', lw=1.2, linestyle='--', zorder=2)


# =========================================================================
# PANEL (B): CAD-SPAB MICRO-ARCHITECTURE
# Detailed view of Curvature-Aware Directional Attention Mechanism
# =========================================================================
draw_panel(ax, 0.30, 0.25, 12.80, 7.45,
           '(b) CAD-SPAB Micro-Architecture: Differential Attention Block',
           'Content (H + Xin) modulated by Curvature Gate A_mod')

# Yellow parchment background for Panel (b) body echoing original paper
card_bg = FancyBboxPatch((0.40, 0.32), 12.60, 6.75, boxstyle="round,pad=0.03,rounding_size=0.12",
                         linewidth=1.2, edgecolor='#FDE68A', facecolor='#FFFDF7', zorder=2)
ax.add_patch(card_bg)

# 1. Real Input Feature Map Heatmap (Feat. Map 2 from CAD-SPAN)
hm2_path = os.path.join(ASSETS_DIR, 'diag_feat_map2.png')
if os.path.exists(hm2_path):
    hm2_pil = Image.open(hm2_path)
    ax.imshow(hm2_pil, extent=[0.55, 1.45, 4.30, 5.70], aspect='auto', zorder=4)
    rect_hm2 = Rectangle((0.55, 4.30), 0.90, 1.40, linewidth=1.4, edgecolor='#3B82F6', facecolor='none', zorder=5)
    ax.add_patch(rect_hm2)
    ax.text(1.00, 4.05, r'$\mathrm{Feat.Map}_2$' + '\n' + r'$X_{in} \in \mathbb{R}^{B \times 48 \times H \times W}$',
            ha='center', va='top', fontsize=7.2, fontweight='bold', color='#1D4ED8', zorder=5)

# 2. Local Identity Shortcut across top (Cleanly placed at y = 6.55)
ax.plot([1.00, 1.00, 8.20], [5.70, 6.55, 6.55], color='#2563EB', lw=1.6, zorder=4)
draw_conn(ax, 8.20, 6.55, 8.20, 6.45)
ax.text(3.50, 6.68, r'Local Identity Shortcut: $X_{in} \in \mathbb{R}^{B \times 48 \times H \times W}$',
        fontsize=8.0, color='#1D4ED8', fontweight='bold', va='bottom', zorder=5)

# 3. Main Convolutional Trunk: 3x Conv3XC with interleaved SiLU activations
draw_conn(ax, 1.50, 5.00, 1.85, 5.00)

# Conv3XC 1 (3D Sage Prism)
draw_iso_cube(ax, 1.85, 4.30, 0.75, 1.40, d=0.25,
              front_color=C_CONV_FRONT, top_color=C_CONV_TOP, side_color=C_CONV_SIDE, edge_color=C_CONV_EDGE,
              title=r'$\mathrm{Conv3}_1$', chan_label='48')

# SiLU Activation 1 (3D Slate Slab)
draw_conn(ax, 2.75, 5.00, 3.05, 5.00)
draw_iso_cube(ax, 3.05, 4.45, 0.35, 1.10, d=0.20,
              front_color=C_ACT_FRONT, top_color=C_ACT_TOP, side_color=C_ACT_SIDE, edge_color=C_ACT_EDGE,
              title=r'$\sigma$')

# Conv3XC 2 (3D Sage Prism)
draw_conn(ax, 3.55, 5.00, 3.85, 5.00)
draw_iso_cube(ax, 3.85, 4.30, 0.75, 1.40, d=0.25,
              front_color=C_CONV_FRONT, top_color=C_CONV_TOP, side_color=C_CONV_SIDE, edge_color=C_CONV_EDGE,
              title=r'$\mathrm{Conv3}_2$', chan_label='48')

# SiLU Activation 2 (3D Slate Slab)
draw_conn(ax, 4.75, 5.00, 5.05, 5.00)
draw_iso_cube(ax, 5.05, 4.45, 0.35, 1.10, d=0.20,
              front_color=C_ACT_FRONT, top_color=C_ACT_TOP, side_color=C_ACT_SIDE, edge_color=C_ACT_EDGE,
              title=r'$\sigma$')

# Conv3XC 3 (3D Sage Prism - No activation afterwards!)
draw_conn(ax, 5.55, 5.00, 5.85, 5.00)
draw_iso_cube(ax, 5.85, 4.30, 0.75, 1.40, d=0.25,
              front_color=C_CONV_FRONT, top_color=C_CONV_TOP, side_color=C_CONV_SIDE, edge_color=C_CONV_EDGE,
              title=r'$\mathrm{Conv3}_3$', chan_label='48')

# Output Feature H Block
draw_conn(ax, 6.75, 5.00, 7.05, 5.00)
draw_block(ax, 7.05, 4.55, 0.85, 0.90, r'Feature $H$', r'$H \equiv \mathrm{out}_3$', '[B, 48, H, W]',
           color='#FEF3C7', border='#D97706', title_size=8.4, sub_size=7.2, dim_size=6.6)

# Feature H splits:
# Path A: Up to Addition (+) with Xin
ax.plot([7.47, 7.47, 8.00], [5.45, 6.25, 6.25], color='#D97706', lw=1.5, zorder=4)
draw_conn(ax, 8.00, 6.25, 8.05, 6.25)
ax.text(7.65, 6.35, r'$H$', fontsize=8.0, color='#B45309', fontweight='bold', zorder=5)

# Element-wise Addition (+) Node: (H + Xin)
draw_op(ax, 8.20, 6.25, r'$\oplus$', r=0.20, color='#DBEAFE', border='#2563EB', text_color='#1D4ED8', size=12)

# Content Residual Block: R = H + Xin
draw_conn(ax, 8.40, 6.25, 8.70, 6.25)
draw_block(ax, 8.70, 5.80, 1.40, 0.90, r'Content Residual', r'$R = H + X_{in}$', '[B, 48, H, W]',
           color='#EFF6FF', border='#3B82F6', title_size=8.4, sub_size=7.4, dim_size=6.6)

# Path B: Down to Novel Differential Attention Generator
ax.plot([7.47, 7.47], [4.55, 1.15], color='#B45309', lw=1.6, zorder=4)

# Stream 1: Origin-Symmetric Gate (Top Lane)
draw_conn(ax, 7.47, 3.90, 7.85, 3.90)
draw_block(ax, 7.85, 3.50, 2.70, 0.85,
           '1. Origin-Symmetric Gate',
           r'$A_{sym}(H) = \frac{1}{2}\tanh(\tau \cdot H)$  $(\tau^* = 0.502)$' + '\n' +
           r'Exact odd symmetry: $A_{sym}(-H) = -A_{sym}(H)$',
           'Range: $(-0.5, 0.5)$',
           color=C_SIGMA_FRONT, border=C_SIGMA_EDGE, title_size=8.2, sub_size=6.8, dim_size=6.4)

# Stream 2: Directional Gradient Flux Operator (Middle Lane)
draw_conn(ax, 7.47, 2.45, 7.85, 2.45)
box_flux = FancyBboxPatch((7.85, 1.75), 2.80, 1.40, boxstyle="round,pad=0.03,rounding_size=0.10",
                          linewidth=1.4, edgecolor='#D97706', facecolor='#FFFBEB', zorder=3)
ax.add_patch(box_flux)
ax.text(7.98, 2.95, '2. Directional Gradient Flux Operator', ha='left', va='center', fontsize=8.0, fontweight='bold', color='#0F172A', zorder=4)
ax.text(7.98, 2.70, r'$G_x = \frac{1}{8}K_x *_{dw} H, \quad G_y = \frac{1}{8}K_y *_{dw} H$', ha='left', va='center', fontsize=7.2, color='#334155', zorder=4)
ax.text(7.98, 2.45, r'$\|\nabla H\|_2 = \sqrt{G_x^2 + G_y^2 + \epsilon}$', ha='left', va='center', fontsize=7.2, color='#334155', zorder=4)
ax.text(7.98, 2.22, 'Directional Step Boundary Flux', ha='left', va='center', fontsize=6.6, color='#475569', zorder=4)
ax.text(7.98, 1.98, '[B, 48, H, W] (groups=48)', ha='left', va='center', fontsize=6.4, fontfamily='monospace', fontweight='bold', color='#1D4ED8', zorder=4)

sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
draw_kernel_grid(ax, 9.45, 1.82, sobel_x, r'$K_x$', factor_label=r'$\frac{1}{8}$', cell_size=0.13, font_size=4.8)
draw_kernel_grid(ax, 10.02, 1.82, sobel_y, r'$K_y$', factor_label=r'$\frac{1}{8}$', cell_size=0.13, font_size=4.8)

# Stream 3: Isotropic Curvature Operator (Bottom Lane)
draw_conn(ax, 7.47, 1.15, 7.85, 1.15)
box_curv = FancyBboxPatch((7.85, 0.55), 2.80, 1.10, boxstyle="round,pad=0.03,rounding_size=0.10",
                          linewidth=1.4, edgecolor='#16A34A', facecolor='#F0FDF4', zorder=3)
ax.add_patch(box_curv)
ax.text(7.98, 1.48, '3. Isotropic Curvature Operator', ha='left', va='center', fontsize=8.0, fontweight='bold', color='#0F172A', zorder=4)
ax.text(7.98, 1.25, r'$\Delta H = |\nabla^2 H| = |K_L *_{dw} H|$', ha='left', va='center', fontsize=7.2, color='#334155', zorder=4)
ax.text(7.98, 1.02, 'Corner & Ridge Extrema Detector', ha='left', va='center', fontsize=6.6, color='#475569', zorder=4)
ax.text(7.98, 0.80, '[B, 48, H, W] (groups=48)', ha='left', va='center', fontsize=6.4, fontfamily='monospace', fontweight='bold', color='#1D4ED8', zorder=4)

laplacian_k = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
draw_kernel_grid(ax, 9.75, 0.60, laplacian_k, r'$K_L$', cell_size=0.13, font_size=4.8)

# Fusion of Streams 2 & 3 into Prior Phi(H)
ax.plot([10.70, 10.85], [2.45, 1.75], color='#334155', lw=1.5, zorder=4)
ax.plot([10.70, 10.85], [1.15, 1.75], color='#334155', lw=1.5, zorder=4)
draw_conn(ax, 10.85, 1.75, 10.95, 1.75)

draw_block(ax, 10.95, 1.20, 1.05, 1.15,
           r'Prior $\Phi(H)$',
           r'$\Phi = \tanh($' + '\n' + r'$\gamma_s \|\nabla H\| +$' + '\n' + r'$\gamma_l \Delta H)$',
           r'$\gamma_s^*=0.008, \gamma_l^*=0.001$',
           color=C_DIFF_FRONT, border=C_DIFF_EDGE, title_size=7.8, sub_size=6.2, dim_size=5.8)

# Affine Scaling Node: (1 + Phi)
ax.plot([11.47, 11.47], [2.35, 2.75], color='#7C3AED', lw=1.5, zorder=4)
draw_op(ax, 11.47, 2.98, r'$\oplus$', r=0.17, color='#F3E8FF', border='#7C3AED', text_color='#6D28D9', size=11)
ax.text(11.72, 2.98, r'$\mathbf{1}$', fontsize=8.5, fontweight='bold', color='#6D28D9', va='center', zorder=5)

# Hadamard Product with Asym: A_mod = Asym * (1 + Phi)
draw_op(ax, 11.47, 3.90, r'$\odot$', r=0.19, color='#FCE7F3', border='#EC4899', text_color='#BE185D', size=12)
draw_conn(ax, 10.55, 3.90, 11.28, 3.90, text=r'$A_{sym}$')
draw_conn(ax, 11.47, 3.15, 11.47, 3.71, text=r'$\mathbf{1} + \Phi$', text_pos='right')

# Output Gate A_mod
draw_conn(ax, 11.47, 4.10, 11.47, 4.35)
draw_block(ax, 10.75, 4.35, 1.20, 0.75,
           r'Gate $\widetilde{A}(H)$',
           r'$A_{sym} \odot (\mathbf{1} + \Phi)$',
           '[B, 48, H, W]', color='#FDF2F8', border='#DB2777', title_size=7.8, sub_size=6.6, dim_size=6.2)

# Real Attention Map Heatmap (att1 from CAD-SPAN)
att_img_path = os.path.join(ASSETS_DIR, 'diag_att_map.png')
if os.path.exists(att_img_path):
    att_pil = Image.open(att_img_path)
    ax.imshow(att_pil, extent=[12.10, 12.80, 3.65, 4.75], aspect='auto', zorder=4)
    rect_att = Rectangle((12.10, 3.65), 0.70, 1.10, linewidth=1.2, edgecolor='#DB2777', facecolor='none', zorder=5)
    ax.add_patch(rect_att)
    ax.text(12.45, 3.42, r'$\mathrm{Att.Map}$' + '\n' + r'$\widetilde{A}(H)$', ha='center', va='top', fontsize=6.8, fontweight='bold', color='#9D174D', zorder=5)
    ax.annotate('', xy=(12.10, 4.45), xytext=(11.95, 4.45), arrowprops=dict(arrowstyle='->', color='#DB2777', lw=1.2, linestyle='--'), zorder=5)

# 4. FINAL HADAMARD CONTENT GATING (odot): R * A_mod
draw_op(ax, 11.47, 6.25, r'$\odot$', r=0.20, color='#DCFCE7', border='#16A34A', text_color='#15803D', size=13)
draw_conn(ax, 10.10, 6.25, 11.27, 6.25, text=r'$R$')
draw_conn(ax, 11.47, 5.10, 11.47, 6.05, text=r'$\widetilde{A}$')

# Output to Final Feature Map Heatmap
draw_conn(ax, 11.67, 6.25, 12.05, 6.25)

# Real Output Feature Map Heatmap (Feat. Map 3 from CAD-SPAN)
hm3_path = os.path.join(ASSETS_DIR, 'diag_feat_map3.png')
if os.path.exists(hm3_path):
    hm3_pil = Image.open(hm3_path)
    ax.imshow(hm3_pil, extent=[12.05, 12.85, 5.65, 6.85], aspect='auto', zorder=4)
    rect_hm3 = Rectangle((12.05, 5.65), 0.80, 1.20, linewidth=1.4, edgecolor='#16A34A', facecolor='none', zorder=5)
    ax.add_patch(rect_hm3)
    ax.text(12.45, 5.48, r'$\mathrm{Feat.Map}_3$' + '\n' + r'$O \in \mathbb{R}^{B \times 48 \times H \times W}$',
            ha='center', va='top', fontsize=6.8, fontweight='bold', color='#14532D', zorder=5)

# Mathematical Identity Fallback Banner (Bottom of Panel b)
fb_patch = FancyBboxPatch((0.45, 0.35), 12.40, 0.28, boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=1.2, edgecolor='#10B981', facecolor='#ECFDF5', zorder=3)
ax.add_patch(fb_patch)
ax.text(6.65, 0.49, r'Exact Mathematical Fallback: When $\gamma_s = \gamma_l = 0$ and $\tau = 0.50$, $\Phi(H) \equiv 0$ and $\widetilde{A}(H) \equiv \sigma(H) - 0.5$ (Exact Baseline SPAB)',
        fontsize=7.6, fontweight='bold', color='#065F46', ha='center', va='center', zorder=4)


# =========================================================================
# PANEL (C): CONV3XC STRUCTURAL RE-PARAMETERIZATION MECHANISM
# Training Multi-Branch vs Hardware-Optimized Single Conv3x3
# =========================================================================
draw_panel(ax, 13.35, 0.25, 6.35, 7.45,
           '(c) Conv3XC: Structural Re-param.', 'Multi-Branch -> Single 3x3')

# Training Phase Sub-box
t_box = FancyBboxPatch((13.50, 4.10), 6.05, 3.10, boxstyle="round,pad=0.03,rounding_size=0.10",
                       linewidth=1.4, edgecolor='#6366F1', facecolor='#F5F3FF', zorder=2)
ax.add_patch(t_box)
ax.text(13.65, 7.00, 'TRAINING PHASE (High-Capacity Linear Multi-Branch)', fontsize=8.2, fontweight='bold', color='#4338CA', zorder=3)
ax.text(13.65, 6.78, 'Strictly Linear: Zero non-linearities between layers allows associative fusion', fontsize=6.6, color='#4F46E5', zorder=3)

# Input X
draw_block(ax, 13.65, 5.20, 0.75, 0.85, 'Input', '$C_{in}$', color='#FFFFFF', border=C_TENSOR_BORDER, title_size=8.0, sub_size=7.0)

# Branch 1: Shortcut (1x1 Conv)
ax.plot([14.40, 14.65, 14.65, 18.35, 18.35], [5.62, 5.62, 6.45, 6.45, 5.85], color='#4F46E5', lw=1.4, linestyle='--', zorder=3)
draw_block(ax, 15.20, 6.20, 2.45, 0.50, r'Shortcut: Conv $1\times 1$ ($C_{in} \to C_{out}$)', '', color='#EEF2FF', border='#4F46E5', title_size=7.2)

# Branch 2: Bottleneck (1x1 -> 3x3 -> 1x1)
draw_conn(ax, 14.40, 5.62, 14.65, 5.62)
draw_block(ax, 14.65, 5.20, 0.95, 0.85, r'Conv $1\times 1$', '$C_{in}\to 2C$', color=C_CONV_FRONT, border=C_CONV_EDGE, title_size=7.4, sub_size=6.6)

draw_conn(ax, 15.60, 5.62, 15.85, 5.62)
draw_block(ax, 15.85, 5.20, 0.95, 0.85, r'Conv $3\times 3$', '$2C\to 2C$', color=C_CONV_FRONT, border=C_CONV_EDGE, title_size=7.4, sub_size=6.6)

draw_conn(ax, 16.80, 5.62, 17.05, 5.62)
draw_block(ax, 17.05, 5.20, 0.95, 0.85, r'Conv $1\times 1$', '$2C\to C_{out}$', color=C_CONV_FRONT, border=C_CONV_EDGE, title_size=7.4, sub_size=6.6)

# Addition (+) of Bottleneck + Shortcut
draw_op(ax, 18.35, 5.62, r'$\oplus$', r=0.18, color='#DBEAFE', border='#2563EB', text_color='#1D4ED8', size=11)
draw_conn(ax, 18.00, 5.62, 18.17, 5.62)
draw_conn(ax, 18.53, 5.62, 18.75, 5.62)
draw_block(ax, 18.75, 5.20, 0.70, 0.85, 'Train\nOutput', '$C_{out}$', color='#EDE9FE', border='#6366F1', title_size=7.6, sub_size=6.8)

# Offline Exact Tensor Fusion Banner
ax.annotate('', xy=(16.52, 3.48), xytext=(16.52, 3.98),
            arrowprops=dict(arrowstyle='->', color='#B91C1C', lw=2.2), zorder=4)

fuse_box = FancyBboxPatch((13.50, 3.52), 6.05, 0.50, boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=1.2, edgecolor='#EF4444', facecolor='#FEE2E2', zorder=4)
ax.add_patch(fuse_box)
ax.text(16.52, 3.86, r'Offline Kernel Fusion: $\mathbf{W}_{eval} = \mathbf{W}_3 * \mathbf{W}_2 * \mathbf{W}_1 + \mathrm{Pad}_{3\times 3}(\mathbf{W}_{sk})$',
        ha='center', va='center', fontsize=7.2, fontweight='bold', color='#B91C1C', zorder=5)
ax.text(16.52, 3.66, r'Offline Bias Fusion: $\mathbf{B}_{eval} = \mathbf{W}_3 * (\mathbf{W}_2 * \mathbf{B}_1 + \mathbf{B}_2) + \mathbf{B}_3 + \mathbf{B}_{sk}$',
        ha='center', va='center', fontsize=7.0, color='#991B1B', zorder=5)

# Inference Phase Sub-box
i_box = FancyBboxPatch((13.50, 0.45), 6.05, 2.90, boxstyle="round,pad=0.03,rounding_size=0.10",
                       linewidth=1.4, edgecolor='#16A34A', facecolor='#F0FDF4', zorder=2)
ax.add_patch(i_box)
ax.text(13.65, 3.12, 'INFERENCE PHASE (Hardware-Optimized Deployment)', fontsize=8.2, fontweight='bold', color='#15803D', zorder=3)
ax.text(13.65, 2.92, 'Zero Multi-Branch Overhead · 100% Cache Locality · Max Edge FPS', fontsize=6.6, color='#16A34A', zorder=3)

# Inference Single Equivalent Conv 3x3
draw_block(ax, 13.65, 1.45, 0.75, 0.85, 'Input', '$C_{in}$', color='#FFFFFF', border=C_TENSOR_BORDER, title_size=8.0, sub_size=7.0)
draw_conn(ax, 14.40, 1.87, 14.85, 1.87)

draw_block(ax, 14.85, 1.25, 3.40, 1.25,
           r'Single Equivalent Standard Conv $3\times 3$',
           r'Kernel: $\mathbf{W}_{eval} \in \mathbb{R}^{C_{out} \times C_{in} \times 3 \times 3}$' + '\n' +
           r'Bias: $\mathbf{B}_{eval} \in \mathbb{R}^{C_{out}}$',
           'Single-layer execution on Mobile/NPU/GPU',
           color='#DCFCE7', border='#16A34A', title_size=8.2, sub_size=7.0, dim_size=6.6)

draw_conn(ax, 18.25, 1.87, 18.70, 1.87)
draw_block(ax, 18.70, 1.45, 0.75, 0.85, 'Output', '$C_{out}$', color='#DCFCE7', border='#16A34A', title_size=8.0, sub_size=7.0)

ax.text(16.52, 0.65, r'Mathematical Identity: $\mathrm{Conv3XC}(X) \equiv \mathrm{Conv2d}(X, \mathbf{W}_{eval}, \mathbf{B}_{eval})$',
        ha='center', va='center', fontsize=7.2, fontweight='bold', color='#14532D', zorder=3)

out_arch = os.path.join(BASE_DIR, "methodology_architecture.png")
plt.savefig(out_arch, bbox_inches='tight', dpi=300)
plt.close()
print(f"Flawless, publication-grade Architecture Diagram saved to {out_arch}")
