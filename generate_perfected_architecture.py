import os
os.environ['MPLCONFIGDIR'] = r'C:\Users\anike\.gemini\antigravity\scratch\mpl_cache'

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

# Font styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

# Canvas setup: ultra high-res 20 x 13.5 inches at 300 DPI (6000 x 4050 px)
fig = plt.figure(figsize=(20, 13.5), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 20)
ax.set_ylim(0, 13.5)
ax.axis('off')

# Publication Color Palette matching authentic SPAN paper + modern IEEE/CVF standards
C_PANEL_BG = '#F8FAFC'
C_PANEL_BORDER = '#94A3B8'
C_HEADER_BG = '#0F172A'

C_CONV = '#ECFDF5'          # Light emerald green (Conv3 / Conv3XC)
C_CONV_BORDER = '#059669'   # Emerald 600
C_SPAB = '#EFF6FF'          # Light blue (CAD-SPAB stages)
C_SPAB_BORDER = '#2563EB'   # Blue 600
C_ACT = '#F1F5F9'           # Slate 100 for activation layers
C_ACT_BORDER = '#64748B'    # Slate 500
C_SIGMA_A = '#FFF1F2'       # Rose/pink for parameter-free attention
C_SIGMA_A_BORDER = '#E11D48'# Rose 600
C_CONCAT = '#FEF3C7'        # Warm amber for concat layer
C_CONCAT_BORDER = '#D97706' # Amber 600
C_PIXELSHUFFLE = '#FEF9C3'  # Yellow for PixelShuffle
C_PIXELSHUFFLE_BORDER = '#CA8A04'
C_TENSOR = '#FFFFFF'
C_TENSOR_BORDER = '#475569'
C_DIFF = '#F5F3FF'          # Purple/violet for differential priors
C_DIFF_BORDER = '#7C3AED'

def draw_panel(ax, x, y, w, h, title, subtitle=''):
    panel = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.18",
                           linewidth=1.8, edgecolor=C_PANEL_BORDER, facecolor=C_PANEL_BG, zorder=1)
    ax.add_patch(panel)
    hbar = FancyBboxPatch((x, y + h - 0.58), w, 0.58, boxstyle="round,pad=0.02,rounding_size=0.15",
                          linewidth=0, facecolor=C_HEADER_BG, zorder=2)
    ax.add_patch(hbar)
    ax.text(x + 0.30, y + h - 0.29, title, ha='left', va='center',
            fontsize=10.8, fontweight='bold', color='#FFFFFF', zorder=3)
    if subtitle:
        ax.text(x + w - 0.30, y + h - 0.29, subtitle, ha='right', va='center',
                fontsize=8.5, fontfamily='monospace', color='#94A3B8', zorder=3)

def draw_block(ax, x, y, w, h, title, sub='', dim='', color='#FFFFFF', border='#64748B', lw=1.5,
               title_size=9.0, sub_size=7.5, dim_size=7.0, title_color='#0F172A', zorder=3):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.10",
                         linewidth=lw, edgecolor=border, facecolor=color, zorder=zorder)
    ax.add_patch(box)
    cy = y + h/2
    if dim and sub:
        ax.text(x + w/2, cy + 0.24, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=title_color, zorder=zorder+1)
        ax.text(x + w/2, cy + 0.01, sub, ha='center', va='center', fontsize=sub_size, color='#334155', zorder=zorder+1)
        ax.text(x + w/2, cy - 0.24, dim, ha='center', va='center', fontsize=dim_size, fontfamily='monospace', fontweight='bold', color='#1D4ED8', zorder=zorder+1)
    elif sub:
        ax.text(x + w/2, cy + 0.14, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=title_color, zorder=zorder+1)
        ax.text(x + w/2, cy - 0.16, sub, ha='center', va='center', fontsize=sub_size, color='#334155', zorder=zorder+1)
    elif dim:
        ax.text(x + w/2, cy + 0.14, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=title_color, zorder=zorder+1)
        ax.text(x + w/2, cy - 0.16, dim, ha='center', va='center', fontsize=dim_size, fontfamily='monospace', fontweight='bold', color='#1D4ED8', zorder=zorder+1)
    else:
        ax.text(x + w/2, cy, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=title_color, zorder=zorder+1)

def draw_op(ax, x, y, symbol, r=0.20, color='#E0F2FE', border='#0284C7', text_color='#0369A1', size=13):
    circle = Circle((x, y), r, facecolor=color, edgecolor=border, linewidth=1.6, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, symbol, ha='center', va='center', fontsize=size, fontweight='bold', color=text_color, zorder=6)

def draw_conn(ax, x1, y1, x2, y2, color='#334155', lw=1.5, text='', text_pos='top', style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=1.5, shrinkB=1.5), zorder=4)
    if text:
        tx = (x1 + x2)/2
        ty = (y1 + y2)/2 + (0.15 if text_pos=='top' else -0.17)
        ax.text(tx, ty, text, ha='center', va='center', fontsize=7.8, color='#0F172A', fontweight='bold', zorder=5)

def draw_kernel_grid(ax, x, y, matrix, label, factor_label='', cell_size=0.17, font_size=6.2):
    title = f"{factor_label} {label}".strip() if factor_label else label
    ax.text(x + 1.5 * cell_size, y + 3 * cell_size + 0.08, title, ha='center', va='bottom',
            fontsize=7.6, fontweight='bold', color='#92400E')
    for r in range(3):
        for c in range(3):
            val = matrix[r][c]
            bg = '#FEF3C7' if val != 0 else '#FFFFFF'
            rect = patches.Rectangle((x + c * cell_size, y + (2 - r) * cell_size), cell_size, cell_size,
                                     facecolor=bg, edgecolor='#D97706', lw=0.8, zorder=4)
            ax.add_patch(rect)
            ax.text(x + (c + 0.5) * cell_size, y + (2.5 - r) * cell_size, str(val),
                    ha='center', va='center', fontsize=font_size, fontweight='bold', color='#78350F', zorder=5)

# =========================================================================
# PANEL (A): GLOBAL MACRO-ARCHITECTURE OF CAD-SPAN
# Strictly faithful to Figure 2 of original SPAN paper (Hongyuan Yu et al., CVPRW 2024)
# =========================================================================
draw_panel(ax, 0.25, 7.75, 19.50, 5.50,
           '(a) Global Macro-Architecture of Proposed CAD-SPAN Framework',
           'Input LR -> Shallow Conv3XC -> 6 Cascaded CAD-SPABs -> Refine Conv -> Concat -> PixelShuffle -> HR Output')

# 1. LR Input Image Tensor
draw_block(ax, 0.45, 8.85, 1.30, 1.45, 'LR Input', r'$I_{LR} \in \mathbb{R}^{3 \times H \times W}$',
           '[B, 3, H, W]', color=C_TENSOR, border=C_TENSOR_BORDER, title_size=9.5, sub_size=8.2, dim_size=7.6)

# Mean-shift normalization annotation
ax.text(1.10, 8.55, 'Mean Shift:\n$(I_{LR} - \mu) \\times 255$', ha='center', va='top', fontsize=7.2, color='#475569')

# 2. Shallow Conv3XC (conv_1 in code: 3 -> 48)
draw_conn(ax, 1.75, 9.57, 2.15, 9.57)
draw_block(ax, 2.15, 8.85, 1.35, 1.45, 'Shallow Conv', 'Conv3XC (3 to 48)\n(conv_1)',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=9.0, sub_size=7.6, dim_size=7.2)

# Output of conv_1 is Feat.Map 1 (F0)
draw_conn(ax, 3.50, 9.57, 3.95, 9.57)
ax.text(3.72, 9.78, r'$F_0$', ha='center', va='bottom', fontsize=8.6, fontweight='bold', color='#047857')

# 3. Cascaded Blocks: CAD-SPAB #1 through #6
# CAD-SPAB #1
draw_block(ax, 3.95, 8.85, 1.25, 1.45, 'CAD-SPAB #1', 'Stage 1 Block', '[B, 48, H, W]',
           color=C_SPAB, border=C_SPAB_BORDER, title_size=9.0, sub_size=7.6, dim_size=7.2)

draw_conn(ax, 5.20, 9.57, 5.65, 9.57)
ax.text(5.42, 9.78, r'$F_{b1}$', ha='center', va='bottom', fontsize=8.6, fontweight='bold', color='#1D4ED8')

# CAD-SPAB #2
draw_block(ax, 5.65, 8.85, 1.25, 1.45, 'CAD-SPAB #2', 'Stage 2 Block', '[B, 48, H, W]',
           color=C_SPAB, border=C_SPAB_BORDER, title_size=9.0, sub_size=7.6, dim_size=7.2)

# Cascaded ellipsis connection (CAD-SPAB #3 and #4 in series)
ax.plot([6.90, 7.50], [9.57, 9.57], color='#2563EB', lw=2.0, linestyle='--', zorder=4)
ax.text(7.20, 9.78, r'$\cdots$', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#1D4ED8')
ax.text(7.20, 9.20, 'Stages #3, #4\nin series', ha='center', va='top', fontsize=6.8, color='#64748B')

draw_conn(ax, 7.50, 9.57, 7.90, 9.57)

# CAD-SPAB #5
draw_block(ax, 7.90, 8.85, 1.25, 1.45, 'CAD-SPAB #5', 'Stage 5 Block', '[B, 48, H, W]',
           color=C_SPAB, border=C_SPAB_BORDER, title_size=9.0, sub_size=7.6, dim_size=7.2)

draw_conn(ax, 9.15, 9.57, 9.55, 9.57)
ax.text(9.35, 9.78, r'$F_{b5}$', ha='center', va='bottom', fontsize=8.6, fontweight='bold', color='#B45309')

# CAD-SPAB #6
draw_block(ax, 9.55, 8.85, 1.25, 1.45, 'CAD-SPAB #6', 'Stage 6 Block', '[B, 48, H, W]',
           color=C_SPAB, border=C_SPAB_BORDER, title_size=9.0, sub_size=7.6, dim_size=7.2)

# 4. Refinement Conv3XC Layer (conv_2 in code: 48 -> 48) - EXACTLY AS IN SPAN!
draw_conn(ax, 10.80, 9.57, 11.20, 9.57)
ax.text(11.00, 9.78, r'$F_{b6}$', ha='center', va='bottom', fontsize=8.6, fontweight='bold', color='#1D4ED8')

draw_block(ax, 11.20, 8.85, 1.35, 1.45, 'Refine Conv', 'Conv3XC (48 to 48)\n(conv_2 in code)',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=9.0, sub_size=7.6, dim_size=7.2)

# 5. ALL 4 HIERARCHICAL SKIP CONNECTIONS ROUTED OVER THE TOP (Exact match to Figure 2)
# Track 1: Feat.Map 1 (F0) from after conv_1
ax.plot([3.72, 3.72, 13.05], [9.57, 12.25, 12.25], color='#059669', lw=1.6, zorder=3)
ax.plot([13.05, 13.05], [12.25, 10.45], color='#059669', lw=1.6, zorder=3)
ax.annotate('', xy=(13.05, 10.30), xytext=(13.05, 10.45), arrowprops=dict(arrowstyle='->', color='#059669', lw=1.6), zorder=4)
ax.text(5.20, 12.38, r'$\mathrm{Feat.Map}_1$ ($F_0$, Shallow Features)', fontsize=7.8, color='#047857', fontweight='bold')

# Track 2: Feat.Map 2 (Fb1) from after CAD-SPAB #1
ax.plot([5.42, 5.42, 13.35], [9.57, 11.80, 11.80], color='#2563EB', lw=1.6, zorder=3)
ax.plot([13.35, 13.35], [11.80, 10.45], color='#2563EB', lw=1.6, zorder=3)
ax.annotate('', xy=(13.35, 10.30), xytext=(13.35, 10.45), arrowprops=dict(arrowstyle='->', color='#2563EB', lw=1.6), zorder=4)
ax.text(6.30, 11.93, r'$\mathrm{Feat.Map}_2$ ($F_{b1}$, Early Stage Details)', fontsize=7.8, color='#1D4ED8', fontweight='bold')

# Track 3: Feat.Map 6 (Fb5 / out_b5_2) intermediate tap from Stage 5 / Block 6
ax.plot([9.35, 9.35, 13.65], [9.57, 11.35, 11.35], color='#B45309', lw=1.6, zorder=3)
ax.plot([13.65, 13.65], [11.35, 10.45], color='#B45309', lw=1.6, zorder=3)
ax.annotate('', xy=(13.65, 10.30), xytext=(13.65, 10.45), arrowprops=dict(arrowstyle='->', color='#B45309', lw=1.6), zorder=4)
ax.text(9.90, 11.48, r'$\mathrm{Feat.Map}_6$ ($F_{b5}$, Deep Residual)', fontsize=7.8, color='#92400E', fontweight='bold')

# Track 4: Feat.Map 7 (Fb6_conv) from refine conv_2 enters Concat horizontally from left
draw_conn(ax, 12.55, 9.57, 12.90, 9.57)
ax.text(12.72, 9.85, r'$\mathrm{Feat.Map}_7$', ha='center', va='bottom', fontsize=7.6, fontweight='bold', color='#059669')
ax.text(12.72, 9.38, r'($F_{b6\_conv}$)', ha='center', va='top', fontsize=6.8, color='#047857')

# 6. Concatenation Layer (torch.cat([F0, Fb6_conv, Fb1, Fb5_2], dim=1))
draw_block(ax, 12.90, 8.85, 1.85, 1.45, 'Concat. Layer',
           r'$[F_0, F_{b6\_conv}, F_{b1}, F_{b5}]$' + '\nChannel Concatenation',
           r'$[B, 192, H, W]$', color=C_CONCAT, border=C_CONCAT_BORDER, title_size=9.0, sub_size=7.6, dim_size=7.2)

# 7. 1x1 Conv Compression Layer (conv_cat in code: 192 -> 48)
draw_conn(ax, 14.75, 9.57, 15.15, 9.57)
draw_block(ax, 15.15, 8.85, 1.35, 1.45, r'Conv $1\times 1$',
           'conv_cat (192 to 48)\nLinear Channel Fusion',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=9.0, sub_size=7.6, dim_size=7.2)

# 8. PixelShuffle Sub-Pixel Upsampler (upsampler in code: 48 -> 48 -> PixelShuffle x4 -> 3)
draw_conn(ax, 16.50, 9.57, 16.90, 9.57)
draw_block(ax, 16.90, 8.85, 1.45, 1.45, 'PixelShuffle',
           r'Conv $3\times 3$ ($48\to 48$)' + '\n' + r'Sub-Pixel ($\times 4$)',
           r'$48 \to 3 \times 4^2 \to 3$', color=C_PIXELSHUFFLE, border=C_PIXELSHUFFLE_BORDER,
           title_size=9.2, sub_size=7.6, dim_size=7.2)

# 9. High-Resolution Output Image Tensor
draw_conn(ax, 18.35, 9.57, 18.65, 9.57)
draw_block(ax, 18.65, 8.85, 0.95, 1.45, 'HR Output', r'$I_{SR}$', r'$[B, 3, 4H, 4W]$',
           color='#DCFCE7', border='#16A34A', lw=1.8, title_size=9.0, sub_size=8.2, dim_size=6.8, title_color='#14532D')

# Component Legend Box (Upper right of Panel a)
leg_box = FancyBboxPatch((14.60, 11.00), 4.90, 1.70, boxstyle="round,pad=0.03,rounding_size=0.08",
                         linewidth=1.2, edgecolor='#CBD5E1', facecolor='#FFFFFF', zorder=2)
ax.add_patch(leg_box)
ax.text(14.75, 12.45, 'COMPONENT COLOR LEGEND', fontsize=7.8, fontweight='bold', color='#334155')

def draw_legend_item(ax, x, y, color, border, text):
    p = FancyBboxPatch((x, y), 0.35, 0.20, boxstyle="round,pad=0.02,rounding_size=0.04",
                       linewidth=1.0, edgecolor=border, facecolor=color, zorder=3)
    ax.add_patch(p)
    ax.text(x + 0.44, y + 0.10, text, ha='left', va='center', fontsize=7.0, color='#1E293B', fontweight='bold')

draw_legend_item(ax, 14.75, 12.05, C_CONV, C_CONV_BORDER, 'Conv3XC / Standard Conv')
draw_legend_item(ax, 14.75, 11.70, C_SPAB, C_SPAB_BORDER, 'CAD-SPAB Module')
draw_legend_item(ax, 14.75, 11.35, C_CONCAT, C_CONCAT_BORDER, 'Concat Layer (192 channels)')
draw_legend_item(ax, 17.20, 12.05, C_PIXELSHUFFLE, C_PIXELSHUFFLE_BORDER, 'PixelShuffle Upsampler')
draw_legend_item(ax, 17.20, 11.70, C_SIGMA_A, C_SIGMA_A_BORDER, 'Origin-Symmetric Gate')
draw_legend_item(ax, 17.20, 11.35, C_DIFF, C_DIFF_BORDER, 'Dual-Differential Modulator')


# =========================================================================
# PANEL (B): MICRO-ARCHITECTURE OF CAD-SPAB BLOCK
# Mathematically & logically rigorous:
# Content residual (H + Xin) modulated by modulated attention gate A_mod(H)
# =========================================================================
draw_panel(ax, 0.25, 0.25, 12.90, 7.30,
           '(b) CAD-SPAB Micro-Architecture: Differential-Geometric Attention Block',
           'Content Residual (H + Xin) modulated by Curvature-Aware Gate A_mod')

# Block Input Xin
draw_block(ax, 0.45, 4.25, 1.15, 1.35, 'Block Input', r'$X_{in}$',
           '[B, 48, H, W]', color=C_TENSOR, border=C_TENSOR_BORDER, title_size=9.2, sub_size=8.5, dim_size=7.6)

# 1. Main Convolutional Trunk: 3x Conv3XC with interleaved SiLU non-linearities
# Conv3XC 1
draw_conn(ax, 1.60, 4.92, 1.90, 4.92)
draw_block(ax, 1.90, 4.30, 1.10, 1.25, r'$\mathrm{Conv3XC}_1$', 'Linear Cascade\n(48 to 48)',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=8.6, sub_size=7.0, dim_size=6.8)

# SiLU Activation 1
draw_conn(ax, 3.00, 4.92, 3.25, 4.92)
draw_block(ax, 3.25, 4.50, 0.55, 0.85, r'$\sigma$', 'SiLU', '', color=C_ACT, border=C_ACT_BORDER, title_size=8.5, sub_size=7.0)

# Conv3XC 2
draw_conn(ax, 3.80, 4.92, 4.05, 4.92)
draw_block(ax, 4.05, 4.30, 1.10, 1.25, r'$\mathrm{Conv3XC}_2$', 'Linear Cascade\n(48 to 48)',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=8.6, sub_size=7.0, dim_size=6.8)

# SiLU Activation 2
draw_conn(ax, 5.15, 4.92, 5.40, 4.92)
draw_block(ax, 5.40, 4.50, 0.55, 0.85, r'$\sigma$', 'SiLU', '', color=C_ACT, border=C_ACT_BORDER, title_size=8.5, sub_size=7.0)

# Conv3XC 3 (No activation afterwards!)
draw_conn(ax, 5.95, 4.92, 6.20, 4.92)
draw_block(ax, 6.20, 4.30, 1.10, 1.25, r'$\mathrm{Conv3XC}_3$', 'No Activation\n(48 to 48)',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=8.6, sub_size=7.0, dim_size=6.8)

# Output of Conv3 is Feature H
draw_conn(ax, 7.30, 4.92, 7.60, 4.92)
draw_block(ax, 7.60, 4.47, 1.00, 0.90, r'Feature $H$', r'$H \equiv \mathrm{out}_3$',
           '[B, 48, H, W]', color='#FEF3C7', border='#D97706', title_size=8.8, sub_size=7.6, dim_size=7.0)

# 2. Local Identity Shortcut & Residual Addition (H + Xin)
# Xin travels across the top to the Addition node (+)
draw_op(ax, 9.20, 6.25, r'$\oplus$', r=0.22, color='#DBEAFE', border='#2563EB', text_color='#1D4ED8', size=13)
# Xin line placed cleanly at y = 6.45
ax.plot([1.02, 1.02, 9.20], [5.60, 6.45, 6.45], color='#2563EB', lw=1.6, zorder=3)
draw_conn(ax, 9.20, 6.45, 9.20, 6.47)
# Label placed at y = 6.66 with comfortable clearance below panel header
ax.text(3.50, 6.66, r'Local Identity Shortcut: $X_{in} \in \mathbb{R}^{B \times 48 \times H \times W}$',
        fontsize=8.0, color='#1D4ED8', fontweight='bold', va='bottom')

# H branch to Addition node (+)
ax.plot([8.10, 8.10, 8.98], [5.37, 6.25, 6.25], color='#D97706', lw=1.5, zorder=3)
draw_conn(ax, 8.98, 6.25, 8.98, 6.25)
ax.text(8.50, 6.38, r'$H$', fontsize=8.2, color='#B45309', fontweight='bold')

# Residual Sum Result: (H + Xin)
draw_conn(ax, 9.42, 6.25, 9.75, 6.25)
draw_block(ax, 9.75, 5.80, 1.45, 0.90, r'Content Residual', r'$R = H + X_{in}$',
           '[B, 48, H, W]', color='#EFF6FF', border='#3B82F6', title_size=8.8, sub_size=7.6, dim_size=7.0)

# 3. Novel Dual-Differential Attention Generator (Feature H splits into 3 streams)
# Main vertical distribution bus from H
ax.plot([8.10, 8.10], [4.47, 1.15], color='#B45309', lw=1.6, zorder=3)

# Stream 1: Origin-Symmetric Hyperbolic Tangent Gate (Top lane)
draw_conn(ax, 8.10, 4.05, 8.55, 4.05)
draw_block(ax, 8.55, 3.65, 2.75, 0.85,
           '1. Calibrated Origin-Symmetric Gate',
           r'$A_{sym}(H) = \frac{1}{2}\tanh(\tau \cdot H)$  $(\tau^* = 0.502)$' + '\n' +
           r'Exact odd symmetry: $A_{sym}(-H) = -A_{sym}(H)$',
           'Range: $(-0.5, 0.5)$',
           color=C_SIGMA_A, border=C_SIGMA_A_BORDER, title_size=8.2, sub_size=7.0, dim_size=6.6)

# Stream 2: First-Order Directional Sobel Gradient Flux Operator (Middle lane)
draw_conn(ax, 8.10, 2.55, 8.55, 2.55)
box2 = FancyBboxPatch((8.55, 1.85), 2.85, 1.40, boxstyle="round,pad=0.03,rounding_size=0.10",
                      linewidth=1.4, edgecolor='#D97706', facecolor='#FFFBEB', zorder=3)
ax.add_patch(box2)
ax.text(8.68, 3.06, '2. Directional Gradient Flux Operator', ha='left', va='center', fontsize=8.2, fontweight='bold', color='#0F172A', zorder=4)
ax.text(8.68, 2.80, r'$G_x = \frac{1}{8}K_x *_{dw} H, \quad G_y = \frac{1}{8}K_y *_{dw} H$', ha='left', va='center', fontsize=7.3, color='#334155', zorder=4)
ax.text(8.68, 2.54, r'$\|\nabla H\|_2 = \sqrt{G_x^2 + G_y^2 + \epsilon}$', ha='left', va='center', fontsize=7.3, color='#334155', zorder=4)
ax.text(8.68, 2.30, 'Directional Step Boundary Flux', ha='left', va='center', fontsize=6.6, color='#475569', zorder=4)
ax.text(8.68, 2.06, '[B, 48, H, W]', ha='left', va='center', fontsize=6.4, fontfamily='monospace', fontweight='bold', color='#1D4ED8', zorder=4)

# Normalized Sobel Kernel Grids
sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
draw_kernel_grid(ax, 10.25, 1.92, sobel_x, r'$K_x$', factor_label=r'$\frac{1}{8}$', cell_size=0.14, font_size=5.0)
draw_kernel_grid(ax, 10.82, 1.92, sobel_y, r'$K_y$', factor_label=r'$\frac{1}{8}$', cell_size=0.14, font_size=5.0)

# Stream 3: Second-Order Isotropic Laplacian Curvature Operator (Bottom lane)
draw_conn(ax, 8.10, 1.15, 8.55, 1.15)
box3 = FancyBboxPatch((8.55, 0.60), 2.85, 1.15, boxstyle="round,pad=0.03,rounding_size=0.10",
                      linewidth=1.4, edgecolor='#16A34A', facecolor='#F0FDF4', zorder=3)
ax.add_patch(box3)
ax.text(8.68, 1.56, '3. Isotropic Curvature Operator', ha='left', va='center', fontsize=8.2, fontweight='bold', color='#0F172A', zorder=4)
ax.text(8.68, 1.32, r'$\Delta H = |\nabla^2 H| = |K_L *_{dw} H|$', ha='left', va='center', fontsize=7.3, color='#334155', zorder=4)
ax.text(8.68, 1.08, 'Corner & Ridge Extrema Detector', ha='left', va='center', fontsize=6.6, color='#475569', zorder=4)
ax.text(8.68, 0.86, 'Isotropic 8-Neighbor Stencil', ha='left', va='center', fontsize=6.6, color='#475569', zorder=4)
ax.text(8.68, 0.66, '[B, 48, H, W]', ha='left', va='center', fontsize=6.4, fontfamily='monospace', fontweight='bold', color='#1D4ED8', zorder=4)

laplacian_k = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
draw_kernel_grid(ax, 10.55, 0.66, laplacian_k, r'$K_L$', cell_size=0.14, font_size=5.0)

# Fusion of Streams 2 & 3 into Dual-Differential Prior Phi(H)
ax.plot([11.40, 11.55], [2.55, 1.85], color='#334155', lw=1.5)
ax.plot([11.40, 11.55], [1.15, 1.85], color='#334155', lw=1.5)
draw_conn(ax, 11.55, 1.85, 11.65, 1.85)

draw_block(ax, 11.65, 1.25, 1.15, 1.20,
           r'Prior $\Phi(H)$',
           r'$\Phi = \tanh($' + '\n' + r'$\gamma_s \|\nabla H\| +$' + '\n' + r'$\gamma_l \Delta H)$',
           r'$\gamma_s^*=0.008$' + '\n' + r'$\gamma_l^*=0.001$',
           color=C_DIFF, border=C_DIFF_BORDER, title_size=8.0, sub_size=6.4, dim_size=6.0)

# Affine Modulation Node: A_mod = A_sym * (1 + Phi)
# Phi travels up to affine scaling node (+ 1)
ax.plot([12.22, 12.22], [2.45, 2.85], color='#7C3AED', lw=1.5, zorder=3)
draw_op(ax, 12.22, 3.08, r'$\oplus$', r=0.18, color='#F3E8FF', border='#7C3AED', text_color='#6D28D9', size=11)
ax.text(12.46, 3.08, r'$\mathbf{1}$', fontsize=8.8, fontweight='bold', color='#6D28D9', va='center')

# (1 + Phi) meets A_sym at Hadamard Product Node (odot)
draw_op(ax, 12.22, 4.05, r'$\odot$', r=0.20, color='#FCE7F3', border='#EC4899', text_color='#BE185D', size=12)
draw_conn(ax, 11.30, 4.05, 12.02, 4.05, text=r'$A_{sym}$')
draw_conn(ax, 12.22, 3.26, 12.22, 3.85, text=r'$\mathbf{1} + \Phi$', text_pos='right')

# Output of Attention Gate Modulation: A_mod
draw_conn(ax, 12.22, 4.25, 12.22, 4.60)
draw_block(ax, 11.60, 4.60, 1.25, 0.80,
           r'Gate $\widetilde{A}(H)$',
           r'$A_{sym} \odot (\mathbf{1} + \Phi)$',
           '[B, 48, H, W]', color='#FDF2F8', border='#DB2777', title_size=8.0, sub_size=6.8, dim_size=6.4)

# 4. FINAL CONTENT GATING NODE (odot):
# Content Residual (H + Xin) meets Modulated Attention Gate A_mod!
draw_op(ax, 12.22, 6.25, r'$\odot$', r=0.22, color='#DCFCE7', border='#16A34A', text_color='#15803D', size=13)
# Connection from Content Residual (R = H + Xin) to gating node
draw_conn(ax, 11.20, 6.25, 12.00, 6.25, text=r'$R$')
# Connection from Gate A_mod to gating node
draw_conn(ax, 12.22, 5.40, 12.22, 6.03, text=r'$\widetilde{A}$')

# Final Block Output O
draw_conn(ax, 12.44, 6.25, 12.55, 6.25)
draw_block(ax, 12.55, 5.75, 0.50, 1.00, 'Block\nOutput', r'$O$', '',
           color='#F0FDF4', border='#15803D', lw=1.8, title_size=7.6, sub_size=8.2, title_color='#14532D')

# Mathematical Identity Fallback Banner (Bottom of Panel b)
fb_patch = FancyBboxPatch((0.45, 0.30), 12.40, 0.26, boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=1.2, edgecolor='#10B981', facecolor='#ECFDF5', zorder=3)
ax.add_patch(fb_patch)
ax.text(6.65, 0.43, r'Exact Mathematical Identity Fallback: When $\gamma_s = \gamma_l = 0$ and $\tau = 0.50$, $\Phi(H) \equiv 0$ and $\widetilde{A}(H) \equiv \sigma(H) - 0.5$ (Exact Baseline SPAB)',
        fontsize=7.8, fontweight='bold', color='#065F46', ha='center', va='center', zorder=4)


# =========================================================================
# PANEL (C): CONV3XC STRUCTURAL RE-PARAMETERIZATION MECHANISM
# Training Phase (High-Capacity Linear Multi-Branch) vs Inference (Single 3x3 Conv)
# =========================================================================
draw_panel(ax, 13.35, 0.25, 6.40, 7.30,
           '(c) Conv3XC: Structural Re-param.', 'Train -> Eval Fused 3x3')

# Training Phase Sub-box
t_box = FancyBboxPatch((13.50, 4.00), 6.10, 3.15, boxstyle="round,pad=0.03,rounding_size=0.10",
                       linewidth=1.4, edgecolor='#6366F1', facecolor='#F5F3FF', zorder=2)
ax.add_patch(t_box)
ax.text(13.65, 6.95, 'TRAINING PHASE (High-Capacity Linear Multi-Branch)', fontsize=8.2, fontweight='bold', color='#4338CA', zorder=3)
ax.text(13.65, 6.75, 'Strictly Linear: No non-linearities between layers allows associative fusion', fontsize=6.8, color='#4F46E5', zorder=3)

# Input X
draw_block(ax, 13.65, 5.15, 0.75, 0.85, 'Input', '$C_{in}$', color=C_TENSOR, border=C_TENSOR_BORDER, title_size=8.2, sub_size=7.2)

# Branch 1: Shortcut (1x1 Conv)
ax.plot([14.40, 14.65, 14.65, 18.35, 18.35], [5.57, 5.57, 6.40, 6.40, 5.80], color='#4F46E5', lw=1.4, linestyle='--', zorder=3)
draw_block(ax, 15.20, 6.15, 2.45, 0.50, r'Shortcut: Conv $1\times 1$ ($C_{in} \to C_{out}$)', '', color='#EEF2FF', border='#4F46E5', title_size=7.4)

# Branch 2: Bottleneck (1x1 -> 3x3 -> 1x1)
draw_conn(ax, 14.40, 5.57, 14.65, 5.57)
draw_block(ax, 14.65, 5.15, 0.98, 0.85, r'Conv $1\times 1$', '$C_{in}\to 2C$', color=C_CONV, border=C_CONV_BORDER, title_size=7.6, sub_size=6.8)

draw_conn(ax, 15.63, 5.57, 15.88, 5.57)
draw_block(ax, 15.88, 5.15, 0.98, 0.85, r'Conv $3\times 3$', '$2C\to 2C$', color=C_CONV, border=C_CONV_BORDER, title_size=7.6, sub_size=6.8)

draw_conn(ax, 16.86, 5.57, 17.11, 5.57)
draw_block(ax, 17.11, 5.15, 0.98, 0.85, r'Conv $1\times 1$', '$2C\to C_{out}$', color=C_CONV, border=C_CONV_BORDER, title_size=7.6, sub_size=6.8)

# Addition (+) of Bottleneck + Shortcut
draw_op(ax, 18.35, 5.57, r'$\oplus$', r=0.18, color='#DBEAFE', border='#2563EB', text_color='#1D4ED8', size=11)
draw_conn(ax, 18.09, 5.57, 18.17, 5.57)
draw_conn(ax, 18.53, 5.57, 18.75, 5.57)
draw_block(ax, 18.75, 5.15, 0.75, 0.85, 'Train\nOutput', '$C_{out}$', color='#EDE9FE', border='#6366F1', title_size=7.8, sub_size=7.0)

# Offline Exact Tensor Fusion Banner
ax.annotate('', xy=(16.55, 3.42), xytext=(16.55, 3.90),
            arrowprops=dict(arrowstyle='->', color='#B91C1C', lw=2.2), zorder=4)

fuse_box = FancyBboxPatch((13.50, 3.45), 6.10, 0.48, boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=1.2, edgecolor='#EF4444', facecolor='#FEE2E2', zorder=4)
ax.add_patch(fuse_box)
ax.text(16.55, 3.78, r'Offline Kernel Fusion: $\mathbf{W}_{eval} = \mathbf{W}_3 * \mathbf{W}_2 * \mathbf{W}_1 + \mathrm{Pad}_{3\times 3}(\mathbf{W}_{sk})$',
        ha='center', va='center', fontsize=7.2, fontweight='bold', color='#B91C1C', zorder=5)
ax.text(16.55, 3.58, r'Offline Bias Fusion: $\mathbf{B}_{eval} = \mathbf{W}_3 * (\mathbf{W}_2 * \mathbf{B}_1 + \mathbf{B}_2) + \mathbf{B}_3 + \mathbf{B}_{sk}$',
        ha='center', va='center', fontsize=7.0, color='#991B1B', zorder=5)

# Inference Phase Sub-box
i_box = FancyBboxPatch((13.50, 0.45), 6.10, 2.90, boxstyle="round,pad=0.03,rounding_size=0.10",
                       linewidth=1.4, edgecolor='#16A34A', facecolor='#F0FDF4', zorder=2)
ax.add_patch(i_box)
ax.text(13.65, 3.12, 'INFERENCE PHASE (Hardware-Optimized Deployment)', fontsize=8.2, fontweight='bold', color='#15803D', zorder=3)
ax.text(13.65, 2.92, 'Zero Multi-Branch Overhead · 100% Cache Locality · Max Edge FPS', fontsize=6.8, color='#16A34A', zorder=3)

# Inference Single Equivalent Conv 3x3
draw_block(ax, 13.65, 1.45, 0.75, 0.85, 'Input', '$C_{in}$', color=C_TENSOR, border=C_TENSOR_BORDER, title_size=8.2, sub_size=7.2)
draw_conn(ax, 14.40, 1.87, 14.85, 1.87)

draw_block(ax, 14.85, 1.25, 3.45, 1.25,
           r'Single Equivalent Standard Conv $3\times 3$',
           r'Kernel: $\mathbf{W}_{eval} \in \mathbb{R}^{C_{out} \times C_{in} \times 3 \times 3}$' + '\n' +
           r'Bias: $\mathbf{B}_{eval} \in \mathbb{R}^{C_{out}}$',
           'Single-layer execution on Mobile/NPU/GPU',
           color='#DCFCE7', border='#16A34A', title_size=8.4, sub_size=7.2, dim_size=6.8)

draw_conn(ax, 18.30, 1.87, 18.75, 1.87)
draw_block(ax, 18.75, 1.45, 0.75, 0.85, 'Output', '$C_{out}$', color='#DCFCE7', border='#16A34A', title_size=8.2, sub_size=7.2)

ax.text(16.55, 0.65, r'Mathematical Identity: $\mathrm{Conv3XC}(X) \equiv \mathrm{Conv2d}(X, \mathbf{W}_{eval}, \mathbf{B}_{eval})$',
        ha='center', va='center', fontsize=7.4, fontweight='bold', color='#14532D', zorder=3)

out_arch = r"C:\Users\anike\.gemini\antigravity\scratch\SPAN\methodology_architecture.png"
plt.savefig(out_arch, bbox_inches='tight', dpi=300)
plt.close()
print(f"Flawless, publication-grade Architecture Diagram saved to {out_arch}")
