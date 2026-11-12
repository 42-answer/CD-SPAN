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

# Color Palette inspired by original SPAN paper and modern publication standards
C_PANEL_BG = '#F8FAFC'
C_PANEL_BORDER = '#94A3B8'
C_HEADER_BG = '#0F172A'

C_CONV = '#ECFDF5'          # Light emerald/green (like original SPAN Conv3)
C_CONV_BORDER = '#059669'   # Emerald 600
C_SPAB = '#EFF6FF'          # Light blue/indigo (like original SPAB cubes)
C_SPAB_BORDER = '#2563EB'   # Blue 600
C_SIGMA_A = '#FFF1F2'       # Rose/pink (like original sigma_a block)
C_SIGMA_A_BORDER = '#E11D48'# Rose 600
C_CONCAT = '#FEF3C7'        # Warm amber (like original concat block)
C_CONCAT_BORDER = '#D97706' # Amber 600
C_PIXELSHUFFLE = '#FEF9C3'  # Yellow (like original PixelShuffle)
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
    ax.text(x + 0.35, y + h - 0.29, title, ha='left', va='center',
            fontsize=11.5, fontweight='bold', color='#FFFFFF', zorder=3)
    if subtitle:
        ax.text(x + w - 0.35, y + h - 0.29, subtitle, ha='right', va='center',
                fontsize=8.8, fontfamily='monospace', color='#94A3B8', zorder=3)

def draw_block(ax, x, y, w, h, title, sub='', dim='', color='#FFFFFF', border='#64748B', lw=1.5,
               title_size=9.2, sub_size=7.8, dim_size=7.2, title_color='#0F172A'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.10",
                         linewidth=lw, edgecolor=border, facecolor=color, zorder=3)
    ax.add_patch(box)
    cy = y + h/2
    if dim and sub:
        ax.text(x + w/2, cy + 0.24, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=title_color, zorder=4)
        ax.text(x + w/2, cy + 0.01, sub, ha='center', va='center', fontsize=sub_size, color='#334155', zorder=4)
        ax.text(x + w/2, cy - 0.24, dim, ha='center', va='center', fontsize=dim_size, fontfamily='monospace', fontweight='bold', color='#1D4ED8', zorder=4)
    elif sub:
        ax.text(x + w/2, cy + 0.14, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=title_color, zorder=4)
        ax.text(x + w/2, cy - 0.16, sub, ha='center', va='center', fontsize=sub_size, color='#334155', zorder=4)
    elif dim:
        ax.text(x + w/2, cy + 0.14, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=title_color, zorder=4)
        ax.text(x + w/2, cy - 0.16, dim, ha='center', va='center', fontsize=dim_size, fontfamily='monospace', fontweight='bold', color='#1D4ED8', zorder=4)
    else:
        ax.text(x + w/2, cy, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=title_color, zorder=4)

def draw_op(ax, x, y, symbol, r=0.20, color='#E0F2FE', border='#0284C7', text_color='#0369A1', size=13):
    circle = Circle((x, y), r, facecolor=color, edgecolor=border, linewidth=1.6, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, symbol, ha='center', va='center', fontsize=size, fontweight='bold', color=text_color, zorder=6)

def draw_conn(ax, x1, y1, x2, y2, color='#334155', lw=1.5, text='', text_pos='top', style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=1.5, shrinkB=1.5), zorder=4)
    if text:
        tx = (x1 + x2)/2
        ty = (y1 + y2)/2 + (0.14 if text_pos=='top' else -0.16)
        ax.text(tx, ty, text, ha='center', va='center', fontsize=8.0, color='#0F172A', fontweight='bold', zorder=5)

def draw_kernel_grid(ax, x, y, matrix, label, cell_size=0.17, font_size=6.2):
    ax.text(x + 1.5 * cell_size, y + 3 * cell_size + 0.08, label, ha='center', va='bottom',
            fontsize=7.8, fontweight='bold', color='#92400E')
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
# Strictly faithful to Figure 2 of original SPAN paper + PyTorch code
# =========================================================================
draw_panel(ax, 0.25, 8.2, 19.5, 5.05,
           '(a) Global Macro-Architecture of Proposed CAD-SPAN Framework',
           'Input LR -> Shallow Conv3XC -> 6 Cascaded CAD-SPABs -> Conv3XC Refine -> Concat -> PixelShuffle -> HR Output')

# 1. LR Input
draw_block(ax, 0.5, 9.85, 1.35, 1.45, 'LR Input', r'$I_{LR} \in \mathbb{R}^{3 \times H \times W}$',
           '[B, 3, H, W]', color=C_TENSOR, border=C_TENSOR_BORDER, title_size=10, sub_size=8.5, dim_size=8.0)

# Mean shift note
ax.text(1.17, 9.55, 'Mean Shift:\n$(I_{LR} - \mu) \\times 255$', ha='center', va='top', fontsize=7.2, color='#475569')

# 2. Shallow Conv3XC (conv_1 in code: 3 -> 48)
draw_conn(ax, 1.85, 10.57, 2.25, 10.57)
draw_block(ax, 2.25, 9.85, 1.45, 1.45, 'Shallow Conv', 'Conv3XC (3 to 48)\nReparameterizable',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=9.5, sub_size=8.0, dim_size=7.8)

# Output of conv_1 is Feat.Map 1 (F0)
draw_conn(ax, 3.7, 10.57, 4.15, 10.57, text=r'$\mathrm{Feat.Map}_1$ ($F_0$)')

# 3. Cascaded Blocks: CAD-SPAB 1 to 6
draw_block(ax, 4.15, 9.9, 1.25, 1.35, 'CAD-SPAB #1', r'$\mathrm{Feat.Map}_2$ ($F_{b1}$)', '[B, 48, H, W]',
           color=C_SPAB, border=C_SPAB_BORDER, title_size=9.0, sub_size=7.8, dim_size=7.4)

draw_conn(ax, 5.4, 10.57, 5.8, 10.57)
draw_block(ax, 5.8, 9.9, 1.25, 1.35, 'CAD-SPAB #2', r'$\mathrm{Feat.Map}_3$ ($F_{b2}$)', '[B, 48, H, W]',
           color=C_SPAB, border=C_SPAB_BORDER, title_size=9.0, sub_size=7.8, dim_size=7.4)

# Dashed connection for intermediate blocks (SPAB 3, 4)
ax.plot([7.05, 7.65], [10.57, 10.57], color='#2563EB', lw=2.0, linestyle='--', zorder=4)
ax.text(7.35, 10.75, r'$\cdots$', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#1D4ED8')

draw_conn(ax, 7.65, 10.57, 8.0, 10.57)
draw_block(ax, 8.0, 9.9, 1.25, 1.35, 'CAD-SPAB #5', r'$\mathrm{Feat.Map}_5$ ($F_{b5}$)', '[B, 48, H, W]',
           color=C_SPAB, border=C_SPAB_BORDER, title_size=9.0, sub_size=7.8, dim_size=7.4)

draw_conn(ax, 9.25, 10.57, 9.65, 10.57)
draw_block(ax, 9.65, 9.9, 1.35, 1.35, 'CAD-SPAB #6', r'Outputs: $F_{b6}$ and' + '\n' + r'$F_{b5\_2}$ ($\mathrm{Feat.Map}_6$)',
           '[B, 48, H, W]', color=C_SPAB, border=C_SPAB_BORDER, title_size=9.0, sub_size=7.5, dim_size=7.2)

# 4. Refinement Conv3XC (conv_2 in code: 48 -> 48) - EXACTLY AS IN SPAN!
draw_conn(ax, 11.0, 10.57, 11.45, 10.57, text=r'$F_{b6}$')
draw_block(ax, 11.45, 9.85, 1.45, 1.45, 'Refine Conv', 'Conv3XC (48 to 48)\n(conv_2 in SPAN)',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=9.5, sub_size=8.0, dim_size=7.8)

# Output of conv_2 is Feat.Map 7
draw_conn(ax, 12.9, 10.57, 13.4, 10.57, text=r'$\mathrm{Feat.Map}_7$')

# 5. The Four Multi-Scale Hierarchical Tap Connections into Concat Layer (Mirroring SPAN Figure 2)
# Tap 1: Feat.Map 1 (F0) from after conv_1
ax.plot([3.92, 3.92, 13.55], [10.57, 12.2, 12.2], color='#059669', lw=1.6, linestyle='-', zorder=3)
ax.plot([13.55, 13.55], [12.2, 11.25], color='#059669', lw=1.6, linestyle='-', zorder=3)
ax.annotate('', xy=(13.55, 11.1), xytext=(13.55, 11.25), arrowprops=dict(arrowstyle='->', color='#059669', lw=1.6), zorder=4)
ax.text(6.8, 12.35, r'Tap 1: $\mathrm{Feat.Map}_1$ ($F_0$, Shallow Features)', fontsize=8.2, color='#047857', fontweight='bold')

# Tap 2: Feat.Map 2 (Fb1) from after CAD-SPAB #1
ax.plot([4.77, 4.77, 13.75], [9.9, 8.8, 8.8], color='#2563EB', lw=1.5, linestyle='-', zorder=3)
ax.plot([13.75, 13.75], [8.8, 9.85], color='#2563EB', lw=1.5, linestyle='-', zorder=3)
ax.annotate('', xy=(13.75, 10.0), xytext=(13.75, 9.85), arrowprops=dict(arrowstyle='->', color='#2563EB', lw=1.5), zorder=4)
ax.text(7.2, 8.62, r'Tap 2: $\mathrm{Feat.Map}_2$ ($F_{b1}$, Early Block Details)', fontsize=8.2, color='#1D4ED8', fontweight='bold')

# Tap 3: Feat.Map 6 (Fb5_2) intermediate tap from CAD-SPAB #6
ax.plot([10.32, 10.32, 13.95], [9.9, 9.2, 9.2], color='#B45309', lw=1.5, linestyle='-', zorder=3)
ax.plot([13.95, 13.95], [9.2, 9.85], color='#B45309', lw=1.5, linestyle='-', zorder=3)
ax.annotate('', xy=(13.95, 10.0), xytext=(13.95, 9.85), arrowprops=dict(arrowstyle='->', color='#B45309', lw=1.5), zorder=4)
ax.text(11.2, 9.32, r'Tap 3: $\mathrm{Feat.Map}_6$ ($F_{b5\_2}$, Mid-Stage Tap)', fontsize=8.0, color='#92400E', fontweight='bold')

# Tap 4: Feat.Map 7 from refine conv_2 goes straight in (x=12.9 to 13.4)

# 6. Concatenation Layer (torch.cat([F0, Fb6_conv, Fb1, Fb5_2], dim=1))
draw_block(ax, 13.4, 10.0, 1.85, 1.15, 'Concat. Layer', r'$[F_0, F_{b6\_conv}, F_{b1}, F_{b5\_2}]$',
           '[B, 192, H, W]', color=C_CONCAT, border=C_CONCAT_BORDER, title_size=9.2, sub_size=8.0, dim_size=7.6)

# 7. 1x1 Conv Compression (conv_cat in code: 192 -> 48)
draw_conn(ax, 15.25, 10.57, 15.65, 10.57)
draw_block(ax, 15.65, 10.0, 1.35, 1.15, r'Conv $1\times 1$', 'conv_cat (192 to 48)\nChannel Fusion',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=9.0, sub_size=7.8, dim_size=7.6)

# 8. PixelShuffle Block (upsampler in code: 48 -> 48 -> PixelShuffle x4 -> 3 channels)
# NOTE: Directly from SPAN code: NO BICUBIC SKIP! NO EXTRA RESIDUAL ADD!
draw_conn(ax, 17.0, 10.57, 17.4, 10.57)
draw_block(ax, 17.4, 9.85, 1.45, 1.45, 'PixelShuffle\nUpsampler',
           r'Conv $3\times 3$ ($48\to 48$)' + '\n' + r'PixelShuffle ($\times 4$)',
           r'$48 \to 3 \times 4^2 \to 3$', color=C_PIXELSHUFFLE, border=C_PIXELSHUFFLE_BORDER,
           title_size=9.2, sub_size=7.8, dim_size=7.5)

# 9. High-Resolution Output
draw_conn(ax, 18.85, 10.57, 19.15, 10.57)
draw_block(ax, 19.15, 9.85, 0.55, 1.45, '', '', '', color='#DCFCE7', border='#16A34A', lw=2.0)
ax.text(19.42, 10.57, 'HR\nOutput\n$I_{SR}$', ha='center', va='center', fontsize=9.2, fontweight='bold', color='#14532D')
ax.text(19.42, 9.55, r'$[B, 3, 4H, 4W]$', ha='center', va='top', fontsize=7.2, fontfamily='monospace', fontweight='bold', color='#16A34A')


# =========================================================================
# PANEL (B): MICRO-ARCHITECTURE OF CAD-SPAB BLOCK
# Correct signal flow: (H + Xin) @ Addition -> Final Output @ Multiplication with A_mod!
# =========================================================================
draw_panel(ax, 0.25, 0.25, 12.8, 7.75,
           '(b) CAD-SPAB Micro-Architecture: Curvature-Aware Directional Attention Block',
           'Residual Content (H + Xin) modulated by Curvature Gate A_mod')

# Block Input Xin
draw_block(ax, 0.5, 4.2, 1.25, 1.35, 'Block Input', r'$X_{in}$',
           '[B, 48, H, W]', color=C_TENSOR, border=C_TENSOR_BORDER, title_size=10, sub_size=9.0, dim_size=8.0)

# 1. Main Convolutional Trunk: 3x Conv3XC Cascade
draw_conn(ax, 1.75, 4.87, 2.05, 4.87)
draw_block(ax, 2.05, 4.25, 1.25, 1.25, r'$\mathrm{Conv3XC}_1$', r'SiLU Act ($\mathrm{act}_1$)' + '\n' + r'Yields $\mathrm{out}_1$',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=9.0, sub_size=7.5, dim_size=7.2)

draw_conn(ax, 3.3, 4.87, 3.55, 4.87)
draw_block(ax, 3.55, 4.25, 1.25, 1.25, r'$\mathrm{Conv3XC}_2$', r'SiLU Act ($\mathrm{act}_1$)' + '\n' + r'Yields $\mathrm{out}_2$',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=9.0, sub_size=7.5, dim_size=7.2)

draw_conn(ax, 4.8, 4.87, 5.05, 4.87)
draw_block(ax, 5.05, 4.25, 1.25, 1.25, r'$\mathrm{Conv3XC}_3$', 'No activation\nYields Feature $H$',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=9.0, sub_size=7.5, dim_size=7.2)

# Output of Conv3 is Feature H
draw_conn(ax, 6.3, 4.87, 6.65, 4.87)
draw_block(ax, 6.65, 4.42, 1.05, 0.9, r'Feature $H$', r'$H \equiv \mathrm{out}_3$',
           '[B, 48, H, W]', color='#FEF3C7', border='#D97706', title_size=9.2, sub_size=7.8, dim_size=7.2)

# 2. Local Input Residual Addition (H + Xin) - EXACTLY AS IN SPAN FIGURE 2
# Xin travels over the top to the Addition node (+)
draw_op(ax, 8.4, 6.6, '+', r=0.22, color='#DBEAFE', border='#2563EB', text_color='#1D4ED8')
# Xin skip line
ax.plot([1.12, 1.12, 8.4], [5.55, 7.15, 7.15], color='#2563EB', lw=1.6, linestyle='-', zorder=3)
draw_conn(ax, 8.4, 7.15, 8.4, 6.82)
ax.text(4.0, 7.28, r'Local Input Identity Shortcut: $X_{in}$', fontsize=8.2, color='#1D4ED8', fontweight='bold')

# H branch to Addition node (+)
ax.plot([7.17, 7.17, 8.18], [5.32, 6.6, 6.6], color='#D97706', lw=1.5, linestyle='-', zorder=3)
draw_conn(ax, 8.18, 6.6, 8.18, 6.6)
ax.text(7.45, 6.75, r'$H$', fontsize=8.5, color='#B45309', fontweight='bold')

# Residual Sum Result: (H + Xin)
draw_block(ax, 8.85, 6.15, 1.6, 0.9, r'Residual Sum', r'$H + X_{in}$',
           '[B, 48, H, W]', color='#EFF6FF', border='#3B82F6', title_size=9.0, sub_size=8.0, dim_size=7.2)
draw_conn(ax, 8.62, 6.6, 8.85, 6.6)

# 3. Novel Differential Attention Generation: Feature H splits into 3 streams!
ax.plot([7.17, 7.17], [4.42, 1.35], color='#B45309', lw=1.6, zorder=3)

# Stream 1: Origin-Symmetric Gate (Top of attention area)
draw_conn(ax, 7.17, 4.3, 7.55, 4.3)
draw_block(ax, 7.55, 3.8, 3.25, 1.05,
           '1. Calibrated Origin-Symmetric Gate',
           r'$A_{sym}(H) = \frac{1}{2}\tanh(\tau \cdot H)$  ($\tau^* = 0.502$)' + '\n' +
           r'Identity: $\sigma(x)-0.5 \equiv \frac{1}{2}\tanh(x/2)$',
           'Exact odd symmetry: $A(-x) = -A(x)$',
           color=C_SIGMA_A, border=C_SIGMA_A_BORDER, title_size=8.6, sub_size=7.4, dim_size=6.8)

# Stream 2: First-Order Directional Sobel Gradient Flux (Middle)
draw_conn(ax, 7.17, 2.7, 7.55, 2.7)
box2 = FancyBboxPatch((7.55, 1.95), 3.35, 1.48, boxstyle="round,pad=0.03,rounding_size=0.10",
                      linewidth=1.5, edgecolor='#D97706', facecolor='#FFFBEB', zorder=3)
ax.add_patch(box2)
ax.text(7.72, 3.24, '2. Directional Gradient Flux Operator', ha='left', va='center', fontsize=8.6, fontweight='bold', color='#0F172A', zorder=4)
ax.text(7.72, 2.96, r'$G_x = K_x * H, \quad G_y = K_y * H$', ha='left', va='center', fontsize=7.8, color='#334155', zorder=4)
ax.text(7.72, 2.68, r'$\|\nabla H\|_2 = \sqrt{G_x^2 + G_y^2 + \epsilon}$', ha='left', va='center', fontsize=7.8, color='#334155', zorder=4)
ax.text(7.72, 2.40, 'Step Boundary Flux Detector', ha='left', va='center', fontsize=7.2, color='#475569', zorder=4)
ax.text(7.72, 2.14, '[B, 48, H, W]', ha='left', va='center', fontsize=6.8, fontfamily='monospace', fontweight='bold', color='#1D4ED8', zorder=4)

# Sobel Grids cleanly positioned inside block 2 on right
sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
draw_kernel_grid(ax, 9.60, 2.05, sobel_x, r'$K_x$', cell_size=0.16, font_size=5.8)
draw_kernel_grid(ax, 10.25, 2.05, sobel_y, r'$K_y$', cell_size=0.16, font_size=5.8)

# Stream 3: Second-Order Isotropic Laplacian Curvature (Bottom)
draw_conn(ax, 7.17, 1.1, 7.55, 1.1)
box3 = FancyBboxPatch((7.55, 0.45), 3.35, 1.35, boxstyle="round,pad=0.03,rounding_size=0.10",
                      linewidth=1.5, edgecolor='#16A34A', facecolor='#F0FDF4', zorder=3)
ax.add_patch(box3)
ax.text(7.72, 1.58, '3. Isotropic Curvature Operator', ha='left', va='center', fontsize=8.6, fontweight='bold', color='#0F172A', zorder=4)
ax.text(7.72, 1.32, r'$\Delta H = |\nabla^2 H| = |K_L * H|$', ha='left', va='center', fontsize=7.8, color='#334155', zorder=4)
ax.text(7.72, 1.06, 'Corner & Ridge Extrema Prior', ha='left', va='center', fontsize=7.2, color='#475569', zorder=4)
ax.text(7.72, 0.82, '8-Neighbor Isotropic Stencil', ha='left', va='center', fontsize=7.0, color='#475569', zorder=4)
ax.text(7.72, 0.60, '[B, 48, H, W]', ha='left', va='center', fontsize=6.8, fontfamily='monospace', fontweight='bold', color='#1D4ED8', zorder=4)

laplacian_k = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
draw_kernel_grid(ax, 9.95, 0.56, laplacian_k, r'$K_L$', cell_size=0.16, font_size=5.8)

# Fusion of Streams 2 & 3 into Dual-Differential Prior Phi(H)
ax.plot([10.9, 11.05], [2.7, 1.9], color='#334155', lw=1.5)
ax.plot([10.9, 11.05], [1.1, 1.9], color='#334155', lw=1.5)
draw_conn(ax, 11.05, 1.9, 11.25, 1.9)
draw_block(ax, 11.25, 1.25, 1.45, 1.3,
           r'Differential $\Phi(H)$',
           r'$\Phi = \tanh(\gamma_s \|\nabla H\| + \gamma_l \Delta H)$' + '\n' +
           r'$\gamma_s^*=0.008, \gamma_l^*=0.001$',
           '[B, 48, H, W]', color=C_DIFF, border=C_DIFF_BORDER, title_size=8.2, sub_size=7.0, dim_size=6.6)

# Hadamard Gate Modulation: A_mod = Asym * (1 + Phi)
draw_op(ax, 11.5, 4.3, r'$\odot$', r=0.22, color='#FCE7F3', border='#EC4899', text_color='#BE185D', size=13)
draw_conn(ax, 10.8, 4.3, 11.28, 4.3, text=r'$A_{sym}$')

# Phi(H) travels up through + 1 node to modulation node
draw_op(ax, 12.25, 3.1, '+', r=0.18)
ax.text(12.58, 3.1, r'$\mathbf{1}$', fontsize=9.2, fontweight='bold', color='#1D4ED8', va='center')
ax.plot([12.0, 12.25], [2.55, 2.92], color='#334155', lw=1.4)
draw_conn(ax, 12.25, 3.28, 11.7, 4.15, text=r'$1 + \Phi(H)$', text_pos='right')

# Output of Attention Gate Modulation: A_mod
draw_block(ax, 10.85, 4.95, 1.7, 0.85,
           r'Modulated Gate $\widetilde{A}$',
           r'$\widetilde{A} = A_{sym} \odot (\mathbf{1} + \Phi(H))$',
           '[B, 48, H, W]', color='#FDF2F8', border='#DB2777', title_size=8.5, sub_size=7.4, dim_size=6.8)
draw_conn(ax, 11.5, 4.52, 11.5, 4.95)

# 4. FINAL ELEMENT-WISE MULTIPLICATION (o) NODE:
# Residual Sum (H + Xin) meets Attention Gate A_mod!
draw_op(ax, 11.5, 6.6, r'$\otimes$', r=0.25, color='#DCFCE7', border='#16A34A', text_color='#15803D', size=15)
# Connection from (H + Xin) to multiplication node
draw_conn(ax, 10.45, 6.6, 11.25, 6.6, text=r'$H + X_{in}$')
# Connection from A_mod to multiplication node
draw_conn(ax, 11.5, 5.8, 11.5, 6.35, text=r'$\widetilde{A}$')

# Final Block Output O
draw_conn(ax, 11.75, 6.6, 12.05, 6.6)
draw_block(ax, 12.05, 6.02, 1.15, 1.16,
           'Block Output', r'$O = (H + X_{in}) \otimes \widetilde{A}$',
           '[B, 48, H, W]', color='#F0FDF4', border='#15803D', title_size=8.8, sub_size=7.4, dim_size=6.6)

# Identity Fallback Callout Box
fb_patch = FancyBboxPatch((0.5, 0.32), 12.2, 0.35, boxstyle="round,pad=0.02,rounding_size=0.08",
                          linewidth=1.2, edgecolor='#10B981', facecolor='#ECFDF5', zorder=3)
ax.add_patch(fb_patch)
ax.text(6.6, 0.49, r'Exact Mathematical Identity Fallback: When $\gamma_s = \gamma_l = 0$ and $\tau = 0.50$, $\Phi(H) \equiv 0$ and $\widetilde{A}(H) \equiv \sigma(H) - 0.5$ (Exact Baseline SPAB)',
        fontsize=7.8, fontweight='bold', color='#065F46', ha='center', va='center', zorder=4)


# =========================================================================
# PANEL (C): CONV3XC STRUCTURAL RE-PARAMETERIZATION MECHANISM
# Training Multi-Branch vs Inference Single 3x3 Conv
# =========================================================================
draw_panel(ax, 13.3, 0.25, 6.45, 7.75,
           '(c) Conv3XC: Structural Re-param.', 'Multi-Branch -> Single 3x3')

# Training Phase Sub-box
t_box = FancyBboxPatch((13.5, 4.15), 6.05, 3.4, boxstyle="round,pad=0.03,rounding_size=0.10",
                       linewidth=1.4, edgecolor='#6366F1', facecolor='#F5F3FF', zorder=2)
ax.add_patch(t_box)
ax.text(13.7, 7.28, 'TRAINING PHASE (High-Capacity Multi-Branch)', fontsize=8.8, fontweight='bold', color='#4338CA', zorder=3)

# Input X
draw_block(ax, 13.65, 5.35, 0.85, 0.9, 'Input', '$C_{in}$', color=C_TENSOR, border=C_TENSOR_BORDER, title_size=8.5, sub_size=7.5)

# Branch 1: Shortcut (1x1 Conv)
ax.plot([14.5, 14.75, 14.75, 18.25, 18.25], [5.8, 5.8, 6.75, 6.75, 6.02], color='#4F46E5', lw=1.4, linestyle='--', zorder=3)
draw_block(ax, 15.35, 6.45, 2.3, 0.6, r'Shortcut: Conv $1\times 1$ ($C_{in} \to C_{out}$)', '', color='#EEF2FF', border='#4F46E5', title_size=7.6)

# Branch 2: Bottleneck (1x1 -> 3x3 -> 1x1)
draw_conn(ax, 14.5, 5.8, 14.75, 5.8)
draw_block(ax, 14.75, 5.4, 0.98, 0.8, r'Conv $1\times 1$', '$C_{in}\to 2C$', color=C_CONV, border=C_CONV_BORDER, title_size=7.8, sub_size=6.8)

draw_conn(ax, 15.73, 5.8, 15.98, 5.8)
draw_block(ax, 15.98, 5.4, 0.98, 0.8, r'Conv $3\times 3$', '$2C\to 2C$', color=C_CONV, border=C_CONV_BORDER, title_size=7.8, sub_size=6.8)

draw_conn(ax, 16.96, 5.8, 17.21, 5.8)
draw_block(ax, 17.21, 5.4, 0.98, 0.8, r'Conv $1\times 1$', '$2C\to C_{out}$', color=C_CONV, border=C_CONV_BORDER, title_size=7.8, sub_size=6.8)

# Addition (+) of Bottleneck + Shortcut
draw_op(ax, 18.25, 5.8, '+', r=0.18)
draw_conn(ax, 18.19, 5.8, 18.07, 5.8, style='<-')
draw_conn(ax, 18.43, 5.8, 18.7, 5.8)
draw_block(ax, 18.7, 5.35, 0.75, 0.9, 'Train\nOutput', '$C_{out}$', color='#EDE9FE', border='#6366F1', title_size=8.0, sub_size=7.2)

# Offline Kernel Fusion Banner
ax.annotate('', xy=(16.5, 3.65), xytext=(16.5, 4.05),
            arrowprops=dict(arrowstyle='->', color='#B91C1C', lw=2.2), zorder=4)
ax.text(16.5, 3.85, r'Offline Kernel Fusion: $\mathbf{W}_{eval} = \mathbf{W}_3 * \mathbf{W}_2 * \mathbf{W}_1 + \mathrm{Pad}_{3\times 3}(\mathbf{W}_{sk})$',
        ha='center', va='center', fontsize=7.6, fontweight='bold', color='#B91C1C',
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEE2E2", edgecolor="#EF4444", lw=1.0), zorder=5)

# Inference Phase Sub-box
i_box = FancyBboxPatch((13.5, 0.45), 6.05, 3.05, boxstyle="round,pad=0.03,rounding_size=0.10",
                       linewidth=1.4, edgecolor='#16A34A', facecolor='#F0FDF4', zorder=2)
ax.add_patch(i_box)
ax.text(13.7, 3.2, 'INFERENCE PHASE (Hardware-Optimized Topology)', fontsize=8.8, fontweight='bold', color='#15803D', zorder=3)

# Inference Single Conv3x3
draw_block(ax, 13.65, 1.55, 0.85, 0.9, 'Input', '$C_{in}$', color=C_TENSOR, border=C_TENSOR_BORDER, title_size=8.5, sub_size=7.5)
draw_conn(ax, 14.5, 2.0, 14.95, 2.0)

draw_block(ax, 14.95, 1.35, 3.15, 1.3,
           r'Single Equivalent Standard Conv $3\times 3$',
           r'Weight: $\mathbf{W}_{eval} \in \mathbb{R}^{C_{out} \times C_{in} \times 3 \times 3}$' + '\n' +
           r'Bias: $\mathbf{B}_{eval} \in \mathbb{R}^{C_{out}}$',
           color='#DCFCE7', border='#16A34A', title_size=8.5, sub_size=7.4)

draw_conn(ax, 18.1, 2.0, 18.55, 2.0)
draw_block(ax, 18.55, 1.55, 0.85, 0.9, 'Output', '$C_{out}$', color='#DCFCE7', border='#16A34A', title_size=8.5, sub_size=7.5)

ax.text(16.5, 0.72, 'Zero Multi-Branch Overhead · 100% Cache Locality · Max Edge FPS',
        ha='center', va='center', fontsize=7.8, fontweight='bold', color='#14532D', zorder=3)

out_arch = r"C:\Users\anike\.gemini\antigravity\scratch\SPAN\methodology_architecture.png"
plt.savefig(out_arch, bbox_inches='tight', dpi=300)
plt.close()
print(f"Flawless, publication-grade Architecture Diagram saved to {out_arch}")
