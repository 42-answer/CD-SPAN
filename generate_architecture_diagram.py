import os
os.environ['MPLCONFIGDIR'] = r'C:\Users\anike\.gemini\antigravity\scratch\mpl_cache'

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

# Set font family and styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

# Canvas setup - larger canvas with optimal aspect ratio
fig = plt.figure(figsize=(16, 10.2), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 10.2)
ax.axis('off')

# Modern Academic Color Palette
C_BORDER_PANEL = '#CBD5E1' # Slate 300
C_TEXT_MAIN = '#0F172A'    # Slate 900
C_TEXT_MUTED = '#334155'   # Slate 700 (darker for superior readability)

C_CONV = '#EEF2FF'         # Indigo 50
C_CONV_BORDER = '#4F46E5'  # Indigo 600
C_ATTN = '#FFF1F2'         # Rose 50
C_ATTN_BORDER = '#E11D48'  # Rose 600
C_KERNEL = '#FFFBEB'       # Amber 50
C_KERNEL_BORDER = '#D97706'# Amber 600
C_TENSOR = '#F8FAFC'       # Slate 50
C_TENSOR_BORDER = '#475569'# Slate 600
C_OP = '#E0F2FE'           # Sky 100
C_OP_BORDER = '#0284C7'    # Sky 600

# Utility: Draw rounded panel
def draw_panel(ax, x, y, w, h, title, subtitle=''):
    panel = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.18",
                           linewidth=1.6, edgecolor=C_BORDER_PANEL, facecolor='#F8FAFC', zorder=1)
    ax.add_patch(panel)
    # Header bar
    hbar = FancyBboxPatch((x, y + h - 0.55), w, 0.55, boxstyle="round,pad=0.02,rounding_size=0.15",
                          linewidth=0, facecolor='#0F172A', zorder=2)
    ax.add_patch(hbar)
    ax.text(x + 0.3, y + h - 0.28, title, ha='left', va='center',
            fontsize=12, fontweight='bold', color='#FFFFFF', zorder=3)
    if subtitle:
        ax.text(x + w - 0.3, y + h - 0.28, subtitle, ha='right', va='center',
                fontsize=9.2, fontfamily='monospace', color='#94A3B8', zorder=3)

# Utility: Draw functional block with larger typography
def draw_block(ax, x, y, w, h, title, sub='', dim='', color='#FFFFFF', border='#94A3B8', lw=1.5, title_size=10, sub_size=8.2, dim_size=7.6):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                         linewidth=lw, edgecolor=border, facecolor=color, zorder=3)
    ax.add_patch(box)
    cy = y + h/2
    if dim and sub:
        ax.text(x + w/2, cy + 0.25, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=C_TEXT_MAIN, zorder=4)
        ax.text(x + w/2, cy + 0.02, sub, ha='center', va='center', fontsize=sub_size, color=C_TEXT_MUTED, zorder=4)
        ax.text(x + w/2, cy - 0.25, dim, ha='center', va='center', fontsize=dim_size, fontfamily='monospace', fontweight='bold', color='#1D4ED8', zorder=4)
    elif sub:
        ax.text(x + w/2, cy + 0.14, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=C_TEXT_MAIN, zorder=4)
        ax.text(x + w/2, cy - 0.16, sub, ha='center', va='center', fontsize=sub_size, color=C_TEXT_MUTED, zorder=4)
    elif dim:
        ax.text(x + w/2, cy + 0.14, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=C_TEXT_MAIN, zorder=4)
        ax.text(x + w/2, cy - 0.16, dim, ha='center', va='center', fontsize=dim_size, fontfamily='monospace', fontweight='bold', color='#1D4ED8', zorder=4)
    else:
        ax.text(x + w/2, cy, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=C_TEXT_MAIN, zorder=4)

# Utility: Draw mathematical operator node
def draw_op(ax, x, y, symbol, r=0.22, color=C_OP, border=C_OP_BORDER, text_color='#0369A1', size=13):
    circle = Circle((x, y), r, facecolor=color, edgecolor=border, linewidth=1.6, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, symbol, ha='center', va='center', fontsize=size, fontweight='bold', color=text_color, zorder=6)

# Utility: Orthogonal arrow
def draw_conn(ax, x1, y1, x2, y2, color='#334155', lw=1.5, text='', text_pos='top', style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=1.5, shrinkB=1.5), zorder=4)
    if text:
        tx = (x1 + x2)/2
        ty = (y1 + y2)/2 + (0.14 if text_pos=='top' else -0.16)
        ax.text(tx, ty, text, ha='center', va='center', fontsize=7.8, color='#0F172A', fontweight='bold', zorder=5)

# Utility: Draw a 3x3 kernel matrix grid with large numbers
def draw_kernel_grid(ax, x, y, matrix, label, cell_size=0.18, font_size=6.8):
    ax.text(x + 1.5 * cell_size, y + 3 * cell_size + 0.09, label, ha='center', va='bottom',
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
# PANEL (A): MACRO-ARCHITECTURE OF CAD-SPAN
# =========================================================================
draw_panel(ax, 0.2, 6.15, 15.6, 3.85,
           '(a) Macro-Architecture of Proposed CAD-SPAN Framework',
           'Cascade: 6 CAD-SPAB Blocks · Multi-Scale Tap Fusion · PixelShuffle x4')

# 1. Input Tensor
draw_block(ax, 0.45, 7.35, 1.35, 1.45, 'Low-Res Input', r'$I_{LR} \in \mathbb{R}^{3 \times H \times W}$',
           '[B, 3, H, W]', color=C_TENSOR, border=C_TENSOR_BORDER, title_size=10.5, sub_size=8.8, dim_size=8)

# 2. Shallow Feature Extractor
draw_conn(ax, 1.8, 8.07, 2.25, 8.07)
draw_block(ax, 2.25, 7.35, 1.4, 1.45, 'Shallow Conv', 'Conv3XC (3 to 48)\nReparameterizable',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=10, sub_size=8.2, dim_size=8)

# Arrow to Backbone
draw_conn(ax, 3.65, 8.07, 4.05, 8.07, text='$F_0$')

# 3. Cascaded CAD-SPAB Blocks (6 blocks)
block_xs = []
for i in range(1, 7):
    bx = 4.05 + (i - 1) * 1.15
    block_xs.append(bx)
    draw_block(ax, bx, 7.45, 0.96, 1.25, f'CAD-SPAB\n#{i}', f'$F_{{b{i}}}$',
               '[B, 48, H, W]', color=C_ATTN, border=C_ATTN_BORDER, title_size=8.5, sub_size=8, dim_size=7.2)
    if i < 6:
        draw_conn(ax, bx + 0.96, 8.07, bx + 1.15, 8.07)

# 4. Multi-Stage Hierarchical Aggregation (Taps from F0, Fb1, Fb5_2, Fb6)
# F0 Tap
ax.plot([3.85, 3.85, 11.2], [8.07, 9.25, 9.25], color='#4F46E5', lw=1.5, linestyle='--', zorder=3)
ax.plot([11.2, 11.2], [9.25, 8.5], color='#4F46E5', lw=1.5, linestyle='--', zorder=3)
ax.text(5.5, 9.38, r'Shallow Feature Tap $F_0$', fontsize=8.2, color='#4338CA', fontweight='bold')

# Fb1 Tap
ax.plot([block_xs[0] + 0.48, block_xs[0] + 0.48, 11.0], [7.45, 6.6, 6.6], color='#059669', lw=1.4, linestyle=':', zorder=3)
ax.plot([11.0, 11.0], [6.6, 7.55], color='#059669', lw=1.4, linestyle=':', zorder=3)
ax.text(6.2, 6.42, r'Hierarchical Stage Tap $F_{b1}$', fontsize=8.2, color='#047857', fontweight='bold')

# Fb5_2 Tap
ax.plot([block_xs[4] + 0.48, block_xs[4] + 0.48, 11.45], [7.45, 6.95, 6.95], color='#D97706', lw=1.4, linestyle=':', zorder=3)
ax.plot([11.45, 11.45], [6.95, 7.55], color='#D97706', lw=1.4, linestyle=':', zorder=3)
ax.text(9.2, 7.07, r'Intermediate Tap $F_{b5\_2}$', fontsize=8.2, color='#B45309', fontweight='bold')

# Fb6 connection
draw_conn(ax, block_xs[5] + 0.96, 8.07, 11.0, 8.07, text='$F_{b6}$')

# Concat Box
draw_block(ax, 11.0, 7.5, 1.5, 1.15, 'Hierarchical Concat', r'$[F_0, F_{b6}, F_{b1}, F_{b5\_2}]$',
           '[B, 192, H, W]', color='#FEE2E2', border='#EF4444', title_size=8.8, sub_size=7.8, dim_size=7.5)

# 5. Fusion Conv (1x1)
draw_conn(ax, 12.5, 8.07, 12.8, 8.07)
draw_block(ax, 12.8, 7.5, 1.15, 1.15, r'Fusion $1\times 1$', 'Conv + SiLU',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=8.8, sub_size=7.8, dim_size=7.5)

# 6. Global Residual Addition
draw_op(ax, 14.2, 8.07, '+', r=0.20)
draw_conn(ax, 13.95, 8.07, 14.0, 8.07)
# Long skip from F0 to +
ax.plot([3.85, 3.85, 14.2, 14.2], [8.07, 9.6, 9.6, 8.27], color='#4F46E5', lw=1.5, linestyle='-.', zorder=3)
ax.text(8.5, 9.72, r'Global Feature Residual Skip ($+ F_0$)', fontsize=8.5, color='#312E81', fontweight='bold', ha='center')

# 7. PixelShuffle Upsampler
draw_conn(ax, 14.4, 8.07, 14.65, 8.07)
draw_block(ax, 14.65, 7.35, 1.1, 1.45, 'PixelShuffle\nUpsampler', r'Conv3x3 ($48\to 48$)' + '\nShuffle (r=4)',
           '[B, 3, 4H, 4W]', color=C_OP, border=C_OP_BORDER, title_size=8.5, sub_size=7.5, dim_size=7.2)

# Global Bicubic Skip Connection
draw_op(ax, 15.35, 6.6, '+', r=0.18)
ax.plot([1.12, 1.12, 15.35, 15.35], [7.35, 6.3, 6.3, 6.42], color='#0D9488', lw=1.4, linestyle='--', zorder=3)
ax.text(8.0, 6.15, r'Global Bicubic Residual Interpolation Skip: $+ \mathrm{Bicubic}(I_{LR})$', fontsize=8.5, color='#0F766E', fontweight='bold', ha='center')
ax.plot([15.2, 15.35], [7.35, 6.78], color='#334155', lw=1.3, zorder=3)

# Output Tensor
draw_conn(ax, 15.35, 6.6, 15.35, 6.85, style='<-')
ax.plot([15.35, 15.35], [6.6, 6.4], color='#334155', lw=1.4)
ax.text(15.35, 7.05, r'$I_{SR} \in \mathbb{R}^{3 \times 4H \times 4W}$', fontsize=9.2, fontweight='bold', color='#15803D', ha='center')

# =========================================================================
# PANEL (B): MICRO-ARCHITECTURE OF CAD-SPAB
# =========================================================================
draw_panel(ax, 0.2, 0.2, 15.6, 5.75,
           '(b) Micro-Architecture: Curvature-Aware Directional Swift Parameter-free Attention Block (CAD-SPAB)',
           'Dual-Differential Geometric Prior · Temperature Calibration · Exact Origin-Symmetry')

# 1. Block Input
draw_block(ax, 0.45, 2.9, 1.25, 1.25, 'Block Input', r'$X_{in}$',
           '[B, 48, H, W]', color=C_TENSOR, border=C_TENSOR_BORDER, title_size=10, sub_size=9, dim_size=8)

# 2. Three Consecutive Conv3XC Layers with SiLU
draw_conn(ax, 1.7, 3.52, 2.05, 3.52)
draw_block(ax, 2.05, 2.8, 2.15, 1.45, '3x Conv3XC Cascade', 'Conv3XC + SiLU (#1)\nConv3XC + SiLU (#2)\nConv3XC + SiLU (#3)',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=9.5, sub_size=8, dim_size=7.5)

# Output of 3x convs is H
draw_conn(ax, 4.2, 3.52, 4.55, 3.52)
draw_block(ax, 4.55, 3.05, 0.95, 0.95, r'Feature $H$', r'$H \in \mathbb{R}^{C \times H \times W}$',
           '[B, 48, H, W]', color='#FEF3C7', border='#D97706', title_size=9.5, sub_size=8, dim_size=7.5)

# Forking Point from H into 3 Branches
ax.plot([5.5, 5.85], [3.52, 3.52], color='#334155', lw=1.5)
ax.plot([5.85, 5.85], [1.2, 4.75], color='#334155', lw=1.5)

# -------------------------------------------------------------------------
# BRANCH 1: Temperature-Calibrated Origin-Symmetric Gating (Top)
# -------------------------------------------------------------------------
draw_conn(ax, 5.85, 4.75, 6.3, 4.75)
draw_block(ax, 6.3, 4.2, 3.9, 1.1,
           '1. Calibrated Origin-Symmetric Gate',
           r'$A_{sym}(H) = \frac{1}{2}\tanh(\tau \cdot H)$  with  $\tau^* = 0.502$',
           r'Analytically equivalent: $\sigma(x)-0.5 \equiv \frac{1}{2}\tanh(x/2)$',
           color=C_ATTN, border=C_ATTN_BORDER, title_size=9.5, sub_size=8.5, dim_size=8)

# -------------------------------------------------------------------------
# BRANCH 2: First-Order Directional Sobel Gradient Flux (Middle)
# -------------------------------------------------------------------------
draw_conn(ax, 5.85, 2.95, 6.3, 2.95)
draw_block(ax, 6.3, 2.3, 3.9, 1.3,
           '2. Directional Gradient Flux Operator',
           r'$G_x = K_x * H, \quad G_y = K_y * H, \quad \|\nabla H\|_2 = \sqrt{G_x^2 + G_y^2 + \epsilon}$',
           'Captures Horizontal & Vertical Gradient Transitions',
           color=C_KERNEL, border=C_KERNEL_BORDER, title_size=9.5, sub_size=8.2, dim_size=7.8)

# Sobel Kernel Visual Grids (Larger size)
sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
draw_kernel_grid(ax, 8.75, 2.38, sobel_x, r'$K_x$', cell_size=0.17, font_size=6.2)
draw_kernel_grid(ax, 9.55, 2.38, sobel_y, r'$K_y$', cell_size=0.17, font_size=6.2)

# -------------------------------------------------------------------------
# BRANCH 3: Second-Order Isotropic Laplacian Curvature (Bottom)
# -------------------------------------------------------------------------
draw_conn(ax, 5.85, 1.2, 6.3, 1.2)
draw_block(ax, 6.3, 0.65, 3.9, 1.15,
           '3. Isotropic Curvature Operator',
           r'$\Delta H = |\nabla^2 H| = |K_L * H|$   (Inflection & Corner Prior)',
           'Discrete 8-Neighbor Isotropic Stencil',
           color='#F0FDF4', border='#16A34A', title_size=9.5, sub_size=8.2, dim_size=7.8)

# Laplacian Kernel Visual Grid (Larger size)
laplacian_k = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
draw_kernel_grid(ax, 9.15, 0.72, laplacian_k, r'$K_L$', cell_size=0.17, font_size=6.2)

# -------------------------------------------------------------------------
# DUAL-DIFFERENTIAL SPATIAL MODULATOR FUSION
# -------------------------------------------------------------------------
# Merge Branch 2 & Branch 3 into Dual Differential Prior Phi(H)
ax.plot([10.2, 10.55], [2.95, 2.05], color='#334155', lw=1.4)
ax.plot([10.2, 10.55], [1.2, 2.05], color='#334155', lw=1.4)
draw_conn(ax, 10.55, 2.05, 10.85, 2.05)

draw_block(ax, 10.85, 1.35, 2.45, 1.4,
           r'Dual-Differential Prior $\Phi(H)$',
           r'$\Phi(H) = \tanh(\gamma_s \|\nabla H\|_2 + \gamma_l \Delta H)$' + '\n'
           r'$\gamma_s^* = 0.008$ (Sobel Flux Gain)' + '\n'
           r'$\gamma_l^* = 0.001$ (Laplacian Gain)',
           '[B, 48, H, W]', color='#F1F5F9', border='#334155', title_size=9, sub_size=7.6, dim_size=7.2)

# -------------------------------------------------------------------------
# CURVATURE-MODULATED ATTENTION GATE SYNTHESIS
# -------------------------------------------------------------------------
# Shift by 1: (1 + Phi(H))
draw_op(ax, 13.6, 2.05, '+', r=0.20)
draw_conn(ax, 13.3, 2.05, 13.4, 2.05)
ax.text(13.6, 1.6, r'$\mathbf{1}$', fontsize=10, fontweight='bold', color='#1D4ED8', ha='center')
ax.plot([13.6, 13.6], [1.8, 1.85], color='#1D4ED8', lw=1.4)

# Hadamard Modulation with Asym: A_tilde = Asym * (1 + Phi)
draw_op(ax, 13.6, 4.75, r'$\odot$', r=0.24, color='#FCE7F3', border='#EC4899', text_color='#BE185D', size=14)
draw_conn(ax, 10.2, 4.75, 13.36, 4.75, text=r'$A_{sym}$')
draw_conn(ax, 13.6, 2.25, 13.6, 4.51, text=r'$1 + \Phi(H)$', text_pos='right')

# Attention Gate Box
draw_block(ax, 14.1, 4.2, 1.7, 1.1,
           r'Modulated Gate $\widetilde{A}$',
           r'$\widetilde{A} = A_{sym} \odot (\mathbf{1} + \Phi(H))$' + '\n'
           'Zero-parameter Edge Prior',
           '[B, 48, H, W]', color='#FDF2F8', border='#DB2777', title_size=8.5, sub_size=7.5, dim_size=7.2)
draw_conn(ax, 13.84, 4.75, 14.1, 4.75)

# -------------------------------------------------------------------------
# GATED RESIDUAL OUTPUT: O = (H + Xin) * A_tilde
# -------------------------------------------------------------------------
# Local Residual Addition: (H + Xin)
draw_op(ax, 5.0, 0.45, '+', r=0.18)
# Xin skip to local +
ax.plot([1.07, 1.07, 5.0, 5.0], [2.9, 0.45, 0.45, 0.27], color='#2563EB', lw=1.4, linestyle='--', zorder=3)
ax.text(3.0, 0.3, r'Local Input Residual $X_{in}$', fontsize=7.8, color='#1D4ED8', fontweight='bold')
# H to local +
ax.plot([5.02, 5.02, 5.0], [3.05, 1.8, 0.63], color='#D97706', lw=1.3, linestyle=':', zorder=3)

# Final Hadamard Product O = (H + Xin) * A_tilde
draw_op(ax, 14.85, 2.85, r'$\odot$', r=0.26, color='#DCFCE7', border='#16A34A', text_color='#15803D', size=15)
# (H + Xin) path to final Hadamard
ax.plot([5.18, 14.85], [0.45, 0.45], color='#2563EB', lw=1.4, zorder=3)
draw_conn(ax, 14.85, 0.45, 14.85, 2.59, text='$H + X_{in}$', text_pos='right')
# A_tilde to final Hadamard
draw_conn(ax, 14.85, 4.2, 14.85, 3.11, text=r'$\widetilde{A}$', text_pos='right')

# Block Output Box
draw_block(ax, 14.05, 1.45, 1.75, 0.95,
           r'Block Output $O$',
           r'$O = (H + X_{in}) \odot \widetilde{A}$',
           '[B, 48, H, W]', color='#F0FDF4', border='#15803D', title_size=9, sub_size=7.8, dim_size=7.5)
draw_conn(ax, 14.85, 2.59, 14.85, 2.4)

# Identity Fallback Callout Banner
fallback_box = FancyBboxPatch((6.3, 0.22), 9.4, 0.35, boxstyle="round,pad=0.02,rounding_size=0.08",
                              linewidth=1.2, edgecolor='#10B981', facecolor='#ECFDF5', zorder=3)
ax.add_patch(fallback_box)
ax.text(11.0, 0.39, r'Mathematical Identity Fallback Guarantee: When $\gamma_s = \gamma_l = 0$ and $\tau = 0.50$, $\Phi(H) \equiv 0$ and $\widetilde{A}(H) \equiv \sigma(H) - 0.5$ (Exact Baseline SPAB)',
        fontsize=7.8, fontweight='bold', color='#065F46', ha='center', va='center', zorder=4)

out_arch = r"C:\Users\anike\.gemini\antigravity\scratch\SPAN\methodology_architecture.png"
plt.savefig(out_arch, bbox_inches='tight', dpi=300)
print(f"Bigger, publication-grade Architecture Diagram saved to {out_arch}")
