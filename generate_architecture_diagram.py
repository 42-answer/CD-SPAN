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

# Canvas setup - Large, publication-grade high-resolution canvas
fig = plt.figure(figsize=(20, 13.5), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 20)
ax.set_ylim(0, 13.5)
ax.axis('off')

# Modern Academic Color Palette
C_BORDER_PANEL = '#94A3B8' # Slate 400
C_HEADER_BG = '#0F172A'    # Slate 900
C_TEXT_MAIN = '#0F172A'    # Slate 900
C_TEXT_MUTED = '#334155'   # Slate 700

C_CONV = '#EEF2FF'         # Indigo 50
C_CONV_BORDER = '#4338CA'  # Indigo 700
C_ATTN = '#FFF1F2'         # Rose 50
C_ATTN_BORDER = '#BE123C'  # Rose 700
C_KERNEL = '#FFFBEB'       # Amber 50
C_KERNEL_BORDER = '#B45309'# Amber 700
C_TENSOR = '#F8FAFC'       # Slate 50
C_TENSOR_BORDER = '#334155'# Slate 700
C_OP = '#E0F2FE'           # Sky 100
C_OP_BORDER = '#0369A1'    # Sky 700
C_GREEN = '#F0FDF4'        # Green 50
C_GREEN_BORDER = '#15803D' # Green 700

# Utility: Draw rounded panel
def draw_panel(ax, x, y, w, h, title, subtitle=''):
    panel = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.18",
                           linewidth=1.8, edgecolor=C_BORDER_PANEL, facecolor='#F8FAFC', zorder=1)
    ax.add_patch(panel)
    # Header bar
    hbar = FancyBboxPatch((x, y + h - 0.58), w, 0.58, boxstyle="round,pad=0.02,rounding_size=0.15",
                          linewidth=0, facecolor=C_HEADER_BG, zorder=2)
    ax.add_patch(hbar)
    ax.text(x + 0.35, y + h - 0.29, title, ha='left', va='center',
            fontsize=12.5, fontweight='bold', color='#FFFFFF', zorder=3)
    if subtitle:
        ax.text(x + w - 0.35, y + h - 0.29, subtitle, ha='right', va='center',
                fontsize=9.6, fontfamily='monospace', color='#94A3B8', zorder=3)

# Utility: Draw functional block with high typographic contrast
def draw_block(ax, x, y, w, h, title, sub='', dim='', color='#FFFFFF', border='#64748B', lw=1.6, title_size=10, sub_size=8.2, dim_size=7.6):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                         linewidth=lw, edgecolor=border, facecolor=color, zorder=3)
    ax.add_patch(box)
    cy = y + h/2
    if dim and sub:
        ax.text(x + w/2, cy + 0.26, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=C_TEXT_MAIN, zorder=4)
        ax.text(x + w/2, cy + 0.02, sub, ha='center', va='center', fontsize=sub_size, color=C_TEXT_MUTED, zorder=4)
        ax.text(x + w/2, cy - 0.26, dim, ha='center', va='center', fontsize=dim_size, fontfamily='monospace', fontweight='bold', color='#1D4ED8', zorder=4)
    elif sub:
        ax.text(x + w/2, cy + 0.15, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=C_TEXT_MAIN, zorder=4)
        ax.text(x + w/2, cy - 0.16, sub, ha='center', va='center', fontsize=sub_size, color=C_TEXT_MUTED, zorder=4)
    elif dim:
        ax.text(x + w/2, cy + 0.15, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=C_TEXT_MAIN, zorder=4)
        ax.text(x + w/2, cy - 0.16, dim, ha='center', va='center', fontsize=dim_size, fontfamily='monospace', fontweight='bold', color='#1D4ED8', zorder=4)
    else:
        ax.text(x + w/2, cy, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=C_TEXT_MAIN, zorder=4)

# Utility: Draw mathematical operator node
def draw_op(ax, x, y, symbol, r=0.22, color=C_OP, border=C_OP_BORDER, text_color='#0369A1', size=13):
    circle = Circle((x, y), r, facecolor=color, edgecolor=border, linewidth=1.6, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, symbol, ha='center', va='center', fontsize=size, fontweight='bold', color=text_color, zorder=6)

# Utility: Orthogonal arrow
def draw_conn(ax, x1, y1, x2, y2, color='#334155', lw=1.6, text='', text_pos='top', style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=1.5, shrinkB=1.5), zorder=4)
    if text:
        tx = (x1 + x2)/2
        ty = (y1 + y2)/2 + (0.15 if text_pos=='top' else -0.18)
        ax.text(tx, ty, text, ha='center', va='center', fontsize=8.2, color='#0F172A', fontweight='bold', zorder=5)

# Utility: Draw a 3x3 kernel matrix grid
def draw_kernel_grid(ax, x, y, matrix, label, cell_size=0.19, font_size=7.2):
    ax.text(x + 1.5 * cell_size, y + 3 * cell_size + 0.10, label, ha='center', va='bottom',
            fontsize=8.5, fontweight='bold', color='#92400E')
    for r in range(3):
        for c in range(3):
            val = matrix[r][c]
            bg = '#FEF3C7' if val != 0 else '#FFFFFF'
            rect = patches.Rectangle((x + c * cell_size, y + (2 - r) * cell_size), cell_size, cell_size,
                                     facecolor=bg, edgecolor='#D97706', lw=0.9, zorder=4)
            ax.add_patch(rect)
            ax.text(x + (c + 0.5) * cell_size, y + (2.5 - r) * cell_size, str(val),
                    ha='center', va='center', fontsize=font_size, fontweight='bold', color='#78350F', zorder=5)

# =========================================================================
# PANEL (A): GLOBAL MACRO-ARCHITECTURE OF CAD-SPAN
# =========================================================================
draw_panel(ax, 0.25, 8.55, 19.5, 4.7,
           '(a) Global Macro-Architecture of Proposed CAD-SPAN Framework',
           'Cascade: 6 CAD-SPAB Blocks · Multi-Scale Hierarchical Concatenation · PixelShuffle x4')

# 1. Input Tensor
draw_block(ax, 0.5, 10.05, 1.45, 1.55, 'Low-Res Input', r'$I_{LR} \in \mathbb{R}^{3 \times H \times W}$',
           '[B, 3, H, W]', color=C_TENSOR, border=C_TENSOR_BORDER, title_size=10.5, sub_size=9, dim_size=8.2)

# 2. Shallow Feature Extractor
draw_conn(ax, 1.95, 10.82, 2.4, 10.82)
draw_block(ax, 2.4, 10.05, 1.6, 1.55, 'Shallow Conv', 'Conv3XC (3 to 48)\nReparameterizable',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=10, sub_size=8.5, dim_size=8.2)

# Arrow to Backbone
draw_conn(ax, 4.0, 10.82, 4.45, 10.82, text='$F_0$')

# 3. Cascaded CAD-SPAB Blocks (6 blocks)
block_xs = []
for i in range(1, 7):
    bx = 4.45 + (i - 1) * 1.4
    block_xs.append(bx)
    draw_block(ax, bx, 10.15, 1.15, 1.35, f'CAD-SPAB\n#{i}', f'$F_{{b{i}}}$',
               '[B, 48, H, W]', color=C_ATTN, border=C_ATTN_BORDER, title_size=9.2, sub_size=8.5, dim_size=7.6)
    if i < 6:
        draw_conn(ax, bx + 1.15, 10.82, bx + 1.4, 10.82)

# 4. Multi-Stage Hierarchical Aggregation (Taps from F0, Fb1, Fb5_2, Fb6)
# F0 Tap
ax.plot([4.22, 4.22, 13.3], [10.82, 12.35, 12.35], color='#4338CA', lw=1.6, linestyle='--', zorder=3)
ax.plot([13.3, 13.3], [12.35, 11.35], color='#4338CA', lw=1.6, linestyle='--', zorder=3)
ax.text(6.8, 12.5, r'Shallow Feature Tap $F_0$', fontsize=8.8, color='#3730A3', fontweight='bold')

# Fb1 Tap
ax.plot([block_xs[0] + 0.58, block_xs[0] + 0.58, 13.1], [10.15, 9.1, 9.1], color='#047857', lw=1.5, linestyle=':', zorder=3)
ax.plot([13.1, 13.1], [9.1, 10.25], color='#047857', lw=1.5, linestyle=':', zorder=3)
ax.text(7.6, 8.92, r'Hierarchical Stage Tap $F_{b1}$', fontsize=8.8, color='#065F46', fontweight='bold')

# Fb5_2 Tap
ax.plot([block_xs[4] + 0.58, block_xs[4] + 0.58, 13.55], [10.15, 9.5, 9.5], color='#B45309', lw=1.5, linestyle=':', zorder=3)
ax.plot([13.55, 13.55], [9.5, 10.25], color='#B45309', lw=1.5, linestyle=':', zorder=3)
ax.text(11.1, 9.62, r'Intermediate Tap $F_{b5\_2}$', fontsize=8.8, color='#92400E', fontweight='bold')

# Fb6 connection
draw_conn(ax, block_xs[5] + 1.15, 10.82, 13.1, 10.82, text='$F_{b6}$')

# Concat Box
draw_block(ax, 13.1, 10.2, 1.85, 1.25, 'Hierarchical Concat', r'$[F_0, F_{b6}, F_{b1}, F_{b5\_2}]$',
           '[B, 192, H, W]', color='#FEE2E2', border='#EF4444', title_size=9.2, sub_size=8.2, dim_size=7.8)

# 5. Fusion Conv (1x1)
draw_conn(ax, 14.95, 10.82, 15.3, 10.82)
draw_block(ax, 15.3, 10.2, 1.35, 1.25, r'Fusion $1\times 1$', 'Conv + SiLU',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=9.2, sub_size=8.2, dim_size=7.8)

# 6. Global Residual Addition
draw_op(ax, 17.0, 10.82, '+', r=0.22)
draw_conn(ax, 16.65, 10.82, 16.78, 10.82)
# Long skip from F0 to +
ax.plot([4.22, 4.22, 17.0, 17.0], [10.82, 12.75, 12.75, 11.04], color='#312E81', lw=1.6, linestyle='-.', zorder=3)
ax.text(10.2, 12.88, r'Global Feature Residual Skip ($+ F_0$)', fontsize=9.2, color='#312E81', fontweight='bold', ha='center')

# 7. PixelShuffle Upsampler
draw_conn(ax, 17.22, 10.82, 17.55, 10.82)
draw_block(ax, 17.55, 10.05, 1.35, 1.55, 'PixelShuffle\nUpsampler', r'Conv3x3 ($48\to 48$)' + '\nShuffle (r=4)',
           '[B, 3, 4H, 4W]', color=C_OP, border=C_OP_BORDER, title_size=9.2, sub_size=8.0, dim_size=7.6)

# Global Bicubic Skip Connection
draw_op(ax, 18.5, 9.15, '+', r=0.20)
ax.plot([1.22, 1.22, 18.5, 18.5], [10.05, 8.85, 8.85, 8.95], color='#0D9488', lw=1.5, linestyle='--', zorder=3)
ax.text(9.5, 8.7, r'Global Bicubic Residual Interpolation Skip: $+ \mathrm{Bicubic}(I_{LR})$', fontsize=9.0, color='#0F766E', fontweight='bold', ha='center')
ax.plot([18.3, 18.5], [10.05, 9.35], color='#334155', lw=1.4, zorder=3)

# Output Tensor
draw_block(ax, 18.0, 7.8, 1.55, 0.95, 'High-Res Output', r'$I_{SR} \in \mathbb{R}^{3 \times 4H \times 4W}$',
           '[B, 3, 4H, 4W]', color=C_GREEN, border=C_GREEN_BORDER, title_size=9.5, sub_size=8.5, dim_size=7.6)
draw_conn(ax, 18.5, 8.95, 18.5, 8.75)


# =========================================================================
# PANEL (B): MICRO-ARCHITECTURE OF CAD-SPAB BLOCK
# =========================================================================
draw_panel(ax, 0.25, 0.25, 13.3, 8.05,
           '(b) Micro-Architecture: Curvature-Aware Directional Swift Parameter-free Attention Block (CAD-SPAB)',
           'Dual-Differential Prior · Calibrated Origin-Symmetric Gate · Zero-Parameter Edge Steering')

# 1. Block Input
draw_block(ax, 0.5, 4.5, 1.3, 1.35, 'Block Input', r'$X_{in}$',
           '[B, 48, H, W]', color=C_TENSOR, border=C_TENSOR_BORDER, title_size=10.2, sub_size=9.2, dim_size=8.2)

# 2. Three Consecutive Conv3XC Layers with SiLU
draw_conn(ax, 1.8, 5.17, 2.15, 5.17)
draw_block(ax, 2.15, 4.35, 2.3, 1.65, '3x Conv3XC Cascade', 'Conv3XC + SiLU (#1)\nConv3XC + SiLU (#2)\nConv3XC + SiLU (#3)',
           '[B, 48, H, W]', color=C_CONV, border=C_CONV_BORDER, title_size=9.8, sub_size=8.5, dim_size=7.8)

# Output of 3x convs is H
draw_conn(ax, 4.45, 5.17, 4.8, 5.17)
draw_block(ax, 4.8, 4.7, 1.05, 0.95, r'Feature $H$', r'$H \in \mathbb{R}^{C \times H \times W}$',
           '[B, 48, H, W]', color='#FEF3C7', border='#D97706', title_size=9.8, sub_size=8.5, dim_size=7.8)

# Forking Point from H into 3 Branches
ax.plot([5.85, 6.2], [5.17, 5.17], color='#334155', lw=1.6)
ax.plot([6.2, 6.2], [1.75, 6.6], color='#334155', lw=1.6)

# -------------------------------------------------------------------------
# BRANCH 1: Temperature-Calibrated Origin-Symmetric Gating (Top)
# -------------------------------------------------------------------------
draw_conn(ax, 6.2, 6.6, 6.65, 6.6)
draw_block(ax, 6.65, 5.95, 3.8, 1.3,
           '1. Calibrated Origin-Symmetric Gate',
           r'$A_{sym}(H) = \frac{1}{2}\tanh(\tau \cdot H)$  with  $\tau^* = 0.502$' + '\n' +
           r'Analytical identity: $\sigma(x)-0.5 \equiv \frac{1}{2}\tanh(x/2)$',
           'Exact zero-crossing transition control',
           color=C_ATTN, border=C_ATTN_BORDER, title_size=9.6, sub_size=8.4, dim_size=7.8)

# -------------------------------------------------------------------------
# BRANCH 2: First-Order Directional Sobel Gradient Flux (Middle)
# -------------------------------------------------------------------------
draw_conn(ax, 6.2, 4.35, 6.65, 4.35)
draw_block(ax, 6.65, 3.65, 3.8, 1.4,
           '2. Directional Gradient Flux Operator',
           r'$G_x = K_x * H, \quad G_y = K_y * H$' + '\n' +
           r'$\|\nabla H\|_2 = \sqrt{G_x^2 + G_y^2 + \epsilon}$',
           'Captures Orthogonal Step Transitions',
           color=C_KERNEL, border=C_KERNEL_BORDER, title_size=9.6, sub_size=8.4, dim_size=7.8)

# Sobel Kernel Visual Grids
sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
draw_kernel_grid(ax, 8.9, 3.75, sobel_x, r'$K_x$', cell_size=0.17, font_size=6.2)
draw_kernel_grid(ax, 9.65, 3.75, sobel_y, r'$K_y$', cell_size=0.17, font_size=6.2)

# -------------------------------------------------------------------------
# BRANCH 3: Second-Order Isotropic Laplacian Curvature (Bottom)
# -------------------------------------------------------------------------
draw_conn(ax, 6.2, 1.75, 6.65, 1.75)
draw_block(ax, 6.65, 1.1, 3.8, 1.3,
           '3. Isotropic Curvature Operator',
           r'$\Delta H = |\nabla^2 H| = |K_L * H|$' + '\n' +
           'Captures Corner Extrema & Ridge Inflection',
           'Discrete 8-Neighbor Isotropic Stencil',
           color='#F0FDF4', border='#16A34A', title_size=9.6, sub_size=8.4, dim_size=7.8)

# Laplacian Kernel Visual Grid
laplacian_k = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
draw_kernel_grid(ax, 9.3, 1.2, laplacian_k, r'$K_L$', cell_size=0.17, font_size=6.2)

# -------------------------------------------------------------------------
# DUAL-DIFFERENTIAL SPATIAL MODULATOR FUSION
# -------------------------------------------------------------------------
# Merge Branch 2 & Branch 3 into Dual Differential Prior Phi(H)
ax.plot([10.45, 10.75], [4.35, 3.0], color='#334155', lw=1.5)
ax.plot([10.45, 10.75], [1.75, 3.0], color='#334155', lw=1.5)
draw_conn(ax, 10.75, 3.0, 11.0, 3.0)

draw_block(ax, 11.0, 2.25, 2.3, 1.5,
           r'Dual-Differential Prior $\Phi(H)$',
           r'$\Phi(H) = \tanh(\gamma_s \|\nabla H\|_2 + \gamma_l \Delta H)$' + '\n' +
           r'$\gamma_s^* = 0.008$ (Sobel Gain)' + '\n' +
           r'$\gamma_l^* = 0.001$ (Laplacian Gain)',
           '[B, 48, H, W]', color='#F1F5F9', border='#334155', title_size=9.2, sub_size=7.8, dim_size=7.4)

# -------------------------------------------------------------------------
# CURVATURE-MODULATED ATTENTION GATE SYNTHESIS
# -------------------------------------------------------------------------
# Shift by 1: (1 + Phi(H))
draw_op(ax, 12.2, 4.4, '+', r=0.20)
draw_conn(ax, 12.2, 3.75, 12.2, 4.2)
ax.text(12.6, 4.4, r'$\mathbf{1}$', fontsize=10.5, fontweight='bold', color='#1D4ED8', va='center')
ax.plot([12.5, 12.4], [4.4, 4.4], color='#1D4ED8', lw=1.5)

# Hadamard Modulation with Asym: A_tilde = Asym * (1 + Phi)
draw_op(ax, 12.2, 6.6, r'$\odot$', r=0.24, color='#FCE7F3', border='#EC4899', text_color='#BE185D', size=14)
draw_conn(ax, 10.45, 6.6, 11.96, 6.6, text=r'$A_{sym}$')
draw_conn(ax, 12.2, 4.6, 12.2, 6.36, text=r'$1 + \Phi(H)$', text_pos='right')

# Attention Gate Output
draw_block(ax, 11.3, 5.2, 1.8, 1.0,
           r'Modulated Gate $\widetilde{A}$',
           r'$\widetilde{A} = A_{sym} \odot (\mathbf{1} + \Phi(H))$' + '\n' +
           'Zero-parameter Edge Prior',
           '[B, 48, H, W]', color='#FDF2F8', border='#DB2777', title_size=8.8, sub_size=7.6, dim_size=7.2)

# Gated Residual Output: O = (H + Xin) * A_tilde
# Local Residual Addition: (H + Xin)
draw_op(ax, 4.8, 0.9, '+', r=0.18)
# Xin skip to local +
ax.plot([1.15, 1.15, 4.8, 4.8], [4.5, 0.9, 0.9, 0.72], color='#2563EB', lw=1.5, linestyle='--', zorder=3)
ax.text(2.6, 0.75, r'Local Input Residual $X_{in}$', fontsize=8.0, color='#1D4ED8', fontweight='bold')
# H to local +
ax.plot([5.32, 5.32, 4.8], [4.7, 2.5, 1.08], color='#D97706', lw=1.4, linestyle=':', zorder=3)

# Final Block Output
draw_block(ax, 11.4, 0.65, 1.9, 1.0,
           r'Block Output $O$',
           r'$O = (H + X_{in}) \odot \widetilde{A}$',
           '[B, 48, H, W]', color='#F0FDF4', border='#15803D', title_size=9.2, sub_size=8.0, dim_size=7.6)
ax.plot([4.98, 11.4], [0.9, 0.9], color='#2563EB', lw=1.5, zorder=3)

# Identity Fallback Callout Banner
fallback_box = FancyBboxPatch((0.5, 0.32), 12.8, 0.28, boxstyle="round,pad=0.02,rounding_size=0.06",
                              linewidth=1.2, edgecolor='#10B981', facecolor='#ECFDF5', zorder=3)
ax.add_patch(fallback_box)
ax.text(6.9, 0.46, r'Mathematical Identity Fallback Guarantee: When $\gamma_s = \gamma_l = 0$ and $\tau = 0.50$, $\Phi(H) \equiv 0$ and $\widetilde{A}(H) \equiv \sigma(H) - 0.5$ (Exact Baseline SPAB)',
        fontsize=8.0, fontweight='bold', color='#065F46', ha='center', va='center', zorder=4)


# =========================================================================
# PANEL (C): CONV3XC STRUCTURAL RE-PARAMETERIZATION MECHANISM
# =========================================================================
draw_panel(ax, 13.8, 0.25, 5.95, 8.05,
           '(c) Conv3XC: Structural Re-parameterization',
           'Training Multi-Branch -> Inference Single 3x3 Conv')

# Training Phase Sub-panel
train_box = FancyBboxPatch((14.0, 4.3), 5.55, 3.4, boxstyle="round,pad=0.03,rounding_size=0.10",
                           linewidth=1.3, edgecolor='#6366F1', facecolor='#F5F3FF', zorder=2)
ax.add_patch(train_box)
ax.text(14.2, 7.45, 'TRAINING PHASE (Multi-Branch Topology)', fontsize=9.2, fontweight='bold', color='#4338CA', zorder=3)

# Input
draw_block(ax, 14.15, 5.5, 0.95, 0.95, 'Input', '$C_{in}$', color=C_TENSOR, border=C_TENSOR_BORDER, title_size=8.5, sub_size=7.8)

# Shortcut branch (1x1 Conv)
ax.plot([15.1, 15.35, 15.35, 18.2, 18.2], [5.97, 5.97, 6.9, 6.9, 6.17], color='#4F46E5', lw=1.4, linestyle='--', zorder=3)
draw_block(ax, 15.8, 6.55, 1.8, 0.65, r'Shortcut: Conv $1\times 1$ ($C_{in} \to C_{out}$)', '', color='#EEF2FF', border='#4F46E5', title_size=7.8)

# Deep Bottleneck branch (1x1 -> 3x3 -> 1x1)
draw_conn(ax, 15.1, 5.97, 15.35, 5.97)
draw_block(ax, 15.35, 5.55, 1.1, 0.85, r'Conv $1\times 1$', '$C_{in} \to 2C$', color=C_CONV, border=C_CONV_BORDER, title_size=8.0, sub_size=7.2)
draw_conn(ax, 16.45, 5.97, 16.7, 5.97)
draw_block(ax, 16.7, 5.55, 1.1, 0.85, r'Conv $3\times 3$', '$2C \to 2C$', color=C_CONV, border=C_CONV_BORDER, title_size=8.0, sub_size=7.2)
draw_conn(ax, 17.8, 5.97, 18.05, 5.97)
draw_block(ax, 18.05, 5.55, 1.1, 0.85, r'Conv $1\times 1$', '$2C \to C_{out}$', color=C_CONV, border=C_CONV_BORDER, title_size=8.0, sub_size=7.2)

# Sum +
draw_op(ax, 18.2, 4.85, '+', r=0.18)
draw_conn(ax, 18.6, 5.55, 18.3, 4.95)
draw_conn(ax, 18.2, 5.15, 18.2, 5.03)
ax.text(18.9, 4.85, 'Training Output', fontsize=7.8, fontweight='bold', color='#4338CA', va='center')

# Offline Fusion Arrow
ax.annotate('', xy=(16.8, 3.8), xytext=(16.8, 4.2),
            arrowprops=dict(arrowstyle='->', color='#B91C1C', lw=2.4), zorder=4)
ax.text(16.8, 4.0, r'Offline Kernel Fusion: $\mathbf{W}_{eval} = \mathbf{W}_3 * \mathbf{W}_2 * \mathbf{W}_1 + \mathrm{Pad}(\mathbf{W}_{sk})$',
        ha='center', va='center', fontsize=8.2, fontweight='bold', color='#B91C1C',
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEE2E2", edgecolor="#EF4444", lw=1.0), zorder=5)

# Inference Phase Sub-panel
infer_box = FancyBboxPatch((14.0, 0.45), 5.55, 3.2, boxstyle="round,pad=0.03,rounding_size=0.10",
                           linewidth=1.3, edgecolor='#16A34A', facecolor='#F0FDF4', zorder=2)
ax.add_patch(infer_box)
ax.text(14.2, 3.35, 'INFERENCE PHASE (Reparameterized Topology)', fontsize=9.2, fontweight='bold', color='#15803D', zorder=3)

# Inference single 3x3 conv
draw_block(ax, 14.15, 1.7, 0.95, 0.95, 'Input', '$C_{in}$', color=C_TENSOR, border=C_TENSOR_BORDER, title_size=8.5, sub_size=7.8)
draw_conn(ax, 15.1, 2.17, 15.55, 2.17)
draw_block(ax, 15.55, 1.5, 2.5, 1.35,
           r'Single Equivalent Conv $3\times 3$',
           r'Weight: $\mathbf{W}_{eval} \in \mathbb{R}^{C_{out} \times C_{in} \times 3 \times 3}$' + '\n' +
           r'Bias: $\mathbf{B}_{eval} \in \mathbb{R}^{C_{out}}$',
           color='#DCFCE7', border='#16A34A', title_size=8.8, sub_size=7.8)
draw_conn(ax, 18.05, 2.17, 18.45, 2.17)
draw_block(ax, 18.45, 1.7, 0.95, 0.95, 'Output', '$C_{out}$', color=C_GREEN, border=C_GREEN_BORDER, title_size=8.5, sub_size=7.8)

ax.text(16.75, 0.85, 'Zero Multi-Branch Latency · Optimal GPU Cache Locality · Maximum Edge FPS',
        ha='center', va='center', fontsize=8.0, fontweight='bold', color='#14532D', zorder=3)

out_arch = r"C:\Users\anike\.gemini\antigravity\scratch\SPAN\methodology_architecture.png"
plt.savefig(out_arch, bbox_inches='tight', dpi=300)
plt.close()
print(f"Bigger, publication-grade Architecture Diagram saved to {out_arch}")
