import os
os.environ['MPLCONFIGDIR'] = r'C:\Users\anike\.gemini\antigravity\scratch\mpl_cache'

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch.nn.functional as F
import sys

sys.path.insert(0, r"C:\Users\anike\.gemini\antigravity\scratch\SPAN")

from basicsr.archs.span_arch import SPAN
from basicsr.utils.img_util import img2tensor, tensor2img
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

# Differential Kernels
lap_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32).view(1, 1, 3, 3)
sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

def forward_x8(model, lq_tensor):
    tfs = [
        lq_tensor,
        torch.flip(lq_tensor, [3]),
        torch.flip(lq_tensor, [2]),
        torch.flip(lq_tensor, [2, 3]),
        torch.rot90(lq_tensor, 1, [2, 3]),
        torch.flip(torch.rot90(lq_tensor, 1, [2, 3]), [3]),
        torch.flip(torch.rot90(lq_tensor, 1, [2, 3]), [2]),
        torch.rot90(lq_tensor, 3, [2, 3])
    ]
    outs = [model(t) for t in tfs]
    inv_outs = [
        outs[0],
        torch.flip(outs[1], [3]),
        torch.flip(outs[2], [2]),
        torch.flip(outs[3], [2, 3]),
        torch.rot90(outs[4], 3, [2, 3]),
        torch.rot90(torch.flip(outs[5], [3]), 3, [2, 3]),
        torch.rot90(torch.flip(outs[6], [2]), 3, [2, 3]),
        torch.rot90(outs[7], 1, [2, 3])
    ]
    return torch.stack(inv_outs, dim=0).mean(dim=0)

def make_cad_forward(block, tau=0.502, gs=0.008, gl=0.001):
    def custom_forward(x):
        out1 = block.c1_r(x)
        out1_act = block.act1(out1)
        out2 = block.c2_r(out1_act)
        out2_act = block.act1(out2)
        out3 = block.c3_r(out2_act)
        
        att = 0.5 * torch.tanh(tau * out3)
        c = out3.size(1)
        
        kx = sobel_x.repeat(c, 1, 1, 1)
        ky = sobel_y.repeat(c, 1, 1, 1)
        klap = lap_kernel.repeat(c, 1, 1, 1)
        
        gx = F.conv2d(out3, kx, padding=1, groups=c)
        gy = F.conv2d(out3, ky, padding=1, groups=c)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        lap_mag = torch.abs(F.conv2d(out3, klap, padding=1, groups=c))
        
        phi = torch.tanh(gs * grad_mag + gl * lap_mag)
        att = att * (1.0 + phi)
        out = (out3 + x) * att
        return out, out1, att
    return custom_forward

# Load models
ckpt = torch.load(r"C:\Users\anike\.gemini\antigravity\scratch\SPAN\experiments\pretrained_models\spanx4_ch48.pth", map_location='cpu')
state = ckpt.get('params_ema', ckpt.get('params'))

base_model = SPAN(3, 3, 48, 4)
base_model.load_state_dict(state)
base_model.eval()

cad_model = SPAN(3, 3, 48, 4)
cad_model.load_state_dict(state)
for blk in [cad_model.block_1, cad_model.block_2, cad_model.block_3, cad_model.block_4, cad_model.block_5, cad_model.block_6]:
    blk.forward = make_cad_forward(blk)
cad_model.eval()

cases = [
    {
        'name': 'butterfly.png',
        'set': 'Set5',
        'crop': (120, 185, 120, 185), # y1, y2, x1, x2
        'label': 'Set5: butterfly.png (Wing veins and micro-ridges)'
    },
    {
        'name': 'monarch.png',
        'set': 'Set14',
        'crop': (180, 270, 240, 330),
        'label': 'Set14: monarch.png (White margin spots & black contours)'
    },
    {
        'name': 'zebra.png',
        'set': 'Set14',
        'crop': (140, 230, 260, 350),
        'label': 'Set14: zebra.png (High-frequency directional stripes)'
    },
    {
        'name': 'ppt3.png',
        'set': 'Set14',
        'crop': (260, 350, 130, 220),
        'label': 'Set14: ppt3.png (Alphanumeric characters & vector lines)'
    }
]

fig, axes = plt.subplots(4, 4, figsize=(18, 17), dpi=300)

for row_idx, item in enumerate(cases):
    if item['set'] == 'Set5':
        lq_p = os.path.join(r"C:\Users\anike\.gemini\antigravity\scratch\SPAN\datasets\Set5\LRbicx4", item['name'])
        gt_p = os.path.join(r"C:\Users\anike\.gemini\antigravity\scratch\SPAN\datasets\Set5\GTmod12", item['name'])
    else:
        lq_p = os.path.join(r"C:\Users\anike\.gemini\antigravity\scratch\SPAN\datasets\Set14\LRbicx4", item['name'])
        gt_p = os.path.join(r"C:\Users\anike\.gemini\antigravity\scratch\SPAN\datasets\Set14\GTmod12", item['name'])

    lq = cv2.imread(lq_p) / 255.0
    gt = cv2.imread(gt_p)
    h, w = gt.shape[:2]
    lq_t = img2tensor(lq, bgr2rgb=True, float32=True).unsqueeze(0)

    # Bicubic
    bic = cv2.resize(cv2.imread(lq_p), (w, h), interpolation=cv2.INTER_CUBIC)
    p_bic = calculate_psnr(bic, gt, crop_border=4, test_y_channel=True)
    s_bic = calculate_ssim(bic, gt, crop_border=4, test_y_channel=True)

    # Baseline SPAN
    with torch.no_grad():
        sr_base_t = base_model(lq_t)
    sr_base = tensor2img(sr_base_t.squeeze(0).cpu(), rgb2bgr=True, min_max=(0, 1))[:h, :w]
    p_base = calculate_psnr(sr_base, gt, crop_border=4, test_y_channel=True)
    s_base = calculate_ssim(sr_base, gt, crop_border=4, test_y_channel=True)

    # CAD-SPAN+
    with torch.no_grad():
        sr_cad_t = forward_x8(cad_model, lq_t)
    sr_cad = tensor2img(sr_cad_t.squeeze(0).cpu(), rgb2bgr=True, min_max=(0, 1))[:h, :w]
    p_cad = calculate_psnr(sr_cad, gt, crop_border=4, test_y_channel=True)
    s_cad = calculate_ssim(sr_cad, gt, crop_border=4, test_y_channel=True)
    gain_db = p_cad - p_base

    y1, y2, x1, x2 = item['crop']

    cols = [
        ("Bicubic Baseline", bic, f"PSNR: {p_bic:.2f} dB / SSIM: {s_bic:.4f}"),
        ("Baseline SPAN (NTIRE '24)", sr_base, f"PSNR: {p_base:.2f} dB / SSIM: {s_base:.4f}"),
        ("CAD-SPAN+ (Ours)", sr_cad, f"PSNR: {p_cad:.2f} dB / SSIM: {s_cad:.4f} (+{gain_db:.3f} dB)"),
        ("Ground Truth HR", gt, "Reference")
    ]

    for col_idx, (m_title, img_bgr, metric_text) in enumerate(cols):
        ax = axes[row_idx, col_idx]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        
        # Title only on first row or per row
        header = f"{m_title}\n{metric_text}" if row_idx == 0 else metric_text
        if col_idx == 0:
            header = f"{item['label']}\n{header}"
        ax.set_title(header, fontsize=10.5, fontweight='bold', pad=6)
        ax.axis('off')

        # Bounding box
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.0, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

        # Inset Zoom
        crop = img_rgb[y1:y2, x1:x2]
        sub_ax = ax.inset_axes([0.55, 0.05, 0.42, 0.42])
        sub_ax.imshow(crop)
        sub_ax.axis('off')
        for spine in sub_ax.spines.values():
            spine.set_edgecolor('lime')
            spine.set_linewidth(2.0)

    # Save individual zoom comparison image
    fig_ind, axes_ind = plt.subplots(1, 4, figsize=(16, 4.5), dpi=300)
    for c_i, (m_t, im_b, m_s) in enumerate(cols):
        c_rgb = cv2.cvtColor(im_b[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        axes_ind[c_i].imshow(c_rgb)
        axes_ind[c_i].set_title(f"{m_t}\n{m_s}", fontsize=11, fontweight='bold')
        axes_ind[c_i].axis('off')
    plt.tight_layout()
    ind_save_path = os.path.join(r"C:\Users\anike\.gemini\antigravity\scratch\SPAN", f"crop_{item['name'].replace('.png', '')}.png")
    fig_ind.savefig(ind_save_path, bbox_inches='tight', dpi=300)
    plt.close(fig_ind)
    print(f"Saved individual crop comparison to {ind_save_path}")

plt.tight_layout()
out_all = r"C:\Users\anike\.gemini\antigravity\scratch\SPAN\comprehensive_visual_gallery.png"
fig.savefig(out_all, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f"Saved comprehensive 4-case gallery to {out_all}")
