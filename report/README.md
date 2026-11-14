# CAD-SPAN Research Paper & Report Assets

This directory contains all publication artifacts, LaTeX source files, compiled PDFs, 300 DPI architecture diagrams, and qualitative visual comparison figures for the **CAD-SPAN** research project.

---

## 📄 Primary Documents

* **`CAD_SPAN_IEEE_Paper.pdf`**: Publication-grade PDF adhering strictly to official IEEE conference standards (**strictly 6 pages**, including all 21 references).
* **`IEEE-conference-template-062824.tex`**: Official IEEE LaTeX source code (XeTeX / Tectonic compatible).
* **`IEEE-conference-template-062824.pdf`**: Direct engine compilation output.
* **`rendered_pages/`**: High-resolution PNG renders (`page_1.png` through `page_6.png`) for instant visual verification without a PDF viewer.

---

## 🎨 Publication Figures (300 DPI)

* **`methodology_architecture.png`** (Fig. 1): 25-Megapixel ($6000 \times 4200$) 3D isometric shaded architectural diagram faithfully matching the CVPR 2024 NTIRE winner SPAN paper style (Figure 2 in Wan et al.). Features authentic CAD-SPAN feature heatmaps extracted on Set5 `baby.png`, mathematical convolution re-parameterization, zero overlapping text, and verified arrow routing.
* **`combined_visual_comparison.png`** (Fig. 2): Visual comparisons on Set5 `butterfly` and Set14 `monarch` comparing Bicubic, baseline SPAN, CAD-SPAN+, and Ground Truth HR.
* **`ablation_curves.png`** (Fig. 3): 4-panel parameter sensitivity plots ($\gamma$ sweep, $\tau$ sweep, differential operator ablation, per-image gain bar chart).
* **`training_loss_curve.png`** (Fig. 4): Empirical 20-epoch DIV2K training convergence curve ($0.04624 \to 0.02899$).
* **`comprehensive_visual_gallery.png`**: Extended 4-case comparison gallery across Set5 and Set14.
* **`visual_comparison_cad.png`**, **`visual_comparison_monarch.png`**, **`visual_comparison_structural.png`**: High-frequency zoomed structural line and character crops (`crop_butterfly.png`, `crop_monarch.png`, `crop_zebra.png`, `crop_ppt3.png`).

---

## 🛠️ Figure Generation Scripts

To re-render any figure from scratch:

```bash
# Re-generate Figure 1 (Methodology Architecture Diagram)
python report/generate_architecture_diagram.py

# Re-generate Figure 3 (Parameter Ablation Graphs)
python report/generate_ablation_graphs.py

# Re-generate Figure 2 (Qualitative Comparisons)
python report/generate_visual_comparison_monarch.py
python report/generate_comprehensive_visual_gallery.py
```

---

## ⚡ How to Compile LaTeX to PDF

### Using Tectonic (Recommended, Single Command):
```bash
cd report
tectonic IEEE-conference-template-062824.tex
```

### Using Standard XeLaTeX:
```bash
cd report
xelatex IEEE-conference-template-062824.tex
xelatex IEEE-conference-template-062824.tex
```
