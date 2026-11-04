# ABLATION STUDY: XQuAD & MLQA (M1 - M5)

## 1. Table 1: Results on Benchmark XQuAD (In-Domain Transfer)

| Cấu hình | Thành phần Loss | Vietnamese (XQuAD-vi)<br>EM / F1 (%) | Arabic (XQuAD-ar)<br>EM / F1 (%) | Hindi (XQuAD-hi)<br>EM / F1 (%) |
| :--- | :--- | :---: | :---: | :---: |
| **M1** | **Zero-Shot Baseline**<br>*(Chỉ Stage 1 SQuAD EN)* | 46.22 / 63.64 | 43.53 / 57.03 | 44.54 / 60.21 | 
| **M2** | **Global OT only**<br>($\mathcal{L}_{ot} + \mathcal{L}_{reg}$) | 43.28 / 61.62 | 40.34 / 54.84 | 33.70 / 49.62 | 
| **M3** | **OT + Span Projection**<br>($\mathcal{L}_{ot} + \mathcal{L}_{span} + \mathcal{L}_{reg}$) | 46.64 / 67.42 | 43.95 / 62.42 |52.61 / 66.87 |
| **M4** | **OT + Span + Dynamic Margin**<br>*(Anneal .0 \rightarrow 0.3$)* | 49.30 / 69.31 | 46.96 / 65.19 | 51.80 / 65.70 | 
| **M5** | **Full Model: Static Margin (Ours)**<br>*(Static =1.0$)* | **49.70 / 69.73**<br>*(Multi: **49.98 $\pm$ 0.25** / **70.17 $\pm$ 0.38**)* | **47.31 / 65.26**<br>*(Multi: **46.97 $\pm$ 0.39** / **64.92 $\pm$ 0.30**)* | **52.94 / 66.94**<br>*(Multi: **52.91 $\pm$ 0.21** / **67.09 $\pm$ 0.15**)* 

---

## 2. Table 2: Results on Benchmark MLQA (Out-of-Domain Transfer)

| Cấu hình | Thành phần Loss | Vietnamese (MLQA-vi)<br>EM / F1 (%) | Arabic (MLQA-ar)<br>EM / F1 (%) | Hindi (MLQA-hi)<br>EM / F1 (%) | 
| :--- | :--- | :---: | :---: | :---: | 
| **M1** | **Zero-Shot Baseline**<br>*(Chỉ Stage 1 SQuAD EN)* | 46.05 / 67.21 | 37.26 / 56.38 | 42.88 / 60.40 | 
| **M2** | **Global OT only**<br>($\mathcal{L}_{ot} + \mathcal{L}_{reg}$) | 42.89 / 64.08 | 32.72/52.54 | 33.12 / 52.10 | 
| **M3** | **OT + Span Projection**<br>($\mathcal{L}_{ot} + \mathcal{L}_{span} + \mathcal{L}_{reg}$) | 44.35 / 65.33 | 35.16 / 54.02 | 44.27 / 60.35| 
| **M4** | **OT + Span + Dynamic Margin**<br>*(Anneal)* | 45.23 / 67.33 | 36.6/55.83 | 45.87/61.69 | 
| **M5** | **Full Model: Static Margin (Ours)**<br>*(Static =1.0$)* | **46.00 / 67.60**<br>*(Multi: **45.89 $\pm$ 0.20** / **67.44 $\pm$ 0.20**)* | **37.59 / 56.50**<br>*(Multi: **37.43 $\pm$ 0.17** / **56.54 $\pm$ 0.07**)* | **45.87 / 61.69**<br>*(Multi: **45.80 $\pm$ 0.08** / **61.72 $\pm$ 0.05**)* | 

---
