# AidFuse: Box–Body Dual-Reference Skeleton Fusion for Mobility-Aid Users

Code for the paper *"AidFuse: Box–Body Dual-Reference Skeleton Fusion for Robust Activity Recognition of Mobility-Aid Users in Retail Surveillance"* (J. A. Putra, N. Suciati, C. Fatichah).

AidFuse trains two skeleton networks with the same backbone: one on **box-referenced** skeletons (joints normalized by the person box) and one on **body-referenced** skeletons (origin at the mid-shoulder, unit = 3 × median shoulder width). At test time the class probabilities of both streams are averaged with equal weights.

## Repository layout

| Path | Content |
|---|---|
| `aidfuse/maa.py` | data windows, training/evaluation units, paired bootstrap (`compare`), fusion (`FUSE`, `QFUSE`), per-aid-type analysis |
| `aidfuse/augs.py` | augmentations: A1–A4 (standard), U1–U4 / O1–O3 (aid-specific simulations), OCC, ASM, MIX (re-implemented comparison methods), body reference `u5_upper_norm`, variants C1/C2 |
| `aidfuse/maa_safer.py` | SAFER-Activities conversion, protocols and evaluation |
| `aidfuse/maa_analysis.py` | pose-quality and body-proportion analyses |
| `notebooks/` | Google Colab notebooks used for all experiments (see below) |
| `paper/` | scripts that rebuild the tables and figures of the paper from the result CSVs |

### Names used in the code vs. the paper

| Code | Paper |
|---|---|
| `A0` | benchmark augmentation |
| `A4` | standard augmentation (box-referenced stream) |
| `A4+U5` | body-referenced stream (`U5` = body reference) |
| `FUSE:S3@A4\|S3b@A4+U5` | **AidFuse** |
| `FUSE:S3@A4\|S3@A4#1`, `FUSE:S3b@A4+U5\|S3b@A4+U5#1` | box + box and body + body ensembles (second member = next seed) |
| `QFUSE:...` | quality-weighted fusion (ablation) |
| `A4+OCC`, `A4+ASM`, `A4+MIX` | comparison methods |
| CS / CL / CG | cross-subject / cross-location / cross-group protocols of CARR |

## Experiments (notebooks)

| Notebook | Paper |
|---|---|
| `MAA_S1_baselines` | A0 baselines (ST-GCN, ST-GCN++, CTR-GCN, PoseC3D), Fig. 1b |
| `MAA_S2_S8_S10_analisis` | pose quality (Fig. 1a), body proportions, cost (Table 7) |
| `MAA_S3_augmentasi`, `MAA_S3b_penyempurnaan`, `MAA_S3c_augmentasi_observasi` | A4, body reference, aid-specific simulations (Table 6) |
| `MAA_S4_ARC` | C1, C2 and OCC / ASM / MIX on CS and CG |
| `MAA_S6_CL` | A4 and body reference on CL |
| `MAA_S9_SAFER`, `MAA_S9b_SAFER`, `MAA_S9c_SAFER` | training on SAFER-Activities |
| `MAA_analisis_fusi` | AidFuse, ensembles and all paired tests on CARR (Tables 2, 4, 5) |
| `MAA_S9_analisis_SAFER`, `MAA_S9_kontrol_ensemble` | AidFuse and ensembles on SAFER (Table 3) |
| `MAA_S5_per_alat_bantu` | per aid type (Fig. 3) |

The notebooks are designed for Google Colab: open the same notebook in several sessions and use *Run all*; units of work are shared through lock files on Google Drive, and interrupted units resume from their checkpoint. Each notebook writes its own copy of the modules (`%%writefile`), exactly as used for the reported runs. Notebook text is partly in Indonesian.

## Requirements and data

* Python 3.10+, PyTorch 2.x, NumPy, pandas, scikit-learn, matplotlib (`requirements.txt`).
* The CARR benchmark code (data loading, pose extraction and GCN models) from https://github.com/januaradip/CARR, commit `1193346`; the notebooks clone it automatically.
* CARR dataset: Harvard Dataverse, https://doi.org/10.7910/DVN/CY24UB (place it under `MyDrive/CARR`).
* SAFER-Activities: https://huggingface.co/datasets/SAFER-Activities/SAFER-Activities (CC BY-NC-SA 4.0; request access on Hugging Face).

## Reproducing the tables and figures

```bash
cd paper
RESULTS=/path/to/MyDrive/CARR/maa/results python build_data.py   # -> data.json
python make_tables.py                                            # -> tab_*.tex
RESULTS=/path/to/MyDrive/CARR/maa/results python figs.py         # -> fig/
```

## Citation

Please cite the paper and the CARR dataset (doi:10.7910/DVN/CY24UB) if you use this code. If you use the SAFER-Activities experiments, also cite:

> Lamsal D, Wickramatilake P, Moonrinta J, Ekpanyapong M, Dailey MN. SAFER-Activities: A dataset for smart assessment of fall events and routine activities. arXiv:2609.08038, 2026.

## License

MIT (code). The datasets keep their own licenses.
