# Plant Disease Classification: confidence cascade pilot

Code, saved predictions and result tables for the pilot experiment of the article
**"Towards a Multi-Agent AI System for Plant Disease Classification"** (B. Kuttymbek, Astana IT University).

The pilot compares, on the Cassava field dataset, two ImageNet-pre-trained agents (MobileNetV2, ResNet50),
soft voting, and a **confidence cascade**: MobileNetV2 answers alone when its top softmax probability is at least
a threshold `tau`; otherwise the image is passed to ResNet50 and the two outputs are averaged.

> **Scope.** This is a reduced pilot, not the full planned system. It uses 10 % of the Cassava training and
> validation splits (the full test split), frozen backbones, a head trained for 5 epochs, three seeds, and no
> Grad-CAM verifier. PlantVillage, PlantDoc, the third agent (DenseNet121) and the background-replacement test
> are **not** part of this repository. See the Limitations section of the article.

## Repository contents

| Path | Content |
|---|---|
| `01_pilot_v1.ipynb` | The whole pilot in order: data, agents, three seeds, analyses, statistics, stacking and calibration check, latency, environment |
| `latency_e2e.py` | End-to-end CPU latency script (written by the notebook; kept here for reference) |
| `pilot_artifacts/preds_seed{0,1,2}.npz` | Validation/test probabilities of both agents and the predictions of every configuration |
| `pilot_artifacts/pilot_v1_per_seed.csv` | Macro-F1, `tau` and escalation rate `rho` per seed |
| `pilot_artifacts/stats_v1.csv` | Paired bootstrap intervals, exact McNemar tests, Holm-adjusted p-values |
| `pilot_artifacts/preds_v2_seed*.npz`, `pilot_v2_per_seed.csv`, `calibration.csv`, `stats_v2.csv` | Exploratory check: temperature scaling, stacking, calibrated cascade |
| `pilot_artifacts/latency_run{1,2,3}.json`, `latency_e2e.json` | Three latency runs and their pooled summary |
| `pilot_artifacts/environment.json` | Software and hardware of the runtime |
| `tau_sensitivity.pdf`, `confusion_cascade.pdf` | Figures of the article |
| `requirements.txt` | Pinned package versions |

## How to reproduce

1. Open `01_pilot_v1.ipynb` in Google Colab with a **T4 GPU** runtime (Python 3.13).
2. Run the cells from top to bottom. The Cassava data are downloaded automatically through TensorFlow Datasets
   (about 1.3 GB on the first run).
3. Section 9 starts separate CPU-only processes (two threads) and therefore needs the models saved in Section 6
   (`models_tmp/`). If the runtime was reset, run the notebook from the top again.

Outside Colab: `pip install -r requirements.txt` and run the notebook with a GPU for training.
Random seeds are 0, 1 and 2 (head initialisation and data shuffling). The bootstrap uses 1,000 resamples drawn with a
NumPy generator seeded with `1000 + seed`.

## Results at a glance

Test macro-F1 on the Cassava test split (1,885 images), mean ± sample SD over three seeds:

| Configuration | Macro-F1 |
|---|---|
| MobileNetV2 | 0.503 ± 0.030 |
| ResNet50 | 0.557 ± 0.018 |
| Soft voting | 0.553 ± 0.025 |
| Cascade (no verifier) | 0.534 ± 0.033 |

Latency values are in `pilot_artifacts/latency_e2e.json` (three runs, 300 images each, CPU with two threads).
Bootstrap intervals describe the finite test sample of one trained model; they do not describe variation
between retrainings.

## Data

The pilot uses the `cassava` dataset of TensorFlow Datasets (version 0.1.0): 9,430 labelled leaf images in five
classes (CBB, CBSD, CGM, CMD, healthy), predefined splits of 5,656 / 1,889 / 1,885 images (train / validation / test).
The data are **not** stored in this repository. The TFDS catalogue page does not state a licence; please check the
terms of the original source (Kaggle competition page of the iCassava 2019 challenge) before redistributing the images.

Cite the dataset as: E. Mwebaze, T. Gebru, A. Frome, S. Nsumba, J. Tusubira, "iCassava 2019 Fine-Grained Visual
Categorization Challenge," arXiv:1908.02900, 2019.

## Citation and licence

Cite this repository using `CITATION.cff` (GitHub's "Cite this repository" button). Code is released under the MIT licence
(`LICENSE`).
