# Cross-Dataset Generalization of Underwater Instance Segmentation

---

## 0. Interactive Streamlit Dashboard

The results and visualizations described in the following sections, as well as example model predictions for each model, can be explored interactively via the [Streamlit dashboard](https://underwater-segmentation-transfer-xtdrr5mjedczky9mqtwg6p.streamlit.app/) attached to this repository (implemented in `app.py`). We recommend exploring it while reading the following sections to aid in understanding the project. The dashboard has four interactive views: (1) a cross-dataset results explorer with configurable metrics and category spaces; (2) t-SNE/UMAP feature space projections colored by dataset and class, with silhouette score overlays; (3) best and worst prediction visualizations per evaluation cell; and (4) training curve comparisons across models. All data is loaded dynamically from the `results` directory.

## 1. Introduction

Human trash/debris is a growing threat to marine ecosystems, and automated detection via instance segmentation is a promising tool for robotic cleanup operations. Most work in this space trains and evaluates models on a single dataset, implicitly assuming that in-domain performance translates to real-world deployment. But underwater environments vary enormously — in water clarity, lighting, camera hardware, depth, and debris composition — so a model that performs well on its training distribution may fail in a new setting.

This project systematically studies cross-dataset generalization of Mask R-CNN between two publicly available underwater trash datasets: **TrashCan 1.0** and **SeaClear**. These datasets represent distinct underwater visual domains — deep-sea ROV footage versus shallow-water diver and ROV footage — with incompatible annotation schemes and substantially different imaging conditions. The central questions are: (1) how severe is the cross-dataset performance gap, (2) what drives it, and (3) can mitigation strategies like pooled training and category coarsening recover performance?


## 2. Background and Dataset Pivot

This project was originally proposed around the Common Objects Underwater (COU) dataset, which was released in February 2025. During the baseline model phase, we trained 21 Mask R-CNN variants on COU and achieved a best mask mAP of 0.749 on test (with TTA), compared to the paper's reported 0.773 — a gap likely attributable to a self-contradictory configuration in the paper (claiming batch size 128 on a 24GB L4 GPU, which is physically impossible for Mask R-CNN). Even training with Detectron2 itself, the paper's framework, only reached 0.703.

More critically, we discovered that COU's annotation quality is far worse than initially apparent. Beyond the 2.7% of degenerate masks identified during preprocessing, many images contain what appear to be unfiltered Segment Anything Model (SAM) proposals — masses of low-confidence masks, missing annotations for clearly visible objects, and unreliable segmentation boundaries throughout. This undermines both training and evaluation reliability.

We thus pivoted away from COU and toward **TrashCan 1.0** (7,212 deep-sea images, 22 classes) and **SeaClear** (8,610 shallow-water images, 40 classes). Both datasets use COCO-format annotations with substantially better quality. The class overlap is excellent: 20 of 22 TrashCan classes have clear SeaClear equivalents. Many of SeaClear's finer-grained categories (split by material, e.g., `bottle_plastic` / `bottle_glass`) merge cleanly into TrashCan's coarser object-type categories (e.g., `trash_bottle`), though some have no obvious TrashCan equivalent. Importantly, while COU and TrashCan come from the same lab (University of Minnesota), SeaClear is independently produced, reducing potential data correlation from shared cameras or capture environments.


## 3. Datasets

**TrashCan 1.0** contains 7,212 images at 480x270 or 480x360 resolution captured by remotely operated vehicles (ROVs) in deep-sea environments at relatively high resolution. Annotations span 22 fine-grained categories including trash types (bottle, bag, can, rope, net, tarp, etc.), marine organisms (fish, starfish, shells), and the ROV itself. The imaging conditions are relatively controlled — consistent camera hardware, generally clearer water — though object appearances vary due to depth, encrustation, and deformation.

**SeaClear** contains 8,610 images at 1920x1080 resolution (which were downscaled to 480x270 for fairer comparison against TrashCan) captured by both divers and ROVs in shallow coastal waters. Its 40 categories are more granular, splitting objects by material (e.g., `can_metal`, `can_other`) in addition to object type. The visual domain is substantially different from TrashCan: murkier water, variable lighting from surface conditions, more diverse camera perspectives, and different debris compositions typical of coastal versus deep-sea environments.

The two datasets' category spaces are incompatible, so we developed a **generic category hierarchy** that maps both into common evaluation spaces:

- **TC20** (20 classes): A contiguous re-indexing of TrashCan's categories excluding two classes (`animal_crab`, `animal_eel`) that have no SeaClear equivalent. SeaClear's 40 categories map into this space via a manually constructed mapping. Generally, this happens partially by collapsing SeaClear's material granularity (e.g. `bottle_plastic` and `bottle_glass` collapse to `trash_bottle`) but also by completely dropping 12 of SeaClear's categories (and ignoring their corresponding annotations). This allows for greater prediction specificity than the category spaces below, but we prefer those below since they retain all the annotations from the original datasets.
- **Coarse** (5 classes): Groups by extraction difficulty — `rov`, `organic` (plants, animals, branches), `trash_easy` (small extractable items like bottles and cans), `trash_entangled` (flexible items with snagging risk, like rope, netting, and tarp), and `trash_heavy` (large/heavy items requiring human review). 
- **Ternary** (3 classes): `rov`, `non_trash`, `trash`.
- **Binary** (2 classes): `non_trash`, `trash`.

This hierarchy is implemented generically in a `src.category_groups` module. The evaluation, visualization, and Streamlit app layers all consume category mappings from this single source, so any model can be evaluated at any level of the hierarchy without code changes.


## 4. Data Leakage and the Chunksplit Protocol

The dataset split provided by the authors of TrashCan is approximately 85% train and 15% validation (no test set). Both TrashCan and SeaClear contain sequences of temporally adjacent video frames — consecutive images in lexicographic filename order are generally consecutive frames in the same video segment and are often nearly identical.

A naive random split allows near-duplicate frames to appear in both training and test data without mitigation, which significantly inflates in-domain performance and gives a misleading picture of how well the model actually generalizes, even within the same dataset. This is particularly problematic for a study of cross-dataset generalization, since overstated in-domain baselines would cause us to overestimate the cross-domain gap. A naive random split could also deflate the cross-domain performance since it incentivizes the models to memorize scenes instead of learning generic features, though this part won't really play an obvious role in the results sections below.

To address this, we implemented a **chunk-based splitting** procedure (`resplit_by_chunks.py`). Images are sorted lexicographically, grouped into consecutive chunks of 50 frames, and entire chunks are randomly assigned to train/val/test splits (60/20/20). This vastly reduces the probability that temporally adjacent frames land in both the training set and the val/test sets and consequently the severity of the data leakage from near-duplicate frames. Another obvious grouping strategy would be to assign all frames from a single video to one of the splits; however, this turns out to lead to very poorly stratified split (where, for example, all or nearly all labels from certain classes appear in only one part of the split), especially in the case of SeaClear, whose images can only be confidently grouped into a very small number of videos using the filenames and metadata available in the dataset. 

The impact is dramatic. With the original (leaky) TrashCan split, a TC-trained model achieves mask mAP 0.508 on TC validation. With the chunksplit, the same model architecture achieves only 0.185 on TC chunksplit test — a 60%+ drop that reflects the removal of artificially easy near-duplicate test images. All results reported below use the chunksplit protocol, providing a fairer and more conservative measure of both in-domain performance and cross-domain gaps.


## 5. Model Architecture

The model is **Mask R-CNN** with a **ResNet-50 + FPN** backbone, initialized from COCO-pretrained weights (`maskrcnn_resnet50_fpn_v2` from torchvision). We chose Mask R-CNN for three reasons: (1) it allows direct comparison with published baselines on both datasets; (2) among the architectures evaluated in the COU paper, Mask R-CNN and Mask2Former benefited most from COCO pretraining, suggesting their backbone representations transfer well — a relevant property for studying cross-dataset generalization; and (3) the two-stage design (class-agnostic RPN followed by classification and segmentation) may help localize where domain shift manifests.

Key architectural decisions:

- **Backbone freezing**: Stem and `layer1` are frozen (the first two stages of ResNet-50). These encode low-level features likely shared across COCO and underwater domains. `layer2`, `layer3`, and `layer4` remain trainable.
- **Batch normalization**: All backbone BN layers are frozen to preserve the robust normalization statistics learned from COCO pretraining. In the FPN and ROI heads, BN is replaced with **GroupNorm** (32 groups), which is better suited to the small batch sizes (8) used in training.
- **Custom anchor generator**: Five FPN levels with anchor sizes (16, 32, 64, 128, 256) pixels and aspect ratios (0.5, 1.0, 2.0) at each level, designed for the range of object sizes in underwater imagery.

Training uses **SGD** with momentum 0.9, learning rate 0.005, weight decay 1e-4, cosine annealing with linear warmup (1 epoch), and mixed-precision (AMP). Multi-scale training samples from image sizes (256, 288, 320, 352, 384) with max dimension 704. **Copy-paste augmentation** is enabled for the primary models, pasting 1–3 randomly scaled and rotated object instances onto training images (with probability 0.5). Training runs for 50 epochs, and the checkpoint with the best validation mask mAP is selected.


## 6. Experimental Setup

We trained models in six configurations, varying the training data and the category space:

- **TC-only** and **SC-only**: Trained on a single dataset in its native category space (22 or 40 classes).
- **TC (TC20)** and **SC (TC20)**: Trained on a single dataset but in the TC20 common category space (20 classes).
- **TC (coarse)** and **SC (coarse)**: Trained on a single dataset in the coarse category space (5 classes).
- **Pooled (TC20)**: Trained on the union of TrashCan and SeaClear in TC20 space.
- **Pooled (coarse)**: Trained on the union of both datasets in coarse space.

Every model is evaluated on both TC chunksplit test and SC chunksplit test. For cross-dataset evaluation, ground truth annotations are remapped to the coarsest common category space between the model's training space and the target dataset's native space. COCO evaluation metrics (mAP, mAP@50, mAP@75) are computed separately for both mask and box predictions.

We also trained two augmentation variants as exploratory experiments: **SC+grayscale** (random grayscale with p=0.4, aimed at reducing color-based domain cues) and **SC+overlay** (a custom overlay augmentation). These were evaluated in both native and TC20 spaces.


## 7. Results

### 7.1 The Domain Gap

The table below shows mask mAP@50 for the primary models evaluated in TC20 space.

| Training Data | → TC Test | → SC Test |
|:--|:-:|:-:|
| TC-only | 0.329 | 0.025 |
| SC-only | 0.021 | 0.456 |
| Pooled | 0.347 | 0.465 |

Cross-domain performance is catastrophic: TC→SC drops from 0.329 to 0.025 (13×), and SC→TC drops from 0.456 to 0.021 (22×). The models are essentially non-functional on the other dataset.

Pooled training largely resolves this. The pooled model achieves 0.347 on TC test and 0.465 on SC test — matching or slightly exceeding the in-domain baselines on both test sets — despite seeing roughly half as many examples from each domain during training. This suggests the domain gap is not an inherent architectural limitation but a data distribution problem that pooling directly addresses.

### 7.2 Impact of Category Granularity

The same pattern holds in coarse space, with higher absolute numbers.

| Training Data | → TC Test | → SC Test |
|:--|:-:|:-:|
| TC-only (coarse) | 0.443 | 0.011 |
| SC-only (coarse) | 0.009 | 0.518 |
| Pooled (coarse) | 0.449 | 0.540 |

Moving from TC20 (20 classes) to coarse (5 classes) improves in-domain TC performance from 0.329 to 0.443 and in-domain SC from 0.456 to 0.518. The cross-domain numbers remain catastrophic for single-dataset models but the pooled model in coarse space reaches 0.449 on TC and 0.540 on SC — substantially above the TC20 pooled results.

### 7.3 Data Leakage Impact

The effect of the chunksplit protocol is worth emphasizing with a direct comparison on in-domain TC evaluation.

| Split | TC→TC mask mAP | TC→TC mask mAP@50 |
|:--|:-:|:-:|
| Original (leaky) | 0.508 | 0.811 |
| Chunksplit | 0.185 | 0.329 |

The 60%+ drop confirms substantial data leakage in the original split and validates the chunksplit protocol as a fairer evaluation.

### 7.4 Full Evaluation Matrix (mask mAP / mask mAP@50)

For completeness, the full cross-evaluation matrix across all model variants:

**TC20 space:**

| Model | → TC Test | → SC Test |
|:--|:-:|:-:|
| TC (TC20) | 0.184 / 0.305 | 0.017 / 0.022 |
| SC (TC20) | 0.006 / 0.011 | 0.318 / 0.478 |
| Pooled (TC20) | 0.206 / 0.347 | 0.294 / 0.465 |
| SC+grayscale (TC20) | 0.005 / 0.010 | 0.315 / 0.487 |
| SC+overlay (TC20) | 0.011 / 0.017 | 0.215 / 0.352 |

**Coarse space:**

| Model | → TC Test | → SC Test |
|:--|:-:|:-:|
| TC (coarse) | 0.231 / 0.443 | 0.008 / 0.011 |
| SC (coarse) | 0.009 / 0.019 | 0.279 / 0.518 |
| Pooled (coarse) | 0.233 / 0.449 | 0.292 / 0.540 |

### 7.5 Augmentation Experiments

Neither grayscale augmentation nor the overlay augmentation improved cross-domain transfer. SC+grayscale achieved 0.010 mAP@50 on TC test (vs. 0.011 for vanilla SC), and SC+overlay achieved 0.008 (worse). Both maintained near-baseline in-domain SC performance. The failure of grayscale augmentation to help — despite the intuition that it would reduce color-based domain cues — suggests that the domain gap is not primarily driven by color differences.


## 8. Feature Space Analysis

To understand *why* cross-dataset transfer fails, we extracted features from the ResNet-50 backbone and analyzed their structure using dimensionality reduction and silhouette scores.

For each model, we extracted two types of features from both TC and SC test images: **image-level** embeddings (global average pooling of the `layer4` output, yielding a 2048-d vector per image) and **ROI-level** embeddings (7×7 ROI-aligned crops for each detected object, pooled to 2048-d). These were projected to 2D using both t-SNE and UMAP for visualization, and silhouette scores were computed to quantify clustering quality.

### Image-level silhouette scores (chunksplit test, coarse space):

| Model | Dataset silhouette | Class silhouette |
|:--|:-:|:-:|
| TC model | 0.291 | 0.100 |
| SC model | 0.332 | 0.083 |
| Pooled (coarse) | 0.207 | 0.074 |

### Image-level silhouette scores (chunksplit test, TC20 space):

| Model | Dataset silhouette | Class silhouette |
|:--|:-:|:-:|
| TC model | 0.279 | -0.036 |
| SC model | 0.351 | -0.066 |
| Pooled (TC20) | 0.196 | -0.054 |

Dataset silhouette scores are consistently positive and substantial (0.20–0.35), meaning the backbone learns to separate images by dataset regardless of the model variant. This holds even when silhouette is computed within individual classes (the coarse-space class silhouettes are positive because the computation is per-class, and dataset clustering persists at the per-class level). This confirms that the domain gap is driven by image-level visual differences — water conditions, camera characteristics, overall appearance — rather than by confusion between object categories.

The pooled model shows reduced dataset clustering (0.196–0.207 vs. 0.279–0.351 for single-dataset models), consistent with its improved cross-dataset performance. It learns more domain-invariant features, though some dataset separation persists.

ROI-level silhouette scores are much lower (0.03–0.08 for dataset, near-zero for class), suggesting that at the individual-object level, features are less dataset-distinctive — the domain gap is primarily a global, image-level phenomenon.


## 9. Infrastructure

### Evaluation Pipeline

A single orchestrator script (`compile_results.py`) runs the full evaluation and analysis pipeline in four stages: (1) cross-dataset evaluation for all model-target pairs, (2) feature extraction and projection, (3) optional prediction visualizations (best/worst per cell), and (4) summary compilation into a single JSON. The pipeline is idempotent — it skips already-computed results unless `--force` is passed — and supports selective execution via `--skip-evals`, `--skip-features`, or `--compile-only` flags.

All model keys, target datasets, feature split definitions, and comparison groups are defined in a centralized `registry.py`. The evaluation, visualization, and app layers all import from this registry, so adding a new model variant requires only a single registry entry.

### Category Remapping

The `src.category_groups` module implements the full category hierarchy. Functions like `get_scheme()`, `detect_source_space()`, and `coarsest_common_space()` allow the evaluation code to automatically determine the appropriate evaluation space for any model-target pair and remap ground truth annotations accordingly. This is used throughout the cross-dataset evaluation, feature visualization, and prediction visualization pipelines.


## 10. Conclusions

Cross-dataset transfer in underwater instance segmentation fails catastrophically: models trained on one dataset achieve near-zero mAP on the other, a 10–22× drop from in-domain performance. Feature space analysis confirms this gap is visual (driven by image-level domain differences like water conditions and camera characteristics) rather than semantic (category confusion).

Two simple mitigations are effective. **Pooled training** on the union of both datasets recovers or exceeds in-domain performance on both test sets without any architectural changes, demonstrating that the gap is a data distribution problem rather than an architectural limitation. **Category coarsening** further improves performance: reducing from 20 fine-grained classes to 5 coarse classes raises pooled model performance from 0.347/0.465 to 0.449/0.540 (TC/SC mAP@50).

The data leakage issue in the original TrashCan split is also a practical finding: the chunksplit protocol reveals that published in-domain numbers for this dataset are likely substantially inflated, which has implications for any work building on TrashCan.

## 11. Future Work

The techniques described in 6 above (random grayscale and synthetic CCTV-style text overlays) for closing the cross-domain generalization gap were largely unsuccessful. However, we don't believe that attempting to close the gap is futile. The techniques we tried so far all more or less domain-agnostic -- we think it's certainly worth trying some domain-specific techniques such as FDA/FSR (see below).

Some other possible directions for future work:
- Attention-based architectures like Mask2Former may handle cross-domain features more flexibly than the convolutional RPN in Mask R-CNN. 
- Training deployment-oriented models in coarsened category spaces (e.g., binary trash/non-trash detection) tailored to specific cleanup scenarios could yield practically useful performance levels for real-time robotic applications. 
- Finally, expanding the pooled training set with additional underwater datasets would test whether the benefits of pooling scale with data diversity.

### FDA/FSR

Domain adaptation techniques such as Fourier Domain Adaptation (FDA) and Fourier Style Restitution (FSR) could help to close the remaining visual domain gap without requiring pooled training data. This repository implements FDA/FSR for training-time and/or test-time augmentation of the low-frequency disks of images in Fourier space, but we haven't found time yet to train or test models using these implementations. 
