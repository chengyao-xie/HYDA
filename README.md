# Experimental Environment

- **nibabel**: 5.3.2  
- **scipy**: 1.15.3  
- **pymeshlab**: 2023.12.post3  
- **h5py**: 3.14.0  
- **tqdm**: 4.67.1  
- **torch**: 2.7.1  
- **numpy**: 1.24.4  
- **torch_optimizer**: 0.3.0  

---

# Data Extraction

**Dataset source:**  
https://auckland.figshare.com/articles/dataset/NeurIPS_2022_Datasets/21397377/7  

From the following datasets:

- ADHD  
- PPMI  
- Neurocon  
- Taowu_Preprocessed  

We selected:

- 10 control subjects  
- 10 disease subjects  

from each dataset.

## Surface Projection

Functional MRI (fMRI) volumes were projected onto the cortical surface using `mri_vol2surf`.

Three processing configurations were applied:

1. **fsaverage5 → onavg-ico32**
2. **fsaverage5 → fsaverage5**
3. **fsaverage6 → fsaverage6**

### Processing Command

```bash
mri_vol2surf \
  --mov [voxel fMRI path] \
  --hemi [lh/rh] \
  --projfrac 0.5 \
  --interp trilinear \
  --o [output surface functional data path] \
  --trgsubject [target template, e.g., fsaverage6] \
  --reg $FREESURFER_HOME/average/mni152.register.dat
```

---

# Experimental Pipeline

After projection, we obtained:

- `lh.func.mgh` (left hemisphere)
- `rh.func.mgh` (right hemisphere)

## Step 1: Hierarchical Data Construction

Using `getinput.py`, we generated `data.h5`, which stores:

- Multi-level vertices
- Face topology
- Functional signals
- Inter-level mapping relationships

## Step 2: Model Training

Using `train.py` (which calls `data_utils.py`):

- Load `data.h5`
- Preprocess hierarchical data
- Train prediction model
- Predict future brain activity

---

# MAE Metric Explanation

In this study, raw fMRI intensity values (MGH format) were directly used as prediction targets.

Because raw intensities typically range from **thousands to tens of thousands**, absolute error metrics such as:

- MAE
- RMSE

naturally produce large numerical values.

This does **not** indicate excessive model error, but reflects the high physical magnitude of the prediction target.

## Scale-Invariant Metrics

We additionally report:

- **R²**
- **MAPE**

These metrics are scale-independent and measure goodness-of-fit to the dynamic variation of the signal.

Since raw amplitude varies substantially across subjects, absolute error alone cannot fully reflect model performance.

**R²** directly measures the proportion of variance explained by the model and provides a more stable and fair evaluation of neural dynamics prediction.

Across experiments, the method achieved consistently high R² values, demonstrating effective modeling of individual dynamic patterns.

---

# Detailed Experimental Workflow

## (1) Data Extraction

- Dataset source:  
  https://auckland.figshare.com/articles/dataset/NeurIPS_2022_Datasets/21397377/7  

- Selected 10 control and 10 disease subjects from:
  - ADHD  
  - PPMI  
  - Neurocon  
  - Taowu_Preprocessed  

- Projected fMRI to cortical surface using `mri_vol2surf`.

- Three processing strategies:
  1. fsaverage5 → onavg-ico32  
  2. fsaverage5 → fsaverage5  
  3. fsaverage6 → fsaverage6  

---

## (2) Per-Subject Multi-Level Cortical Hierarchy Construction

For each subject:

- Input:
  - Subject-specific fMRI data (`lh.func.mgh`, `rh.func.mgh`)
  - Standard fsaverage cortical geometry (`lh.white`, `rh.white`)

- Output:
  - Five-resolution multi-scale cortical hierarchy

Each level contains:

- Vertex coordinates  
- Face topology  
- Downsampling mappings  
- Aggregated fMRI signals  

Each subject’s hierarchy is generated independently.

---

## (3) Per-Subject, Per-Level Future fMRI Prediction

After hierarchy construction:

- Models were trained independently at:
  - Level 1  
  - Level 2  

### Training Setup

- Per-subject training (no cross-subject sharing)
- Sliding window sampling
- Task: **4 → 2 forecasting**
  - Input: 4 time steps
  - Output: Next 2 time steps

### Training Details

- 500 epochs per subject per level
- Early stopping applied
- Training logs saved
