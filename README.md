# Leveraging Generative AI to Enhance Cine MRI Motility Estimation

This GitHub repository has been developed under GSoC 2025.

Contributor: Miguel Aenlle

Organization: Department of Biomedical Informatics, Emory University

Mentors: Dr. Babak Mahmoudi, Ozgur Kara

## Overview

Gastric motility assessment using cine MRI provides a non‐invasive, radiation‐free approach to quantifying peristaltic motion in the GI tract, yet its widespread adoption is constrained by the scarcity of high‐quality, annotated datasets. 

To overcome this limitation, we designed a modular, end‐to‐end pipeline that leverages generative image models to produce realistic synthetic MRI scans with corresponding segmentation masks, trains and evaluates segmentation networks, and enables automated motility quantification.

The full written report is available [here](https://docs.google.com/document/d/1DyGz7kE48id6yXDDv9JkjFSsO5zS3Njk_AjFDcRDXgg/edit?usp=sharing).

## Pipeline Modules
Our pipeline consists of the following modules: 
1. **Data preprocessing**: Prepare the base dataset of annotated MRI scans. 
2. **Generative model benchmarking**: Acquire representative samples and statistics from the generative models. We support SinGAN-Seg based synthesis and diffuse-gen based synthesis.
3. **Generative model image synthesis:** Generate augmented datasets of varying sizes for use in training. 
4. **Segmentation training/evaluation:** Train the segmentation model on the generated augmented dataset and evaluate its performance with a leave-one-out scheme per subject.
5. **Motility estimation and analysis:** Using trained models from Step 4 and the dataset, quantify motility by acquiring the stomach volume over time, dominant peristaltic frequency, and wave propagation speeds. 

## Dataset
We utilized a private, clinical Cine MRI dataset of the gastrointestinal (GI) tract, consisting of five healthy adult subjects. 

For each subject, 72 paired image and segmentation mask slices were captured across 132 temporal frames, yielding a total of 9,504 image–mask pairs. Manual annotations were available for only one temporal frame per subject; these single-frame labels served as the ground truth for both model training and evaluation.

## Getting started
During our experimentation, we used a Ubuntu-based workstation with 2 NVIDIA 4090 GPUs.

1. Clone the repository
```
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

2. Create and activate the conda 
```
conda env create -f environment.yml
conda activate motility-pipeline
```

## Running Pipeline Modules

Follow these steps to execute each module of the pipeline. You can copy–paste these commands directly into your terminal.

---

### 1. Data Preprocessing  
Prepare the base dataset of annotated MRI scans. 

```bash
cd GI/pipeline-modules/data-preprocessing
python prepare_dataset.py
````

---

### 2. Generative Modelling

#### 2.1 SinGAN-Seg

**Benchmarking**

Acquire representative samples and statistics from the SinGAN model.

```bash
cd GI/pipeline-modules/generative-modelling/singan
python train_singan_models_and_synthesize_images.py
python acquire_singan_statistics.py
```

**Dataset Preparation**

Generate augmented datasets of varying sizes for use in segmentation model training. 

```bash
python train_singan_models_and_synthesize_images.py
python generate_singan_augmented_loo_datasets.py
```

#### 2.2 Diffuse-gen

**Benchmarking**

Acquire representative samples and statistics from the SinGAN model.

```bash
cd GI/pipeline-modules/generative-modelling/diffusion
python train_diffusion_models_and_synthesize_images.py
python acquire_diffusion_statistics.py
```

**Dataset Preparation**

Generate augmented datasets of varying sizes for use in segmentation model training. 

```bash
python train_diffusion_models_and_synthesize_images.py
python generate_diffusion_augmented_loo_datasets.py
```

---

### 3. Segmentation Training & Evaluation

Train the segmentation model on the generated augmented dataset and evaluate its performance with a leave-one-out scheme per subject.

In this case, fine-tuning refers to providing additional subject data as the 

#### 3.1 No Fine-tuning

```bash
cd GI/pipeline-modules/segmentation
python train_eval_unets.py
```

> *Note: This uses the augmented dataset generated above.*

#### 3.2 SinGAN Fine-tuning

Fine-tuning refers to incorporating labeled image-mask pairs from the held-out subject used for validation. These labeled image-mask pairs

1. Go to the finetuning folder:

   ```bash
   cd GI/pipeline-modules/generative-modelling/singan/finetuning
   ```
2. **No augmentation**

   ```bash
   # In generate_finetuning_datasets.py:
   SELF_SUPERVISED = False
   FINETUNING_AUGMENTATION = False
   python generate_finetuning_datasets.py
   python format_finetuning_datasets.py
   ```
3. **With augmentation**

   ```bash
   # In generate_finetuning_datasets.py:
   SELF_SUPERVISED = False
   FINETUNING_AUGMENTATION = True
   python generate_finetuning_datasets.py
   python format_finetuning_datasets.py
   ```
4. **Self-supervised**

   ```bash
   python synthesize_maskless_images.py
   # Then in generate_finetuning_datasets.py:
   SELF_SUPERVISED = True
   # (optional) FINETUNING_AUGMENTATION = True
   python generate_finetuning_datasets.py
   python format_finetuning_datasets.py
   ```
5. Run segmentation again:

   ```bash
   cd GI/pipeline-modules/segmentation
   python train_eval_unets.py
   ```

   > *Use `constants.py` to select the model ID.*

---

### 4. Motility Estimation & Analysis

```bash
cd GI/pipeline-modules/motility-estimation
python extract_roberta_data.py
python generate_comparative_motility_estimation_results.py
```

> *Edit the `models` array in `generate_comparative_motility_estimation_results.py` to point at your trained segmentation models.*

Visualizations are available in `motility-comparison.ipynb`.

```bash
jupyter notebook motility-comparison.ipynb
```

## References
- This work builds upon [A graphical user interface of ML Toolbox for Medical Images](https://github.com/sarperyn/gsoc-2024) implemented by Sarper Yurtseven in Google Summer of Code 2024 and previous work from the Neuroinformatics and Intelligent Systems Laboratory (NISys Lab) at the Department of Biomedical informatics, Emory University
- The SinGAN-Seg code in this repository is from the paper [SinGAN-Seg: Synthetic training data generation for medical image segmentation
](https://arxiv.org/abs/2107.00471), with slight modifications for compatibility with the pipeline.
- The diffuse-gen paper in this code is from the paper [Using diffusion models to generate synthetic labeled data for medical image segmentation](https://link.springer.com/article/10.1007/s11548-024-03213-z), with slight modifications for compatibility with the pipeline.
