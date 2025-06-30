# 2025-qut-mit-capstone

## Unlocking Queensland’s Imagery Data

The project aims to explore and implement a semantic segmentation model using Mask2Former as an example of currently available ML/DL approaches, helping to speed up the extraction of building footprints from aerial images.

The data includes high-resolution RGB aerial imagery (10cm, 8-bit) captured across ~150,000 km² of Queensland annually, along with LiDAR-derived digital terrain models. These datasets vary in appearance across regions, as they are captured in different settings by multiple government authorities and cover areas with diverse vegetation types, building styles, and levels of urban development.

The project outcome, the imagery-based machine learning model, could potentially be used for flood impact modelling, planning development projects, estimating population density in natural disasters, and addressing places for police and ambulance services. It will serve as MVP for additional value-added products derived from the acquired imagery.

### Mask2Former

Mask2Former is a universal image segmentation architecture designed to handle semantic, instance, and panoptic segmentation within a unified framework. Its key innovation is the masked attention mechanism which restricts cross-attention to the foreground region of each predicted mask to improve convergence and segmentation accuracy by focusing on localised features. It also uses a Transformer decoder that processes multi-scale features in a round-robin manner to enhance the detection of small objects while maintaining computational efficiency ([Cheng et al., 2022](https://doi.org/10.1109/CVPR52688.2022.00135)).

## Notebooks Overview

This project is organised into four Jupyter notebooks, each handling a key stage in the building segmentation pipeline.  
Please review the table below to understand the role of each notebook and what data it expects.

| Notebook               | Purpose                                      | Input Folder(s)                  | Output Folder(s)              |
|------------------------|----------------------------------------------|----------------------------------|-------------------------------|
| 1. Data Preparation    | Preprocess and tile raw data, split datasets | Raw_data/                        | Preprocessed_data/, Input_data/ |
| 2. Mask2Former         | Train segmentation model (Mask2Former)       | Input_data/                      | outputs/                      |
| 3. Inference           | Run inference with trained model             | TIFF images, outputs/final_model/| Inference/                    |
| 4. [Additional Function]() | Refine labels with SAM (pseudo-labelling)    | outputs/, segment_anything       | pseudo_dataset/               |

- All notebooks require dependencies listed in `requirements.txt`.
- Edit data and model paths at the start of each notebook as needed.
- Mask2Former and Inference require a GPU for practical runtimes.
- Pseudo-labelling step needs the Segment Anything Model (SAM) library.

## Repository Structure

```
Raw_data/
├── building-outlines/        # Ground truth polygons (shapefiles/raster)
└── source-data/              # High-res imagery (GeoTIFF)

Preprocessed_data/
├── Tiled_Image/              # Image tiles (TIFF)
└── Tiled_Mask/               # Mask tiles (TIFF)

Input_data/
├── train_image/              # Training images (.npy)
├── train_mask/               # Training masks (.npy)
├── val_image/                # Validation images (.npy)
├── val_mask/                 # Validation masks (.npy)
├── test_image/               # Test images (.npy)
└── test_mask/                # Test masks (.npy)

outputs/
├── final_model/              # Final model checkpoint
├── model_iou/                # Model with best IoU
├── model_loss/               # Model with best loss
├── valid_preds/              # Validation predictions
├── Test_overlay_image/       # Test overlays
├── Test_predicted_masks/     # Predicted masks (test set)
├── loss.png                  # Loss curve
└── miou.png                  # mIoU curve

Inference/
├── Final_Inference_Mask/     # Merged final mask PNGs
├── Tiled_Inference_Image/    # Inference image tiles
├── Tiled_Inference_Mask/     # Predicted mask tiles
└── Vector_Inference_Mask/    # Vectorized (shapefile) outputs

pseudo_dataset/
├── train_image/              # (Same as Input_data)
├── train_mask/               # GT + pseudo-labeled masks
├── val_image/                # (Same as Input_data)
└── val_mask/                 # GT + pseudo-labeled masks

notebooks/
├── 1_data_preprocessing.ipynb
├── 2_training.ipynb
├── 3_inference.ipynb

requirements.txt
README.md
```

## Getting Started

### Setup

- **Python Version:** Recommend Python 3.9 ++ for full compatibility.

1. Clone this repo:  
   https://gitlab.com/stateimagery/wil-projects/2025-qut-mit-capstone.git

```
cd 2025-qut-mit-capstone
```

2. Install Python requirements:

```
pip install -r requirements.txt
```

### Data

- Place original aerial images (GeoTIFF format) in Raw_data/source-data/.
- Place ground truth building outlines (shapefiles) in Raw_data/building-outlines/.
- Supported data formats: TIFF or PNG (see 1_data_preprocessing.ipynb for details on format and folder structure).

### Notebooks

Order of Execution:

1. `1_data_preprocessing.ipynb` – Preprocess data and generate training samples.  
   - Adjust folder paths at the top of the notebook to match your environment.  
   - Raw data is not included in the repository due to size.

2. `2_training.ipynb` – Train the segmentation model.  
   - HuggingFace Transformers and PyTorch must be installed with CUDA support.  
   - Update notebook paths for your directory structure if different from the default.

3. `3_inference.ipynb` – Run inference on new data.   
   - The `buildingregulariser` package must be installed (via pip or as a local module).  
   - Update paths for image folder and model checkpoint as needed.

4. `4_Additional Function.ipynb` – Pseudo-labelling with SAM. (Optional)
   - **SAM is not on PyPI.** Install from the [official Meta AI Segment Anything repo](https://github.com/facebookresearch/segment-anything).  
   - This step is optional and only required if you want to experiment with label refinement or semi-supervised learning.

- **Path Configuration:**  
  All notebooks have cell(s) at the top where you should configure file and folder paths. Update these to fit your environment if needed.

- **Data & Runtime:**  
  Processing and model training may require significant RAM, GPU memory, and disk space.  
  Data is not provided due to size; use your own, or contact the repo owner for a small sample if available.


## Best Performance
The model has been tested on multiple data augmentations and training parameters. Kindly acccess this links below for best parameters and tuning logs:
- Best weights and parameters: [Drive link to Mask2Former trained Weights and Paramters](https://drive.google.com/file/d/1pP9paevhfn-Sj1vYajwhd3qmDGftnfIq/view?usp=share_link)
- Tuning logs: [Drive link to attempted parameters and its inference scores](https://docs.google.com/spreadsheets/d/1-P0ZV1DwtSAzaHuOk9ydc-mfKN7T7U8L6d2du3ajDa0/edit?usp=sharing)

**Best Augmentations**
- A.Resize(img_size[1], img_size[0]),
- A.Normalize(mean=ADE_MEAN, std=ADE_STD),
- A.HorizontalFlip(p=0.3),
- A.RandomBrightnessContrast(p=0.3),
- A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.3),
- A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2)

**Best Training parameters**
- Image size: 256x256
- Batch size: 16
- LR: 0.0001
- Epoch: 12

**Model Accuracy**

| **Metric**                    | **Train** | **Validation** | **Test** |
|------------------------------|-----------|----------------|----------|
| Mean Intersection Over Union | 90.87%    | 90.51%         | 88.39%   |
| Precision                    | —         | —              | 88.21%   |
| Recall                       | —         | —              | 90.46%   |
| F1 Score                     | —         | —              | 89.33%   |

**System and Runtime Configuration**
| **Component**        | **Description**                                                                 |
|----------------------|---------------------------------------------------------------------------------|
| **GPU**              | NVIDIA L4 GPU (24GB Memory)                                                     |
| **Task**             | Model Training and Inference                                                    |
| **Dataset**          | - 9 tiles of 10,000 × 10,000 pixels<br>- 80% used for training<br>- Tiled to 1000 × 1000 pixels and downsampled to 256 × 256<br>- 72 tiles of 256 × 256 used for training |
| **Training Time**    | ~16 minutes                                                                     |
| **Inference Dataset**| 4 tiles of 10,000 × 10,000 pixels                                               |
| **Inference Time**   | ~3 minutes   

**Model Convergence**

<img src="Readme_pngs/Model_Convergence.jpg" alt="Loss and Accuracy" width="600"/>


## Limitations

While the model achieved strong performance overall, there are a few limitations worth noting due to nature of the model:

1. **Computational Resources**  
   Training and inference rely heavily on GPU acceleration. Without a compatible GPU, runtimes can be slow and impractical for larger datasets.

2. **Limited Training Data**  
   The training dataset is relatively small, which can restrict the model’s ability to generalise across diverse roof types, materials, and urban layouts.

3. **Obscured Roof Structures**  
   In aerial imagery, some buildings may be partially or fully obscured by trees, solar panels, or other objects, leading to missed or incomplete segmentation.

4. **Large Building Inconsistency**  
   Very large structures, such as shopping centres or warehouses, may result in inconsistent or fragmented masks after postprocessing, particularly if tiled across multiple image sections.

5. **False Positives and Postprocessing Trade-offs**  
   The model sometimes misclassifies vehicles as buildings. While applying a minimum area threshold during postprocessing helps filter out false positives like cars, it may also remove legitimate small structures such as garden sheds or detached utility rooms.



## Recommendations
1. Segment Anything model for pseudo-labeling

- In this project, we briefly explored using the Segment Anything Model (SAM) to generate pseudo labels for false positive (FP) areas based on the Mask2Former model, such as swimming pools or cars incorrectly predicted as buildings. However, the results after retraining the model with enhanced masks were inconsistent. This might be due to the error distribution of the base model—earlier versions had significantly more false negatives (FN), so correcting only FP areas had limited impact. After fine-tuning and data augmentation, FP and FN became more balanced, allowing the SAM pseudo-labeling method show greater potential. 

- Due to time constraints and the need for further refinement, we did not include this strategy in our final workflow. However, future work could explore applying SAM-generated pseudo labels to FN regions, or combining corrections for both FP and FN. The effectiveness of each approach may depend on the error distribution: when FP errors are more dominant, pseudo labels for FP areas may be beneficial; conversely, when FN errors are more prevalent, applying SAM to FN regions may yield better results. For an example of generating pseudo labels on FP areas, please refer to Notebook 4: Additional Function.

2. Hyperparameters tuning
- A batch size of 16 was adopted despite Optuna initially suggesting a batch size of 32, which caused out-of-memory crashes. Optuna could be explored to further fine-tune other hyper parameters. 
- Image size 256×256 was adopted as 512×512 caused CUDA out-of-memory errors.
- Scheduler was not used as manual trials showed no mIOU improvement and removing it simplified training without performance loss.
- RandomBrightnessContrast (p=0.3) showed a consistent increase in average validation mIOU (≈0.9023 with vs. ≈0.8977 without), based on training records using the 60/20/20 data split. This was further supported by several A/B test cases.

3. Data Splitting Strategy
- In this project, we cropped the original 10,000×10,000 pixel images into smaller 1,000×1,000 pixel tiles. These tiles were then randomized and split into 60% for training, 20% for validation, and 20% for testing. As a result of this tiling and random arrangement, it becomes difficult to visually interpret the original large image during the testing phase.
- A more structured approach to data splitting can be considered in future iterations, as shown below. Instead of randomizing all tiles before splitting, we could first divide each 10,000×10,000 image into spatially coherent sections—60% for training, 20% for validation, and 20% for testing. Only the training and validation sections would be randomized. This would retain the spatial integrity of the test set, making visual analysis more meaningful.
- Additionally, this method would allow for more controlled sampling. For instance, instead of extracting only five non-overlapping 1,000×1,000 tiles from a 5,000×5,000 region (which may result in buildings being cut off), we could implement a sliding window approach to generate more overlapping tiles (e.g., 10 tiles). However, caution must be taken to avoid overfitting due to overlapping content in training samples.
<img src="Readme_pngs/Sample_Recommend_Datasplit.png" alt="sample" width="600"/>

## Acknowledgements

This project was developed as part of the QUT IFN711 Capstone in collaboration with the Queensland Government's Department of Natural Resources and Mines, Manufacturing and Regional and Rural Development.
This project makes use of the following open-source tools and repositories:

- [**Mask2Former**](https://github.com/facebookresearch/Mask2Former) – for the core semantic segmentation model architecture used in training and inference.
- [**Segment Anything Model (SAM)**](https://github.com/facebookresearch/segment-anything) – for exploring pseudo-labelling techniques to enhance training data quality.
- [**Building Regulariser**](https://github.com/DPIRD-DMA/Building-Regulariser) – for postprocessing predicted masks to produce clean, vector-based building outlines.

We appreciate the developers and maintainers of these projects for their valuable contributions to the open-source community.

## Notes on Confidentiality

This repository only includes non-confidential components and publicly reproducible elements of the project.
Due to intellectual property and confidentiality agreements with the Queensland Government, raw datasets, visual outputs (e.g., imagery, overlays, predicted masks), and certain implementation details cannot be disclosed.
All project materials included here comply with the terms of the WIL Student Confidentiality and IP Agreement (Assignment).

## Personal Contribution:

During this project, I was responsible for troubleshooting and contributing to the implementation and application of evaluation metrics—such as IoU, precision, and recall—to validate the model’s performance.
Additionally, I explored the use of the Segment Anything Model (SAM) as an auxiliary segmentation method. Although the results were inconsistent and it was not integrated into the final production pipeline, it is documented in Notebook 4 as an optional module for future research and experimentation.
